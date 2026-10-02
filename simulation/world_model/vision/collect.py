"""Collect paired real grasp/lift sequences and matched-posture object placements."""
import argparse
import copy
import json
import pathlib
import time

import cv2
import mujoco
import numpy as np

from environment import EpisodeLayout, PickPlaceEnv, load_config
from data_collection.scripted_policy import ScriptedPickPlace
from world_model.prepare import P_COLUMNS, A_COLUMNS
from .data import provenance, write_json

CONTROL_FIELDS = ("target_pos", "step_start", "q_des", "base_quat", "yaw", "gripper_open", "posture")


def snapshot(env):
  # INTEGRATION includes warmstart, activation, controls and other integration state.
  spec = mujoco.mjtState.mjSTATE_INTEGRATION
  state = np.empty(mujoco.mj_stateSize(env.model, spec))
  mujoco.mj_getState(env.model, env.data, state, spec)
  return {"state": state, "controller": {k: copy.deepcopy(getattr(env.controller, k)) for k in CONTROL_FIELDS},
          "reward": copy.deepcopy(env.reward.__dict__), "step_count": env.step_count,
          "success_steps": env._success_steps}


def restore(env, saved):
  mujoco.mj_setState(env.model, env.data, saved["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
  for k, v in saved["controller"].items():
    setattr(env.controller, k, copy.deepcopy(v))
  env.reward.__dict__.update(copy.deepcopy(saved["reward"]))
  env.step_count, env._success_steps = saved["step_count"], saved["success_steps"]
  mujoco.mj_forward(env.model, env.data)


def approach(env, seed, replay_actions=None, on_step=None):
  actions = []
  policy = ScriptedPickPlace(env, np.random.default_rng(seed))
  for i in range(180 if replay_actions is None else len(replay_actions)):
    action = policy.act() if replay_actions is None else np.asarray(replay_actions[i])
    _, _, terminated, truncated, _ = env.step(action)
    mujoco.mj_forward(env.model, env.data)
    actions.append(action.tolist())
    if on_step:
      on_step(i)
    if terminated or truncated:
      raise RuntimeError("approach ended before pre-grasp source; inspect layout/seed")
    if replay_actions is None and policy.phase == "grasp":
      return actions
  if replay_actions is not None:
    return actions
  raise RuntimeError("could not reach the pre-grasp pose within 180 steps")


def sequence(close, hold_steps, lift_steps, lift_action):
  grip = -1.0 if close else 1.0
  return [[0.0, 0.0, 0.0, 0.0, grip]] * hold_steps + [
    [0.0, 0.0, lift_action, 0.0, grip]] * lift_steps


def record(env, key, scene, split, history, action=None, info=None, reward=0):
  obs = env._observation()
  grasp, obstacle, table = env._check_contacts()
  return {"key": key, "scene": scene, "split": split, "frames": history[-64:],
          "p": obs["proprio"].tolist(), "object_xyz": obs["state"][:3].tolist(),
          "object_quat": obs["state"][3:7].tolist(), "held": bool(grasp),
          "grasped_during_step": bool((info or {}).get("grasped", grasp)),
          "obstacle_contact": bool(obstacle), "table_contact": bool(table),
          "obstacle_contact_during_step": bool((info or {}).get("obstacle_contact", obstacle)),
          "table_contact_during_step": bool((info or {}).get("table_contact", table)),
          "reward_stage": (info or {}).get("stage", env.reward.stage),
          "reward": float(reward), "reward_components": (info or {}).get("reward_components", {}),
          "action_from_previous": action, "time": float(env.data.time)}


def collect_scene(root, manifest, scene, split, seed, anchor, args, cfg):
  base = json.loads(pathlib.Path(args.layout).read_text())
  common_actions, first_p = None, None
  start = time.monotonic()
  for placement in ("under", "offset"):
    layout_dict = copy.deepcopy(base)
    layout_dict["pick"] = (anchor + ([0, 0] if placement == "under" else [args.offset, 0])).tolist()
    layout = EpisodeLayout.from_dict(layout_dict)
    env = PickPlaceEnv(cfg)
    try:
      env.reset(seed=seed, layout=layout)
      history = []
      folder = root / "episodes" / scene / placement
      folder.mkdir(parents=True, exist_ok=True)

      def frame(name):
        path = folder / f"{name}.jpg"
        if not cv2.imwrite(str(path), env.render(args.camera)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, cfg.data.jpeg_quality]):
          raise RuntimeError(f"could not save frame {path}")
        return path.relative_to(root).as_posix()

      common_actions = approach(env, seed, common_actions,
                                lambda i: history.append(frame(f"context_{i:04d}")))
      mujoco.mj_forward(env.model, env.data)
      saved = snapshot(env)
      source_p = env._observation()["proprio"].copy()
      if first_p is None:
        first_p = source_p
      else:
        error = float(np.max(np.abs(source_p - first_p)))
        if error > args.pair_tolerance:
          raise RuntimeError(f"{scene}: placements changed starting robot p by {error:.6g}; "
                             "this scene is not a controlled position pair (try another seed/offset)")
      source = len(manifest["states"])
      manifest["states"].append(record(env, f"{scene}/{placement}/source", scene, split, history))
      np.savez_compressed(folder / "source.npz", integration_state=saved["state"],
                          **{k: np.asarray(v) for k, v in saved["controller"].items()})
      write_json(folder / "replay.json", {"seed": seed, "layout": layout_dict,
                 "approach_actions": common_actions, "source_step": saved["step_count"],
                 "source_p": source_p.tolist(), "config": cfg})
      for branch in ("close_lift", "open_lift"):
        restore(env, saved)
        p_error = float(np.max(np.abs(env._observation()["proprio"] - source_p)))
        actual = snapshot(env)["state"]
        state_error = float(np.max(np.abs(actual - saved["state"])))
        if p_error > 1e-5 or state_error > 1e-5:
          raise RuntimeError(f"source restore mismatch: p={p_error}, integration={state_error}")
        ids, branch_history = [source], history.copy()
        actions = sequence(branch == "close_lift", args.hold_steps, args.lift_steps, args.lift_action)
        for step, action in enumerate(actions, 1):
          _, reward, terminated, truncated, info = env.step(action)
          mujoco.mj_forward(env.model, env.data)
          branch_history.append(frame(f"{branch}_{step:04d}"))
          ids.append(len(manifest["states"]))
          manifest["states"].append(record(env, f"{scene}/{placement}/{branch}/{step}",
                    scene, split, branch_history, action, info, reward))
          if terminated or truncated:
            raise RuntimeError(f"{scene}/{placement}/{branch} ended at {step}; no complete rollout")
        r = {"id": f"{scene}/{placement}/{branch}", "scene": scene, "split": split,
             "placement": placement, "branch": branch, "states": ids, "actions": actions,
             "restore_p_error": p_error, "restore_integration_error": state_error}
        manifest["rollouts"].append(r)
        final, initial = manifest["states"][ids[-1]], manifest["states"][source]
        print(f"  {placement:6s} {branch:10s} rise="
              f"{100*(final['object_xyz'][2]-initial['object_xyz'][2]):+.2f} cm "
              f"held={final['held']} restore={state_error:.1e}", flush=True)
    finally:
      env.close()
  print(f"{scene} {split} anchor={anchor.round(3).tolist()} elapsed={time.monotonic()-start:.1f}s", flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--run", required=True, type=pathlib.Path)
  p.add_argument("--config", default="configs/grade_e.yml")
  p.add_argument("--layout", default="configs/grade_e_layout.json")
  p.add_argument("--train-scenes", type=int, default=12)
  p.add_argument("--val-scenes", type=int, default=4)
  p.add_argument("--test-scenes", type=int, default=4)
  p.add_argument("--seed", type=int, default=34)
  p.add_argument("--camera", choices=("static", "wrist"), default="static")
  p.add_argument("--hold-steps", type=int, default=8)
  p.add_argument("--lift-steps", type=int, default=16)
  p.add_argument("--lift-action", type=float, default=1.0)
  p.add_argument("--offset", type=float, default=0.10)
  p.add_argument("--pair-tolerance", type=float, default=1e-3)
  args = p.parse_args()
  if min(args.train_scenes, args.val_scenes, args.test_scenes, args.hold_steps, args.lift_steps) < 1:
    p.error("scene counts and sequence lengths must be positive")
  if args.run.exists() and any(args.run.iterdir()):
    p.error("run directory is not empty; use a new run name to preserve the experiment")
  if not 0 < args.lift_action <= 1 or args.offset < 0.08 or args.pair_tolerance <= 0:
    p.error("lift-action must be in (0,1], offset >= .08 m, pair-tolerance > 0")
  cfg = load_config(args.config, overrides={"robot": {"posture_noise": 0.0}})
  if not cfg.control.yaw.enabled:
    p.error("this experiment uses five-dimensional actions; enable control.yaw")
  args.run.mkdir(parents=True, exist_ok=True)
  manifest = {"schema": "vision_consequences_v1", "camera": args.camera, "clip_frames": 64,
              "clip_padding": "repeat_first_observed_frame", "control_hz": cfg.control.hz,
              "proprio_columns": P_COLUMNS, "action_columns": A_COLUMNS,
              "config": cfg, "settings": {k: str(v) if isinstance(v, pathlib.Path) else v
                                          for k, v in vars(args).items()},
              "provenance": {**provenance(), "mujoco": mujoco.__version__},
              "states": [], "rollouts": [], "complete": False}
  rng = np.random.default_rng(args.seed)
  # Entire scene groups are split before any branches or images are generated.
  regions = {"train": (0.14, 0.22), "val": (0.225, 0.245), "test": (0.105, 0.125)}
  n = 0
  try:
    for split, count in (("train", args.train_scenes), ("val", args.val_scenes), ("test", args.test_scenes)):
      for _ in range(count):
        anchor = np.array([rng.uniform(*regions[split]), rng.uniform(-0.26, -0.21)])
        collect_scene(args.run, manifest, f"scene_{n:04d}", split, args.seed+n, anchor, args, cfg)
        n += 1
        write_json(args.run / "manifest.json", manifest)
    manifest["complete"] = True
  except Exception as error:
    manifest["collection_error"] = repr(error)
    raise
  finally:
    write_json(args.run / "manifest.json", manifest)
  print(f"saved {n} scene groups, {len(manifest['rollouts'])} rollouts, "
        f"{len(manifest['states'])} states to {args.run}; next: world_model.vision.audit", flush=True)


if __name__ == "__main__":
  main()
