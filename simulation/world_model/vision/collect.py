"""Collect paired real grasp/lift sequences and matched-posture object placements."""
import argparse
import copy
import json
import pathlib
import shutil
import time

import cv2
import mujoco
import numpy as np

from environment import EpisodeLayout, PickPlaceEnv, load_config
from data_collection.scripted_policy import ScriptedPickPlace
from world_model.prepare import P_COLUMNS, A_COLUMNS
from .data import provenance, write_json

CONTROL_FIELDS = ("target_pos", "step_start", "q_des", "base_quat", "yaw", "gripper_open", "posture")


class SceneRejected(RuntimeError):
  """A physics/source check failed; resample only this scene group."""


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


def approach(env, seed, replay_actions=None, on_step=None, source_clearance=None):
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
      raise SceneRejected("approach ended before pre-grasp source; inspect layout/seed")
    if replay_actions is None and ((source_clearance is None and policy.phase == "grasp") or
        (source_clearance is not None and policy.phase == "descend" and
         env.data.site_xpos[env.controller.site_id][2] <= env.object_rest_z + source_clearance)):
      return actions
  if replay_actions is not None:
    return actions
  raise SceneRejected("could not reach the pre-grasp pose within 180 steps")


def sequence(close, hold_steps, lift_steps, lift_action):
  grip = -1.0 if close else 1.0
  return [[0.0, 0.0, 0.0, 0.0, grip]] * hold_steps + [
    [0.0, 0.0, lift_action, 0.0, grip]] * lift_steps


def scene_parameters(rng, index, args):
  if args.profile == "controlled":
    return {"offset_xy": [args.offset, 0], "yaw_action": 0, "height_action": 0,
            "lift_action": args.lift_action, "side_xy": [0, 0]}
  # A reference group every four scenes keeps successful grasp contrasts represented.
  reference = index % 4 == 0
  category = "far" if reference else ("near", "edge", "far")[index % 3]
  radius = float(rng.uniform(*{"near": (.008, .018), "edge": (.025, .045), "far": (.08, .12)}[category]))
  angle = float(rng.uniform(-np.pi, np.pi))
  side_angle = float(rng.uniform(-np.pi, np.pi))
  side = float(rng.uniform(.15, .30))
  return {"offset_category": category, "offset_xy": [radius*np.cos(angle), radius*np.sin(angle)],
          "yaw_action": 0.0 if reference else float(rng.uniform(-.18, .18)),
          "height_action": 0.0 if reference else float(rng.choice([0, .05, .10])),
          "lift_action": args.lift_action if reference else float(rng.uniform(.5, 1.0)),
          "side_xy": [side*np.cos(side_angle), side*np.sin(side_angle)]}


def branches(args, parameters):
  lift = parameters["lift_action"]
  result = {name: sequence(close, args.hold_steps, args.lift_steps, lift)
            for name, close in (("close_lift", True), ("open_lift", False))}
  if getattr(args, "decision_branches", False):
    result.update({name: sequence(close, args.hold_steps, args.lift_steps, 0.0)
                   for name, close in (("close_hold", True), ("open_hold", False))})
  if args.profile == "varied":
    side = [a.copy() for a in result["close_lift"]]
    release = [a.copy() for a in result["close_lift"]]
    for i in range(args.hold_steps + args.lift_steps//2, len(side)):
      side[i][:2] = parameters["side_xy"]
      release[i][-1] = 1.0
      release[i][2] = 0.0
    result.update(close_side_lift=side, close_lift_release=release)
    # Source stays above contact so near placements do not alter the matched initial robot state.
    result = {name: [[0, 0, -.25, 0, 1] for _ in range(6)] + actions for name, actions in result.items()}
  return result


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


def collect_scene(root, manifest, scene, split, seed, anchor, args, cfg, parameters=None):
  parameters = parameters or scene_parameters(np.random.default_rng(seed), 0, args)
  base = json.loads(pathlib.Path(args.layout).read_text())
  common_actions, first_p = None, None
  start = time.monotonic()
  for placement in ("under", "offset"):
    layout_dict = copy.deepcopy(base)
    layout_dict["pick"] = (anchor + ([0, 0] if placement == "under" else parameters["offset_xy"])).tolist()
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
                                lambda i: history.append(frame(f"context_{i:04d}")),
                                source_clearance=.06 if args.profile == "varied" else None)
      if args.profile == "varied":
        # Same open-gripper preparation in both placements; retain only physically matched pairs.
        for i in range(6):
          action = [0, 0, parameters["height_action"] if i < 2 else 0,
                    parameters["yaw_action"] if i < 3 else 0, 1]
          _, _, terminated, truncated, _ = env.step(action)
          if terminated or truncated:
            raise SceneRejected("source preparation ended the episode")
          history.append(frame(f"prepare_{i:04d}"))
      mujoco.mj_forward(env.model, env.data)
      saved = snapshot(env)
      source_p = env._observation()["proprio"].copy()
      if first_p is None:
        first_p = source_p
      else:
        error = float(np.max(np.abs(source_p - first_p)))
        if error > args.pair_tolerance:
          raise SceneRejected(f"{scene}: placements changed starting robot p by {error:.6g}; "
                             "this scene is not a controlled position pair (try another seed/offset)")
      source = len(manifest["states"])
      manifest["states"].append(record(env, f"{scene}/{placement}/source", scene, split, history))
      np.savez_compressed(folder / "source.npz", integration_state=saved["state"],
                          **{k: np.asarray(v) for k, v in saved["controller"].items()})
      write_json(folder / "replay.json", {"seed": seed, "layout": layout_dict,
                 "approach_actions": common_actions, "source_step": saved["step_count"],
                 "source_p": source_p.tolist(), "config": cfg, "parameters": parameters})
      for branch, actions in branches(args, parameters).items():
        restore(env, saved)
        p_error = float(np.max(np.abs(env._observation()["proprio"] - source_p)))
        actual = snapshot(env)["state"]
        state_error = float(np.max(np.abs(actual - saved["state"])))
        if p_error > 1e-5 or state_error > 1e-5:
          raise RuntimeError(f"source restore mismatch: p={p_error}, integration={state_error}")
        ids, branch_history = [source], history.copy()
        for step, action in enumerate(actions, 1):
          _, reward, terminated, truncated, info = env.step(action)
          mujoco.mj_forward(env.model, env.data)
          branch_history.append(frame(f"{branch}_{step:04d}"))
          ids.append(len(manifest["states"]))
          manifest["states"].append(record(env, f"{scene}/{placement}/{branch}/{step}",
                    scene, split, branch_history, action, info, reward))
          if terminated or truncated:
            raise SceneRejected(f"{scene}/{placement}/{branch} ended at {step}; no complete rollout")
        r = {"id": f"{scene}/{placement}/{branch}", "scene": scene, "split": split,
             "placement": placement, "branch": branch, "states": ids, "actions": actions,
             "restore_p_error": p_error, "restore_integration_error": state_error,
             "parameters": parameters, "anchor_xy": anchor.tolist(), "simulation_seed": seed}
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
  p.add_argument("--profile", choices=("controlled", "varied"), default="controlled")
  p.add_argument("--resume", action="store_true", help="continue an interrupted collection after its last complete scene")
  p.add_argument("--decision-branches", action="store_true", help="also record close/open without lifting")
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
  p.add_argument("--max-scene-attempts", type=int, default=8)
  args = p.parse_args()
  if min(args.train_scenes, args.val_scenes, args.test_scenes, args.hold_steps, args.lift_steps, args.max_scene_attempts) < 1:
    p.error("scene counts and sequence lengths must be positive")
  if args.run.exists() and not args.resume and any(
      f.name not in ("pipeline.json", "logs") for f in args.run.iterdir()):
    p.error("run directory is not empty; use --resume for this exact interrupted collection or a new name")
  if not 0 < args.lift_action <= 1 or args.offset < 0.08 or args.pair_tolerance <= 0:
    p.error("lift-action must be in (0,1], offset >= .08 m, pair-tolerance > 0")
  cfg = load_config(args.config, overrides={"robot": {"posture_noise": 0.0}})
  if not cfg.control.yaw.enabled:
    p.error("this experiment uses five-dimensional actions; enable control.yaw")
  args.run.mkdir(parents=True, exist_ok=True)
  settings = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()
              if k not in ("resume", "decision_branches")}
  if args.decision_branches:
    settings["decision_branches"] = True
  manifest = {"schema": "vision_consequences_v1", "camera": args.camera, "clip_frames": 64,
              "clip_padding": "repeat_first_observed_frame", "control_hz": cfg.control.hz,
              "proprio_columns": P_COLUMNS, "action_columns": A_COLUMNS,
              "config": cfg, "settings": settings,
              "provenance": {**provenance(), "mujoco": mujoco.__version__},
              "states": [], "rollouts": [], "rejected_attempts": [], "complete": False}
  rng = np.random.default_rng(args.seed)
  # Entire scene groups are split before any branches or images are generated.
  regions = {"train": (0.14, 0.22), "val": (0.225, 0.245), "test": (0.105, 0.125)}
  n = 0
  if args.resume:
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["settings"] != settings or manifest["config"] != cfg:
      p.error("collection resume settings/configuration differ; use the original command")
    if manifest.get("complete"):
      p.error("collection is already complete")
    n = manifest.get("completed_scene_groups", 0)
    del manifest["states"][manifest.get("committed_states", 0):]
    del manifest["rollouts"][manifest.get("committed_rollouts", 0):]
    shutil.rmtree(args.run / "episodes" / f"scene_{n:04d}", ignore_errors=True)
    if "collection_rng_state" in manifest:
      rng.bit_generator.state = manifest["collection_rng_state"]
    manifest.pop("collection_error", None)
    print(f"resuming collection after {n} complete scene groups", flush=True)
  write_json(args.run / "manifest.json", manifest)
  scene_index = 0
  try:
    for split, count in (("train", args.train_scenes), ("val", args.val_scenes), ("test", args.test_scenes)):
      for index in range(count):
        scene_index += 1
        if scene_index <= n:
          continue
        scene = f"scene_{n:04d}"
        for attempt in range(args.max_scene_attempts):
          anchor = np.array([rng.uniform(*(regions[split] if args.profile == "controlled" else (.12, .24))),
                             rng.uniform(*((-.26, -.21) if args.profile == "controlled" else (-.28, -.16)))])
          parameters = scene_parameters(rng, index, args)
          state_start, rollout_start = len(manifest["states"]), len(manifest["rollouts"])
          scene_seed = args.seed + n + attempt*100000
          try:
            collect_scene(args.run, manifest, scene, split, scene_seed, anchor, args, cfg, parameters)
            break
          except SceneRejected as error:
            if args.profile != "varied":
              raise
            manifest["rejected_attempts"].append({"scene": scene, "split": split, "attempt": attempt,
                 "seed": scene_seed, "anchor": anchor.tolist(), "parameters": parameters, "reason": str(error)})
            del manifest["states"][state_start:]; del manifest["rollouts"][rollout_start:]
            shutil.rmtree(args.run / "episodes" / scene, ignore_errors=True)
            write_json(args.run / "manifest.json", manifest)
            print(f"REJECT {scene} attempt {attempt+1}: {error}; resampling", flush=True)
        else:
          raise RuntimeError(f"{scene}: exhausted {args.max_scene_attempts} attempts; inspect rejected_attempts")
        n += 1
        manifest.update(completed_scene_groups=n, committed_states=len(manifest["states"]),
                        committed_rollouts=len(manifest["rollouts"]), collection_rng_state=rng.bit_generator.state)
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
