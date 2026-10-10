# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import copy
import json
import pathlib

import numpy as np
from PIL import Image

from environment.config import Config
from data_collection.scripted_policy import ScriptedPickPlace
from rl.task_control import TaskSession
from .common import new_manifest, record, write_json

# Collect full pick-and-place trajectories on the SAC's layout for training Q and D. Per scene (a cube start
# jittered by --jitter, 3 cm so Q cannot guess the cube from the arm) four cases, each in two views:
#   cases  normal            the frozen SAC drives the task
#          drop_early/middle/late  the scripted policy drives; at 15 / 50 / 75 % of the carry the gripper is
#                            forced open for 7 steps, the cube drops, the scripted policy restarts and recovers
#   views  empty             the plain table
#          clutter           the SAME executed actions replayed with 1-3 boxes at the far table edge, so the
#                            latent sees clutter that changes nothing physically
# Every step is one state in the manifest (robot state, cube pose, held, reward, the camera frame).
#   python -m world_model.collect_full --out data/full_test1

CASES = ("normal", "drop_early", "drop_middle", "drop_late")
VIEWS = ("empty", "clutter")


# The empty and the cluttered layout of one scene, same cube start in both
def layouts(base, seed, jitter):
  rng = np.random.default_rng(seed)
  empty = copy.deepcopy(base)
  empty["pick"] = (np.asarray(base["pick"]) + rng.uniform(-jitter, jitter, 2)).tolist()
  empty["obstacles"] = []
  clutter = copy.deepcopy(empty)
  # 1 to 3 boxes along the far edge, never on the route
  ys = rng.choice(np.linspace(-.34, .34, 9), int(rng.integers(1, 4)), replace=False)
  for y in sorted(ys):
    half = rng.uniform([.018, .018, .025], [.03, .028, .1])
    clutter["obstacles"].append({"kind": "box", "pos": [float(rng.uniform(.40, .425)), float(y)],
                                 "yaw": float(rng.uniform(-.3, .3)), "size": half.tolist()})
  return {"empty": empty, "clutter": clutter}


# TaskSession that can force the gripper open part way through the carry, so the data holds drops and
# recoveries. Also reports the task phase and the action that was really executed.
class RecoverySession(TaskSession):

  def __init__(self, cfg, layout, jitter=0, case="normal"):
    super().__init__(cfg, layout, jitter)
    if case not in CASES:
      raise ValueError(f"unknown case {case}")
    self.case = case

  def reset(self, seed):
    obs = super().reset(seed)
    self.initial_xyz = self.obs["state"][:3].copy()
    self.release_remaining = 0
    self.intervention_step = None
    self.release_end_step = None
    self.regrasp_step = None
    return obs

  # Where the task is right now, stored with every state
  def phase(self):
    xyz = self.obs["state"][:3]
    held = self.sim._check_contacts()[0]
    if self.release_remaining:
      return "forced_release"
    if self.intervention_step is not None and self.regrasp_step is None:
      return "recovery"
    if self.success or (self.lifted and not held and np.linalg.norm(xyz[:2] - self.goal[:2]) < .07):
      return "release_settle"
    if held:
      if np.linalg.norm(xyz[:2] - self.goal[:2]) < .07:
        return "lower"
      return "carry" if xyz[2] - self.rest_z >= .04 else "grasp_lift"
    return "approach"

  def step(self, action):
    action = np.asarray(action, np.float32).copy()
    xyz = self.obs["state"][:3]
    held = self.sim._check_contacts()[0]

    # trigger the forced release once the carry has progressed far enough
    original_distance = np.linalg.norm(self.initial_xyz[:2] - self.goal[:2])
    progress = 1 - np.linalg.norm(xyz[:2] - self.goal[:2]) / max(original_distance, 1e-6)
    threshold = {"drop_early": .15, "drop_middle": .50, "drop_late": .75}.get(self.case)
    if (threshold is not None and self.intervention_step is None and held
        and xyz[2] - self.rest_z >= .04 and progress >= threshold):
      self.intervention_step = self.sim.step_count + 1
      self.release_remaining = 7
    intervened = self.release_remaining > 0
    if intervened:
      # hold the hand still and open, gravity does the rest
      action = np.array([0, 0, 0, 0, 1], np.float32)
      self.release_remaining -= 1

    obs, reward, done, timeout, info = super().step(action)
    restart = intervened and self.release_remaining == 0
    if restart:
      self.release_end_step = self.sim.step_count
    if (self.release_end_step is not None and self.sim.step_count > self.release_end_step
        and info["held_endpoint"] and self.regrasp_step is None):
      self.regrasp_step = self.sim.step_count
    info.update(executed_action=action.tolist(), forced_release=intervened, restart_scripted=restart, phase=self.phase())
    return obs, reward, done, timeout, info

  def result(self):
    return {**super().result(), "case": self.case,
            "intervention_triggered": self.intervention_step is not None,
            "regrasped_after_intervention": self.regrasp_step is not None}


# Run one rollout, save its frames under observations/<name>/ and append its states to the manifest.
# replay = a list of actions to execute instead of asking the controller (the clutter twin).
def run_rollout(root, manifest, cfg, name, scene, split, layout, seed, case, view, sac, replay=None):
  folder = root / "observations" / name
  session = RecoverySession(cfg, layout, case=case)
  states, actions, history = [], [], []
  try:
    session.reset(seed)
    scripted = ScriptedPickPlace(session.sim, np.random.default_rng(seed))

    def capture(action=None, info=None, reward=0):
      filename = folder / f"static_{len(states):04d}.jpg"
      filename.parent.mkdir(parents=True, exist_ok=True)
      Image.fromarray(session.sim.render("static")).save(filename, quality=cfg.data.jpeg_quality)
      history.append(str(filename.relative_to(root)))
      state = record(session.sim, f"{name}:{len(states):04d}", scene, split, history, action, info, reward)
      state.update(view=view, case=case, phase=session.phase(), task_success=session.success,
                   forced_release=bool((info or {}).get("forced_release")))
      states.append(state)

    capture()
    for step in range(cfg.episode.max_steps):
      if replay is not None:
        if step >= len(replay):
          break
        action = replay[step]
      elif case == "normal":
        action = sac.predict(session.observation(), deterministic=True)[0]
      else:
        action = scripted.act()
      _, reward, done, timeout, info = session.step(action)
      executed = info["executed_action"]
      if replay is not None and not np.allclose(executed, action, atol=1e-7):
        raise RuntimeError(f"the forced release fired at another step in {name}; the twins no longer match")
      actions.append(executed)
      capture(executed, info, reward)
      if info["restart_scripted"]:
        scripted = ScriptedPickPlace(session.sim, np.random.default_rng(seed + step + 1))
      if replay is None and (done or timeout):
        break
    result = session.result()
  finally:
    session.close()

  start = len(manifest["states"])
  manifest["states"].extend(states)
  manifest["rollouts"].append({"id": name, "scene": scene, "split": split, "view": view, "case": case,
                               "states": list(range(start, start + len(states))), "actions": actions, "result": result})
  manifest["scene_settings"].append({"rollout": name, "seed": seed, "layout": layout})
  print(f"COLLECT {name} split={split} steps={len(actions)} placed={result['task_success']} "
        f"intervention={result['intervention_triggered']} regrasp={result['regrasped_after_intervention']}", flush=True)
  return actions


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/full_test1"))
  p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
  p.add_argument("--train-scenes", type=int, default=16)
  p.add_argument("--val-scenes", type=int, default=4)
  p.add_argument("--test-scenes", type=int, default=6)
  p.add_argument("--seed", type=int, default=30464005)
  p.add_argument("--jitter", type=float, default=.03, help="cube start jitter per xy axis in metres")
  args = p.parse_args()

  # the layout and config the SAC was trained with, and the SAC itself
  baseline = json.loads((args.baseline_run / "pipeline.json").read_text())["inputs"]
  cfg = Config.nested(baseline["config"])
  from stable_baselines3 import SAC
  sac = SAC.load(args.baseline_run / "models/sac_best.zip", device="cpu")

  groups = []
  for split, n in (("train", args.train_scenes), ("val", args.val_scenes), ("test", args.test_scenes)):
    for _ in range(n):
      i = len(groups)
      groups.append({"scene": f"scene_{i:04d}", "seed": args.seed + i, "split": split})
  settings = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}

  # an existing manifest is continued, finished rollouts are skipped
  args.out.mkdir(parents=True, exist_ok=True)
  path = args.out / "manifest.json"
  manifest = json.loads(path.read_text()) if path.exists() else new_manifest("full_task", baseline["config"], settings)
  done = {r["id"] for r in manifest["rollouts"]}
  for group in groups:
    pair = layouts(baseline["layout"], group["seed"], args.jitter)
    for case in CASES:
      empty_actions = None
      for view in VIEWS:
        name = f"{group['scene']}_{case}_{view}"
        if name in done:
          if view == "empty":
            empty_actions = next(r for r in manifest["rollouts"] if r["id"] == name)["actions"]
          continue
        actions = run_rollout(args.out, manifest, cfg, name, group["scene"], group["split"], pair[view], group["seed"],
                              case, view, sac, replay=empty_actions if view == "clutter" else None)
        if view == "empty":
          empty_actions = actions
        write_json(path, manifest)

  manifest["complete"] = True
  write_json(path, manifest)
  write_json(args.out / "scene_settings.json", manifest["scene_settings"])
  write_json(args.out / "collection.json", {"campaign": "full_task", "settings": settings, "config": baseline["config"],
                                            "layout": baseline["layout"], "groups": groups})
  print(f"COLLECTION COMPLETE: {len(manifest['states'])} states, {len(manifest['rollouts'])} rollouts", flush=True)


if __name__ == "__main__":
  main()
