"""Test 1: paired full-task scenes with obstacles that actually block the carry corridor.

Per scene group (same cube start, ±3 cm jitter):
  normal_empty        frozen SAC drives the task on the empty table (as in the team's campaign)
  replay_obstacles    the SAME executed actions replayed with 1-3 corridor obstacles from the project's own
                      sampler -> the robot/cube hit or graze them, so collision / proximity / table penalties
                      differ from the empty twin only because of the obstacles the camera can see
  scripted_obstacles  the scripted route planner in the obstacle scene (goes around / over) -> low penalties
Same manifest format as full_task.collect, campaign full_task_obstacles_test1. Simulator labels (contacts,
reward components) are evaluation / training targets only; nothing the controllers see.
"""
import os

# offscreen rendering through EGL; set before mujoco is imported
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import copy
import json
import pathlib
import time

import numpy as np
from PIL import Image

from environment.config import Config
from environment.obstacles import sample_obstacles
from data_collection.scripted_policy import ScriptedPickPlace
from world_model.prepare import P_COLUMNS, A_COLUMNS
from .collect import record
from .control_pipeline import frozen_baseline
from .data import digest, provenance, write_json
from .full_task import RecoverySession
from .pipeline import SIMULATION

CAMPAIGN = "full_task_obstacles_test1"


def layouts(base, cfg, seed, jitter, count):
  rng = np.random.default_rng(seed)
  empty = copy.deepcopy(base)
  empty["pick"] = (np.asarray(base["pick"]) + rng.uniform(-jitter, jitter, 2)).tolist()
  empty["obstacles"] = []
  cfg = copy.deepcopy(cfg)
  cfg.task.obstacles.count = list(count)
  obstacles = sample_obstacles(cfg, rng, np.asarray(empty["pick"]), np.asarray(empty["place"]))
  if obstacles is None:
    raise RuntimeError(f"no obstacle layout for seed {seed}")
  blocked = copy.deepcopy(empty)
  blocked["obstacles"] = [o.to_dict() for o in obstacles]
  return {"empty": empty, "obstacles": blocked}


def run(root, manifest, cfg, name, scene, split, layout, seed, controller, replay=None):
  folder = root / "observations" / name
  session = RecoverySession(cfg, layout, case="normal")
  states, actions, history = [], [], []
  try:
    session.reset(seed)
    policy = None
    if controller == "sac":
      from stable_baselines3 import SAC
      policy = SAC.load(manifest["settings"]["sac_checkpoint"], device="cpu")
    scripted = ScriptedPickPlace(session.sim, np.random.default_rng(seed)) if controller == "scripted" else None

    def capture(action=None, info=None, reward=0):
      image = session.sim.render("static")
      filename = folder / f"static_{len(states):04d}.jpg"
      filename.parent.mkdir(parents=True, exist_ok=True)
      Image.fromarray(image).save(filename, quality=cfg.data.jpeg_quality)
      history.append(str(filename.relative_to(root)))
      state = record(session.sim, f"{name}:{len(states):04d}", scene, split, history, action, info, reward)
      state.update(view="obstacles" if layout["obstacles"] else "empty", case=controller, phase=session.phase(),
                   source_controller=controller, forced_release=False, task_success=session.success,
                   obstacle_distance=float(session.sim.obstacle_distance()))
      states.append(state)

    capture()
    for step in range(cfg.episode.max_steps):
      if replay is not None:
        if step >= len(replay):
          break
        action = np.asarray(replay[step], np.float32)
      elif policy is not None:
        action = policy.predict(session.observation(), deterministic=True)[0]
      else:
        action = np.asarray(scripted.act(), np.float32)
      _, reward, done, timeout, info = session.step(action)
      actions.append([float(v) for v in info["executed_action"]])
      capture(info["executed_action"], info, reward)
      if replay is None and (done or timeout or (scripted is not None and scripted.done)):
        break
    result = session.result()
  finally:
    session.close()
  start = len(manifest["states"])
  manifest["states"].extend(states)
  manifest["rollouts"].append({"id": name, "scene": scene, "split": split,
      "placement": "obstacles" if layout["obstacles"] else "empty", "branch": controller,
      "view": "obstacles" if layout["obstacles"] else "empty", "case": controller,
      "states": list(range(start, start + len(states))), "actions": actions,
      "restore_p_error": 0.0, "restore_integration_error": 0.0, "result": result})
  manifest["scene_settings"].append({"rollout": name, "seed": seed, "layout": layout})
  penalties = sum(1 for s in states if any(s["reward_components"].get(k, 0) for k in ("collision", "proximity", "table_hit")))
  print(f"COLLECT {name} split={split} steps={len(actions)} placed={result['task_success']} "
        f"contacts={result['obstacle_contact_steps']} penalised_states={penalties}", flush=True)
  return actions


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/obstacles_test1"))
  p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
  p.add_argument("--train-scenes", type=int, default=14)
  p.add_argument("--val-scenes", type=int, default=4)
  p.add_argument("--test-scenes", type=int, default=6)
  p.add_argument("--seed", type=int, default=40464005)
  p.add_argument("--jitter", type=float, default=.03)
  p.add_argument("--obstacles", type=int, nargs=2, default=(1, 3))
  args = p.parse_args()
  baseline = json.loads((args.baseline_run / "pipeline.json").read_text())
  cfg_dict, base_layout = baseline["inputs"]["config"], baseline["inputs"]["layout"]
  evidence = frozen_baseline(args.baseline_run.resolve(), cfg_dict, base_layout, baseline["inputs"]["options"]["position_jitter"])
  cfg = Config.nested(cfg_dict)
  groups = []
  for split, n in (("train", args.train_scenes), ("val", args.val_scenes), ("test", args.test_scenes)):
    for _ in range(n):
      i = len(groups)
      groups.append({"scene": f"scene_{i:04d}", "seed": args.seed + i, "split": split})
  settings = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}
  settings["sac_checkpoint"] = str((args.baseline_run / "models/sac_best.zip").resolve())
  settings["sac_checkpoint_sha256"] = evidence["checkpoint_sha256"]
  args.out.mkdir(parents=True, exist_ok=True)
  path = args.out / "manifest.json"
  manifest = json.loads(path.read_text()) if path.exists() else {
      "schema": "vision_consequences_v1", "campaign": CAMPAIGN, "complete": False,
      "settings": {"pair_tolerance": 1e-5, "collection": settings, "sac_checkpoint": settings["sac_checkpoint"]},
      "config": cfg_dict, "camera": "static", "control_hz": cfg.control.hz,
      "p_columns": P_COLUMNS, "action_columns": A_COLUMNS,
      "states": [], "rollouts": [], "scene_settings": [], "groups": groups, "provenance": provenance()}
  done = {r["id"] for r in manifest["rollouts"]}
  for group in groups:
    pair = layouts(base_layout, cfg, group["seed"], args.jitter, args.obstacles)
    names = [f"{group['scene']}_normal_empty", f"{group['scene']}_replay_obstacles", f"{group['scene']}_scripted_obstacles"]
    if all(n in done for n in names):
      continue
    start = time.monotonic()
    actions = run(args.out, manifest, cfg, names[0], group["scene"], group["split"], pair["empty"], group["seed"], "sac")
    run(args.out, manifest, cfg, names[1], group["scene"], group["split"], pair["obstacles"], group["seed"], "replay", replay=actions)
    run(args.out, manifest, cfg, names[2], group["scene"], group["split"], pair["obstacles"], group["seed"], "scripted")
    write_json(path, manifest)
    print(f"{group['scene']} done in {time.monotonic()-start:.0f}s", flush=True)
  manifest["complete"] = True
  write_json(path, manifest)
  write_json(args.out / "scene_settings.json", manifest["scene_settings"])
  write_json(args.out / "collection.json", {"campaign": CAMPAIGN, "settings": settings, "config": cfg_dict,
             "layout": base_layout, "groups": groups, "frozen_baseline": evidence,
             "code_sha256": {str(pathlib.Path(__file__).relative_to(SIMULATION)): digest(__file__)}})
  print(f"COLLECTION COMPLETE: {len(manifest['states'])} states, {len(manifest['rollouts'])} rollouts", flush=True)


if __name__ == "__main__":
  main()
