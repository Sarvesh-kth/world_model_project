# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

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
from .collect_full import RecoverySession
from .common import new_manifest, record, write_json

# Paired scenes with obstacles that really block the carry corridor, the data the penalty head R learns from.
# Per scene (same cube start, --jitter):
#   normal_empty        the frozen SAC drives the task on the empty table
#   replay_obstacles    the SAME actions replayed with 1-3 corridor obstacles -> the arm or cube hits them, so
#                       the collision / proximity / table penalties differ from the empty twin only because of
#                       what the camera can see
#   scripted_obstacles  the scripted route planner in the obstacle scene, goes around -> few penalties
#   python -m world_model.collect_obstacles --out data/obstacles_test1


# The empty layout and the same layout with corridor obstacles from the project's own sampler
def layouts(base, cfg, seed, jitter, count):
  rng = np.random.default_rng(seed)
  empty = copy.deepcopy(base)
  empty["pick"] = (np.asarray(base["pick"]) + rng.uniform(-jitter, jitter, 2)).tolist()
  empty["obstacles"] = []
  cfg = copy.deepcopy(cfg)
  cfg.task.obstacles.count = list(count)
  obstacles = sample_obstacles(cfg, rng, np.asarray(empty["pick"]), np.asarray(empty["place"]))
  if obstacles is None:
    raise RuntimeError(f"no obstacle layout fits for seed {seed}")
  blocked = copy.deepcopy(empty)
  blocked["obstacles"] = [o.to_dict() for o in obstacles]
  return {"empty": empty, "obstacles": blocked}


# One rollout driven by the SAC, the scripted policy or a replayed action list
def run_rollout(root, manifest, cfg, name, scene, split, layout, seed, controller, sac, replay=None):
  folder = root / "observations" / name
  view = "obstacles" if layout["obstacles"] else "empty"
  session = RecoverySession(cfg, layout, case="normal")
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
      state.update(view=view, case=controller, phase=session.phase(), task_success=session.success,
                   obstacle_distance=float(session.sim.obstacle_distance()))
      states.append(state)

    capture()
    for step in range(cfg.episode.max_steps):
      if replay is not None:
        if step >= len(replay):
          break
        action = np.asarray(replay[step], np.float32)
      elif controller == "sac":
        action = sac.predict(session.observation(), deterministic=True)[0]
      else:
        action = np.asarray(scripted.act(), np.float32)
      _, reward, done, timeout, info = session.step(action)
      actions.append([float(v) for v in info["executed_action"]])
      capture(info["executed_action"], info, reward)
      if replay is None and (done or timeout or (controller == "scripted" and scripted.done)):
        break
    result = session.result()
  finally:
    session.close()

  start = len(manifest["states"])
  manifest["states"].extend(states)
  manifest["rollouts"].append({"id": name, "scene": scene, "split": split, "view": view, "case": controller,
                               "states": list(range(start, start + len(states))), "actions": actions, "result": result})
  manifest["scene_settings"].append({"rollout": name, "seed": seed, "layout": layout})
  penalised = sum(1 for s in states if any(s["reward_components"].get(k, 0) for k in ("collision", "proximity", "table_hit")))
  print(f"COLLECT {name} split={split} steps={len(actions)} placed={result['task_success']} "
        f"contacts={result['obstacle_contact_steps']} penalised_states={penalised}", flush=True)
  return actions


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/obstacles_test1"))
  p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
  p.add_argument("--train-scenes", type=int, default=14)
  p.add_argument("--val-scenes", type=int, default=4)
  p.add_argument("--test-scenes", type=int, default=6)
  p.add_argument("--seed", type=int, default=40464005)
  p.add_argument("--jitter", type=float, default=.03)
  p.add_argument("--obstacles", type=int, nargs=2, default=(1, 3), help="min and max corridor obstacles per scene")
  args = p.parse_args()

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

  args.out.mkdir(parents=True, exist_ok=True)
  path = args.out / "manifest.json"
  manifest = json.loads(path.read_text()) if path.exists() else new_manifest("obstacles", baseline["config"], settings)
  done = {r["id"] for r in manifest["rollouts"]}
  for group in groups:
    pair = layouts(baseline["layout"], cfg, group["seed"], args.jitter, args.obstacles)
    names = [f"{group['scene']}_normal_empty", f"{group['scene']}_replay_obstacles", f"{group['scene']}_scripted_obstacles"]
    if all(n in done for n in names):
      continue
    start = time.monotonic()
    actions = run_rollout(args.out, manifest, cfg, names[0], group["scene"], group["split"], pair["empty"], group["seed"], "sac", sac)
    run_rollout(args.out, manifest, cfg, names[1], group["scene"], group["split"], pair["obstacles"], group["seed"], "replay", sac, replay=actions)
    run_rollout(args.out, manifest, cfg, names[2], group["scene"], group["split"], pair["obstacles"], group["seed"], "scripted", sac)
    write_json(path, manifest)
    print(f"{group['scene']} done in {time.monotonic() - start:.0f}s", flush=True)

  manifest["complete"] = True
  write_json(path, manifest)
  write_json(args.out / "scene_settings.json", manifest["scene_settings"])
  write_json(args.out / "collection.json", {"campaign": "obstacles", "settings": settings, "config": baseline["config"],
                                            "layout": baseline["layout"], "groups": groups})
  print(f"COLLECTION COMPLETE: {len(manifest['states'])} states, {len(manifest['rollouts'])} rollouts", flush=True)


if __name__ == "__main__":
  main()
