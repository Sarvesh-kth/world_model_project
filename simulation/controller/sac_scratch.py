# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

from world_model.common import SIMULATION, save_csv, write_json
from . import control_pipeline as cp

# Two SACs trained in imagination with the same recipe (imagined_sac.py): one starts its imagined episodes
# from empty-table states only, the other from empty and obstacle states. Both are then run on the cameras
# (method sac_imagined) on the same fresh scenes, 30 empty tables and 30 with 1-2 obstacles, the scenes the
# mpc rows of results/README.md were run on. Training and evaluation both resume: a finished policy or
# episode is kept, so the script can be restarted after an interruption (an interrupted training folder
# has to be deleted by hand, imagined_sac refuses to overwrite).
#   python -u -m controller.sac_scratch --steps 40000 --episodes 30
# Output: data/<out>/training/<policy>/ (the two runs, models/sac_best.zip), data/<out>/evaluation/<policy>/
# <condition>/episodes/sac_imagined/seed_*/ (gif, steps.csv, result.json) and results/<out>/summary.{csv,txt}.

POLICIES = (("train_empty", False), ("train_mixed", True))
CONDITIONS = (("empty", 20495010, None), ("obstacles", 20496010, [1, 2]))


# Placements, mean final distance and contact steps per policy and condition, written next to the episodes
def summarize(out, rows):
  target = SIMULATION.parent / "results" / out.name
  target.mkdir(parents=True, exist_ok=True)
  table, lines = [], []
  for policy, _ in POLICIES:
    for condition, _, _ in CONDITIONS:
      group = [r for r in rows if r["policy"] == policy and r["condition"] == condition]
      if not group:
        continue
      table.append({"policy": policy, "condition": condition, "episodes": len(group),
                    "placed": sum(r["placed"] for r in group),
                    "mean_final_goal_distance_cm": float(np.mean([r["final_goal_distance_cm"] for r in group])),
                    "obstacle_contact_steps": sum(r["obstacle_contact_steps"] for r in group)})
      lines.append(f"{policy} on {condition}: placed {table[-1]['placed']}/{len(group)}; final distance "
                   f"{table[-1]['mean_final_goal_distance_cm']:.1f} cm; contact steps {table[-1]['obstacle_contact_steps']}")
  save_csv(target / "episodes.csv", rows)
  save_csv(target / "summary.csv", table)
  (target / "summary.txt").write_text("\n".join(lines) + "\n")
  write_json(target / "summary.json", table)
  print("\n".join(lines), flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", type=pathlib.Path, default=pathlib.Path("data/episodes_v3"), help="the run the imagined episodes start from")
  p.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("models/v3c"))
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/sac_scratch_v1"))
  p.add_argument("--steps", type=int, default=40000, help="imagined transitions per policy")
  p.add_argument("--episodes", type=int, default=30, help="scenes per condition")
  p.add_argument("--seed", type=int, default=20474010, help="training seed of both policies")
  p.add_argument("--position-jitter", type=float, default=.01)
  p.add_argument("--keep-frames", action="store_true", help="keep the jpeg frames of every episode (the gif is always kept)")
  cp.run_arguments(p)
  args = p.parse_args()
  args.headless = True
  args.out = args.out.resolve()
  if SIMULATION / "data" not in args.out.parents:
    p.error("keep --out under simulation/data/")

  # 1. the two policies, trained one after the other
  for policy, obstacles in POLICIES:
    run = args.out / "training" / policy
    if (run / "imagined_training.json").exists():
      continue
    command = [sys.executable, "-u", "-m", "controller.imagined_sac", "--run", str(args.run), "--models-run", str(args.models_run),
               "--out", str(run), "--steps", str(args.steps), "--seed", str(args.seed), "--threads", str(args.threads)]
    if obstacles:
      command.append("--obstacles")
    print(f"TRAIN {policy}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=SIMULATION, check=True)

  # 2. both on the cameras, same scenes
  import torch
  torch.set_num_threads(args.threads)
  cfg, rest_z = cp.run_settings(args.models_run, "")
  models = cp.WorldModels(args.models_run, "", args)
  rows = []
  for policy, _ in POLICIES:
    actor = cp.Policy(args.out / "training" / policy)
    for condition, base, obstacles in CONDITIONS:
      root = args.out / "evaluation" / policy / condition
      for i in range(args.episodes):
        seed = base + i
        folder = root / "episodes" / "sac_imagined" / f"seed_{seed}"
        if (folder / "result.json").exists():
          result = json.loads((folder / "result.json").read_text())
        else:
          print(f"EVALUATE {policy} on {condition}, seed {seed}", flush=True)
          layout = cp.scene_layout(None, cfg, seed, obstacles, random_ends=True)
          result = cp.episode(root, "sac_imagined", seed, cfg, layout, rest_z, args, actor, models)
          if not args.keep_frames:
            shutil.rmtree(folder / "frames")
        rows.append({"policy": policy, "condition": condition, "seed": seed, "placed": bool(result["task_success"]),
                     "final_goal_distance_cm": result["final_goal_distance_cm"],
                     "obstacle_contact_steps": result["obstacle_contact_steps"]})
        summarize(args.out, rows)
  print(f"DONE: table in {SIMULATION.parent / 'results' / args.out.name}, gifs in {args.out / 'evaluation'}", flush=True)


if __name__ == "__main__":
  main()
