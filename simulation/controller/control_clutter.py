# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import copy
import json
import pathlib
import shutil

import numpy as np

from environment.config import Config
from world_model.common import SIMULATION, write_json
from . import control_pipeline as cp

# Run the controllers on the held-out obstacle scenes of a collect_obstacles run: the same episode() as
# control_pipeline, but the layout (with its obstacles) comes from the run's scene_settings instead of the
# empty layout file. rl_true and rl_q cannot see obstacles, only the planner can, through the penalty head.
#   python -m controller.control_clutter --methods jepa_mpc --only scene_0019 scene_0021
#   python -m controller.control_clutter --methods rl_q jepa_mpc --no-penalties
#   python -m controller.control_clutter --methods mpc --new-scenes 20      (fresh layouts, never in the training data)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("data/combined_test1"))
  p.add_argument("--tag", default="combined")
  p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
  p.add_argument("--scenes-run", type=pathlib.Path, default=pathlib.Path("data/obstacles_test1"),
                 help="collect_obstacles run whose test scenes are replayed")
  p.add_argument("--layout-suffix", default="_replay_obstacles", help="which rollout's layout to take per scene")
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/control_clutter"))
  p.add_argument("--methods", nargs="+", default=["rl_true", "rl_q", "jepa_mpc"], choices=cp.METHODS)
  p.add_argument("--scenes", type=int, default=6, help="how many test scenes")
  p.add_argument("--only", nargs="*", default=[], help="run just these scenes, e.g. scene_0019 scene_0021")
  p.add_argument("--position-jitter", type=float, default=0.0)
  p.add_argument("--new-scenes", type=int, default=0,
                 help="instead of the run's test scenes, this many fresh layouts from the same obstacle sampler and settings")
  p.add_argument("--scene-seed", type=int, default=40466000,
                 help="first seed of --new-scenes (the obstacles_test1 collection used 40464005-40464028)")
  cp.planner_arguments(p)
  args = p.parse_args()
  args.out = args.out.resolve()
  if SIMULATION / "data" not in args.out.parents:
    p.error("keep --out under simulation/data/")
  export_dir = SIMULATION.parent / "results" / args.out.name
  for folder in (args.out, export_dir):
    if folder.exists():
      shutil.rmtree(folder)

  import torch
  from stable_baselines3 import SAC
  torch.set_num_threads(args.threads)
  collection = json.loads((args.scenes_run / "collection.json").read_text())
  settings = {s["rollout"]: s for s in json.loads((args.scenes_run / "scene_settings.json").read_text())}
  groups = [g for g in collection["groups"] if g["split"] == "test"][:args.scenes]
  if args.only:
    groups = [g for g in collection["groups"] if g["scene"] in args.only]
  if args.new_scenes:
    # fresh scenes: the collector's own layouts() with its settings, at seeds it never used; a seed whose
    # obstacles do not fit is skipped
    from world_model.collect_obstacles import layouts
    base = Config.nested(copy.deepcopy(collection["config"]))
    groups, seed = [], args.scene_seed
    while len(groups) < args.new_scenes:
      try:
        pair = layouts(collection["layout"], base, seed, collection["settings"]["jitter"], collection["settings"]["obstacles"])
        groups.append({"scene": f"new_{len(groups):04d}", "seed": seed, "layout": pair["obstacles"]})
      except RuntimeError:
        pass
      seed += 1
  cfg = Config.nested(copy.deepcopy(collection["config"]))
  cfg.episode.terminate_on_success = False
  _, rest_z = cp.run_settings(args.models_run, args.tag)
  policy = SAC.load(args.baseline_run / "models/sac_best.zip", device="cpu")
  models = cp.WorldModels(args.models_run, args.tag, args) if any(m in cp.CAMERA_METHODS for m in args.methods) else None
  if models is not None:
    print(f"penalty head: {'ON' if models.r is not None else 'off'}", flush=True)

  rows = []
  for group in groups:
    layout = group["layout"] if "layout" in group else settings[f"{group['scene']}{args.layout_suffix}"]["layout"]
    for method in args.methods:
      print(f"SCENE {group['scene']} {method} obstacles={len(layout['obstacles'])}", flush=True)
      result = cp.episode(args.out, method, group["seed"], cfg, layout, rest_z, args, policy, models)
      rows.append({"scene": group["scene"], "method": method, "success": result["task_success"],
                   "goal_cm": result["final_goal_distance_cm"], "ever_held": result["ever_held"],
                   "obstacle_contacts": result["obstacle_contact_steps"], "fallbacks": result["planner_fallback_steps"]})
      write_json(args.out / "scenes.json", rows)
  cp.summarize(args.out, export_dir, args.methods)
  write_json(export_dir / "scenes.json", rows)


if __name__ == "__main__":
  main()
