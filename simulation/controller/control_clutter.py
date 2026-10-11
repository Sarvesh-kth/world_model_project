# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import copy
import json
import pathlib
import shutil

import numpy as np

from environment.config import Config
from environment.rewards import fill_weights
from world_model.common import SIMULATION, write_json
from . import control_pipeline as cp

# Run the controllers on the held-out obstacle scenes (data/obstacles_v2 layouts): the same episode() as
# control_pipeline, but the layout (with its obstacles) comes from the run's scene_settings instead of the
# empty layout file. The planner sees obstacles through the latent (penalty head and height map); the SAC
# only if it was trained with --obstacles (then it reads the height map too).
#   python -m controller.control_clutter --methods jepa_mpc --only scene_0019 scene_0021
#   python -m controller.control_clutter --methods rl_q jepa_mpc --no-penalties


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("data/combined_v2"))
  p.add_argument("--tag", default="combined")
  p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
  p.add_argument("--scenes-run", type=pathlib.Path, default=pathlib.Path("data/obstacles_v2"),
                 help="run whose recorded test scene layouts are replayed")
  p.add_argument("--layout-suffix", default="_replay_obstacles", help="which rollout's layout to take per scene")
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/control_clutter"))
  p.add_argument("--methods", nargs="+", default=["rl_true", "rl_q", "jepa_mpc"], choices=cp.METHODS)
  p.add_argument("--scenes", type=int, default=6, help="how many test scenes")
  p.add_argument("--only", nargs="*", default=[], help="run just these scenes, e.g. scene_0019 scene_0021")
  p.add_argument("--position-jitter", type=float, default=0.0)
  cp.run_arguments(p)
  args = p.parse_args()
  args.out = args.out.resolve()
  if SIMULATION / "data" not in args.out.parents:
    p.error("keep --out under simulation/data/")
  export_dir = SIMULATION.parent / "results" / args.out.name
  for folder in (args.out, export_dir):
    if folder.exists():
      shutil.rmtree(folder)

  import torch
  torch.set_num_threads(args.threads)
  collection = json.loads((args.scenes_run / "collection.json").read_text())
  settings = {s["rollout"]: s for s in json.loads((args.scenes_run / "scene_settings.json").read_text())}
  groups = [g for g in collection["groups"] if g["split"] == "test"][:args.scenes]
  if args.only:
    groups = [g for g in collection["groups"] if g["scene"] in args.only]
  cfg = Config.nested(copy.deepcopy(collection["config"]))
  cfg.episode.terminate_on_success = False
  fill_weights(cfg.rewards.weights)
  _, rest_z = cp.run_settings(args.models_run, args.tag)
  policy = cp.Policy(args.baseline_run) if any(m in ("rl_true", "rl_q") for m in args.methods) else None
  models = cp.WorldModels(args.models_run, args.tag, args) if any(m in ("rl_q", "jepa_mpc") for m in args.methods) else None
  if models is not None:
    print(f"penalty head: {'ON' if models.r is not None else 'off'}", flush=True)

  rows = []
  for group in groups:
    layout = settings[f"{group['scene']}{args.layout_suffix}"]["layout"]
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
