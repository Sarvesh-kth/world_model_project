"""Test 1: run the live controllers on the CLUTTERED versions of held-out full-task scenes.

control_pipeline refuses layouts with obstacles, so this calls its episode() directly, the way
clutter_pipeline.controls does. The rectangles are the collected route-clear clutter at the far table
edge: this measures whether perception survives clutter in view, not avoidance of blocking obstacles
(nothing in the current models sees obstacles).
   usage: python -m world_model.vision.control_clutter --models-run data/full_test1 --tag test1z --out data/control_clutter
"""
import argparse
import json
import pathlib
from types import SimpleNamespace

import numpy as np

from .data import digest, write_json
from . import control_pipeline as cp


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("data/combined_test1"))
  p.add_argument("--tag", default="combined")
  p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/control_clutter"))
  p.add_argument("--scenes", type=int, default=6)
  p.add_argument("--only", nargs="*", default=[], help="run just these scene names, e.g. scene_0019 scene_0021")
  p.add_argument("--methods", nargs="+", default=["rl_true", "rl_q", "jepa_mpc"])
  p.add_argument("--horizon", type=int, default=8)
  p.add_argument("--population", type=int, default=64)
  p.add_argument("--elites", type=int, default=8)
  p.add_argument("--iterations", type=int, default=3)
  p.add_argument("--terminal-weight", type=float, default=0)
  p.add_argument("--cem-std", type=float, default=0.3)
  p.add_argument("--no-penalties", action="store_true", help="ignore a trained penalty head")
  p.add_argument("--penalty-scale", type=float, default=2.0, help="multiply the imagined contact penalties; 2 leaves the guide earlier near obstacles")
  p.add_argument("--guide-margin", type=float, default=0.25, help="see control_pipeline")
  p.add_argument("--free-gripper", action="store_true", help="let CEM also sample the gripper")
  p.add_argument("--cem-smooth", type=float, default=0.5, help="see control_pipeline")
  p.add_argument("--no-width-gate", action="store_true", help="see control_pipeline")
  p.add_argument("--blind", action="store_true", help="control: proprio-only readout_p instead of Q(z, p)")
  p.add_argument("--scenes-run", type=pathlib.Path, default=pathlib.Path("data/obstacles_test1"), help="run whose collection/scene_settings give the test layouts")
  p.add_argument("--layout-suffix", default="_replay_obstacles", help="scene_settings rollout whose layout to use, e.g. _normal_clutter")
  p.add_argument("--headless", action="store_true", help="no MuJoCo window (the default when there is no display)")
  p.add_argument("--speed", type=float, default=1.0, help="playback speed of the window, times real time")
  p.add_argument("--threads", type=int, default=4)
  args = p.parse_args()
  import torch
  from stable_baselines3 import SAC
  torch.set_num_threads(args.threads)
  scenes_run = args.scenes_run or args.models_run
  collection = json.loads((scenes_run / "collection.json").read_text())
  settings = {s["rollout"]: s for s in json.loads((scenes_run / "scene_settings.json").read_text())}
  groups = [g for g in collection["groups"] if g["split"] == "test"][:args.scenes]
  if args.only:
    groups = [g for g in collection["groups"] if g["scene"] in args.only]
  manifest = json.loads((args.models_run / "manifest.json").read_text())
  cfg = collection["config"]
  rest = [s["object_xyz"][2] for s in manifest["states"] if s["split"] == "train" and not s["held"]
          and abs(s["object_xyz"][2]-cfg["table"]["height"]) < .06]
  encoder = json.loads((args.models_run / "features/meta.json").read_text())
  encoder = {k: encoder.get(k) for k in ("model", "model_revision", "camera", "clip_frames", "frame_stride",
                                          "pooling", "pca_sha256", "dtype", "spatial_pool")}
  checkpoint = args.baseline_run / "models/sac_best.zip"
  policy = SAC.load(checkpoint, device="cpu")
  models = cp.WorldModels(args.models_run, args.tag, encoder, blind=args.blind) if any(m in ("rl_q", "jepa_mpc") for m in args.methods) else None
  if models is not None and args.no_penalties:
    models.use_penalties = False
  if models is not None:
    models.width_gate = not args.no_width_gate
    models.penalty_scale = args.penalty_scale
  if models is not None:
    print(f"penalty head: {'ON' if models.use_penalties else 'off'}", flush=True)
  rows = []
  for group in groups:
    layout = settings[f"{group['scene']}{args.layout_suffix}"]["layout"]
    for method in args.methods:
      opts = SimpleNamespace(**vars(args), method=method, episode_seed=group["seed"], position_jitter=0,
                             policy_checkpoint=checkpoint)
      signature = {"config": cfg, "layout": layout, "known_rest_z": float(np.median(rest)), "encoder": encoder,
                   "frozen_baseline": {"checkpoint_sha256": digest(checkpoint)}}
      print(f"CLUTTER {group['scene']} {method} obstacles={len(layout['obstacles'])}", flush=True)
      cp.episode(args.out, signature, opts, models=models if method != "rl_true" else None, policy=policy)
      result = json.loads((args.out / "episodes" / method / f"seed_{group['seed']}" / "result.json").read_text())
      rows.append({"scene": group["scene"], "method": method, "success": result["task_success"], "penalties": bool(models and models.use_penalties),
                   "goal_cm": result["final_goal_distance_cm"], "ever_held": result["ever_held"],
                   "obstacle_contacts": result["obstacle_contact_steps"], "fallbacks": result["model_fallback_steps"],
                   "Q_xyz_mae_cm": result.get("Q_on_actual_observations", {}).get("xyz_mae_cm"),
                   "Q_grasp_f1": result.get("Q_on_actual_observations", {}).get("grasp_f1")})
      write_json(args.out / "summary.json", rows)
  for method in args.methods:
    r = [x for x in rows if x["method"] == method]
    print(f"{method}: placements {sum(x['success'] for x in r)}/{len(r)}; goal {np.mean([x['goal_cm'] for x in r]):.1f} cm; "
          f"obstacle contacts {sum(x['obstacle_contacts'] for x in r)}")


if __name__ == "__main__":
  main()
