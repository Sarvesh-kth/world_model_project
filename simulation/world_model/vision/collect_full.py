"""Test 1: collect full pick-and-place trajectories locally for training Q and D on the new latent.

Reuses full_task.collect unchanged (frozen SAC for normal cases, scripted recovery after forced drops,
empty and clutter views, same manifest format). The one difference: the cube start is jittered by
--jitter metres per axis (default 3 cm) instead of the hardcoded 1 cm, so the cube position can no
longer be guessed from the robot state alone and Q has to look at the image.
"""
import os

# offscreen rendering through EGL; set before mujoco is imported
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import copy
import json
import pathlib

import numpy as np

from .control_pipeline import frozen_baseline
from .data import digest, provenance, write_json
from .pipeline import SIMULATION
from . import full_task


def wide_layouts(jitter):
  def layouts(base, seed):
    rng = np.random.default_rng(seed)
    empty = copy.deepcopy(base)
    empty["pick"] = (np.asarray(base["pick"]) + rng.uniform(-jitter, jitter, 2)).tolist()
    empty["obstacles"] = []
    clutter = copy.deepcopy(empty)
    ys = rng.choice(np.linspace(-.34, .34, 9), int(rng.integers(1, 4)), replace=False)
    for y in sorted(ys):
      half = rng.uniform([.018, .018, .025], [.03, .028, .1])
      clutter["obstacles"].append({"kind": "box", "pos": [float(rng.uniform(.40, .425)), float(y)],
                                   "yaw": float(rng.uniform(-.3, .3)), "size": half.tolist()})
    return {"empty": empty, "clutter": clutter}
  return layouts


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/full_test1"))
  p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
  p.add_argument("--train-scenes", type=int, default=16)
  p.add_argument("--val-scenes", type=int, default=4)
  p.add_argument("--test-scenes", type=int, default=6)
  p.add_argument("--seed", type=int, default=30464005)
  p.add_argument("--jitter", type=float, default=.03, help="cube start jitter per xy axis in metres")
  args = p.parse_args()
  baseline = json.loads((args.baseline_run / "pipeline.json").read_text())
  cfg, layout = baseline["inputs"]["config"], baseline["inputs"]["layout"]
  evidence = frozen_baseline(args.baseline_run.resolve(), cfg, layout, baseline["inputs"]["options"]["position_jitter"])
  groups = []
  reserved = set(evidence["test_seeds"] + evidence["validation_seeds"])
  for split, n in (("train", args.train_scenes), ("val", args.val_scenes), ("test", args.test_scenes)):
    for _ in range(n):
      i = len(groups)
      if args.seed + i in reserved:
        raise ValueError("seed overlaps the baseline evaluation")
      groups.append({"scene": f"scene_{i:04d}", "seed": args.seed + i, "split": split})
  settings = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}
  signature = {"campaign": full_task.CAMPAIGN, "settings": settings, "config": cfg, "layout": layout,
               "groups": groups, "frozen_baseline": evidence,
               "code_sha256": {str(f.relative_to(SIMULATION)): digest(f) for f in
                               [SIMULATION / "world_model/vision/full_task.py", pathlib.Path(__file__)]},
               "provenance": provenance()}
  args.out.mkdir(parents=True, exist_ok=True)
  write_json(args.out / "collection.json", signature)
  # the only behavioural change: wider cube start randomisation
  full_task.layouts = wide_layouts(args.jitter)
  full_task.collect(args.out, signature)


if __name__ == "__main__":
  main()
