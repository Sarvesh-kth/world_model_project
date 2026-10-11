import argparse
import json
import pathlib
from collections import Counter

import numpy as np

# Side by side, per run folder: placements and how the other episodes failed, from the files the episodes wrote
# (no models, no GPU). Categories: placed; never reached the cube (gripper never within 5 cm); grasp failed
# (held under 10 steps); grasped, not placed.
#   python -m controller.compare data/eval_plain_scoring data/eval_rule_gripper data/eval_pure_no_steady data/eval_default

KINDS = ("placed", "never reached", "grasp failed", "grasped, not placed")


def classify(folder):
  result = json.loads((folder / "result.json").read_text())
  t = np.load(folder / "trajectory.npz")
  closest = np.linalg.norm(t["p"][:, 14:17] - t["object_xyz"], axis=1).min()
  if result["task_success"]:
    return "placed", result
  if closest > .05:
    return "never reached", result
  return ("grasp failed" if t["held"].sum() < 10 else "grasped, not placed"), result


def main():
  p = argparse.ArgumentParser(description="placements and failure kinds of several controller runs side by side")
  p.add_argument("runs", nargs="+", type=pathlib.Path, help="run folders data/<out>")
  p.add_argument("--method", default="mpc")
  p.add_argument("--seeds", action="store_true", help="also list the outcome per seed")
  args = p.parse_args()
  print(f"{'run':28s} {'episodes':>8s} " + " ".join(f"{k:>20s}" for k in KINDS) + "   median cm to B (placed)")
  per_seed = {}
  for run in args.runs:
    episodes = sorted((run / "episodes" / args.method).glob("seed_*"))
    episodes = [e for e in episodes if (e / "result.json").exists()]
    counts, distances = Counter(), []
    for e in episodes:
      kind, result = classify(e)
      counts[kind] += 1
      per_seed.setdefault(e.name, {})[run.name] = kind
      if kind == "placed":
        distances.append(result["final_goal_distance_cm"])
    median = f"{np.median(distances):.1f}" if distances else "-"
    print(f"{run.name:28s} {len(episodes):8d} " + " ".join(f"{counts[k]:20d}" for k in KINDS) + f"   {median}")
  if args.seeds:
    print()
    print("seed".ljust(16) + "".join(r.name[:18].ljust(20) for r in args.runs))
    for seed, outcome in sorted(per_seed.items()):
      print(seed.ljust(16) + "".join(outcome.get(r.name, "-").ljust(20) for r in args.runs))


if __name__ == "__main__":
  main()
