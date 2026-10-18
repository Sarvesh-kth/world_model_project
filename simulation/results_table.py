import json
import pathlib
import sys

import numpy as np

# The table of every controller per obstacle count, from the result.json files the controllers leave in
# ../results/<prefix>_<count>obstacles_<controller>/ (evaluate.sh writes those). From simulation/:
#   python results_table.py            # the test_* runs
#   python results_table.py other      # runs named other_<count>...

prefix = sys.argv[1] if len(sys.argv) > 1 else "test"
root = pathlib.Path(__file__).resolve().parents[1] / "results"
rows = {}
for n in range(4):
  for f in sorted(root.glob(f"{prefix}_{n}obstacles*/episodes/*/seed_*/result.json")):
    r = json.loads(f.read_text())
    rows.setdefault(r["method"], {}).setdefault(n, {})[r["seed"]] = r
print("| controller | obstacles | placed | mean final distance | median final distance | contact steps (scenes) | never held | decision |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for method, per in rows.items():
  placed = episodes = contacts = 0
  for n, results in sorted(per.items()):
    rs = list(results.values())
    ok = sum(r["task_success"] for r in rs)
    d = [r["final_goal_distance_cm"] for r in rs]
    c = sum(r["obstacle_contact_steps"] for r in rs)
    scenes = sum(r["obstacle_contact_steps"] > 0 for r in rs)
    ms = 1000 * np.mean([r["mean_decision_seconds"] for r in rs])
    print(f"| {method} | {n} | {ok}/{len(rs)} | {np.mean(d):.1f} cm | {np.median(d):.1f} cm | {c} ({scenes}) | {sum(not r['ever_held'] for r in rs)} | {ms:.0f} ms |")
    placed, episodes, contacts = placed + ok, episodes + len(rs), contacts + c
  print(f"| {method} | all | {placed}/{episodes} | | | {contacts} | | |")
