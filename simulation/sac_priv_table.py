# Markdown table of every controller per obstacle count from results/sac_priv_eval_<n>/episodes/<method>/seed_*/result.json
#   python sac_priv_table.py
import json, pathlib, numpy as np
root = pathlib.Path(__file__).resolve().parents[1] / "results"
rows = {}
for n in range(4):
  for f in sorted((root / f"sac_priv_eval_{n}" / "episodes").glob("*/seed_*/result.json")) if (root / f"sac_priv_eval_{n}").exists() else []:
    r = json.loads(f.read_text())
    rows.setdefault(r["method"], {}).setdefault(n, []).append(r)
print("| Controller | Obstacles | Placements | Mean final B | Median final B | Contact steps (scenes) | Never held | Decision |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for method, per in rows.items():
  tot = [0, 0, 0]
  for n, rs in sorted(per.items()):
    ok = sum(r["task_success"] for r in rs); d = [r["final_goal_distance_cm"] for r in rs]
    c = sum(r["obstacle_contact_steps"] for r in rs); cs = sum(r["obstacle_contact_steps"] > 0 for r in rs)
    print(f"| {method} | {n} | {ok}/{len(rs)} | {np.mean(d):.1f} cm | {np.median(d):.1f} cm | {c} ({cs}) | {sum(not r['ever_held'] for r in rs)} | {1000 * np.mean([r['mean_decision_seconds'] for r in rs]):.0f} ms |")
    tot[0] += ok; tot[1] += len(rs); tot[2] += c
  print(f"| {method} | all | {tot[0]}/{tot[1]} | | | {tot[2]} | | |")
