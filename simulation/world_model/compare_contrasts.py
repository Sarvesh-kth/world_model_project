"""Compare old and retrained D on the same held-out action branches."""

import argparse
import json
import pathlib
import numpy as np


def _reports(folder):
  found = {}
  for path in sorted(folder.glob("*/results.json")):
    report = json.loads(path.read_text())
    source = report["source"]
    key = (source["episode"], source["serial"])
    if key in found:
      raise ValueError(f"duplicate contrast for {key} in {folder}")
    found[key] = report
  if not found:
    raise ValueError(f"no */results.json files in {folder}")
  return found


def _metrics(report):
  real = report["branch_difference"]["actual_ee_x_cm"]
  predicted = report["branch_difference"]["predicted_ee_x_cm"]
  moves = report["ee_x_cm_from_source"]
  return {"real_effect_cm": real, "predicted_effect_cm": predicted,
          "effect_abs_error_cm": abs(predicted - real),
          "branch_x_mae_cm": np.mean([abs(moves[name]["predicted"] - moves[name]["actual"])
                                      for name in ("plus_x", "minus_x")]).item(),
          "effect_sign_correct": bool(np.sign(predicted) == np.sign(real))}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--before", required=True, type=pathlib.Path)
  parser.add_argument("--after", required=True, type=pathlib.Path)
  parser.add_argument("--min-real-effect-cm", type=float, default=0.3,
                      help="include only states where the simulator branches separate by this much")
  parser.add_argument("--out", type=pathlib.Path)
  args = parser.parse_args()
  if args.min_real_effect_cm < 0:
    parser.error("--min-real-effect-cm must be nonnegative")
  before, after = _reports(args.before), _reports(args.after)
  if before.keys() != after.keys():
    parser.error("before and after directories must contain the same episode/serial states")
  rows = []
  for key in sorted(before):
    old, new = _metrics(before[key]), _metrics(after[key])
    if abs(old["real_effect_cm"] - new["real_effect_cm"]) > 0.05:
      parser.error(f"real branch outcome changed between runs for {key}; check replay setup")
    eligible = abs(old["real_effect_cm"]) >= args.min_real_effect_cm
    rows.append({"episode": key[0], "serial": key[1], "eligible": eligible,
                 "before": old, "after": new})
    print(f"{key[0]} serial {key[1]:3d}: real effect={old['real_effect_cm']:+.2f} cm "
          f"D before={old['predicted_effect_cm']:+.2f} "
          f"after={new['predicted_effect_cm']:+.2f} "
          f"{'include' if eligible else 'small real effect; exclude'}")
  eligible = [r for r in rows if r["eligible"]]
  if not eligible:
    parser.error("no states met the real-effect threshold; choose other states or lower it")
  summary = {}
  for phase in ("before", "after"):
    summary[phase] = {
      "mean_effect_abs_error_cm": float(np.mean([r[phase]["effect_abs_error_cm"]
                                                  for r in eligible])),
      "mean_branch_x_mae_cm": float(np.mean([r[phase]["branch_x_mae_cm"]
                                              for r in eligible])),
      "effect_sign_correct": sum(r[phase]["effect_sign_correct"] for r in eligible),
    }
  result = {"min_real_effect_cm": args.min_real_effect_cm,
            "eligible": len(eligible), "total": len(rows),
            "summary": summary, "states": rows}
  print(f"eligible states: {len(eligible)}/{len(rows)}")
  for phase in ("before", "after"):
    value = summary[phase]
    print(f"{phase:6s} mean effect error={value['mean_effect_abs_error_cm']:.2f} cm; "
          f"branch x MAE={value['mean_branch_x_mae_cm']:.2f} cm; "
          f"effect sign correct={value['effect_sign_correct']}/{len(eligible)}")
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved comparison to {args.out}")


if __name__ == "__main__":
  main()
