"""Run exactly one held-out test: persistence, action, no-vision, or positions."""
import argparse
import csv
import json
import pathlib

import numpy as np

from .data import load_run, load_features, state_arrays, outcome_metrics, write_json, digest
from .train import load_model, readout, imagine, baseline_predict, q_gate


def mse(a, b, scale):
  return float(np.square((a-b)/scale).mean())


def predictions(root, manifest, z, p, split, horizon, tag, baseline=False):
  d, dc = load_model(root, "dynamics", tag)
  q, qc = load_model(root, "readout", tag)
  b, bc = load_model(root, "no_vision", tag) if baseline else (None, None)
  results = {}
  for r in manifest["rollouts"]:
    if r["split"] != split:
      continue
    if len(r["actions"]) < horizon:
      raise ValueError(f"requested horizon {horizon} exceeds {r['id']}")
    source = r["states"][0]
    zz, pp = imagine(d, dc, z[source], p[source], r["actions"][:horizon])
    xyz, grasp = readout(q, qc, zz, pp)
    keep_xyz, keep_grasp = readout(q, qc, np.repeat(z[source][None], horizon+1, axis=0), pp)
    actual_ids = r["states"][:horizon+1]
    real_xyz, real_grasp = readout(q, qc, z[actual_ids], p[actual_ids])
    row = {"rollout": r, "z": zz, "p": pp, "xyz": xyz, "grasp": grasp,
           "persistent_q_xyz": keep_xyz, "persistent_q_grasp": keep_grasp,
           "real_q_xyz": real_xyz, "real_q_grasp": real_grasp}
    if baseline:
      row["no_vision_xyz"], row["no_vision_grasp"] = baseline_predict(
        b, bc, p[source], r["actions"][:horizon])
    if not all(np.isfinite(row[k]).all() for k in ("z", "p", "xyz", "grasp")):
      raise RuntimeError(f"nonfinite imagined rollout {r['id']}; examine training curves/scales")
    results[r["id"]] = row
  if not results:
    raise ValueError(f"no {split} rollouts")
  return results, dc, qc


def endpoint_metrics(rows, truth_xyz, truth_g, horizon, prefix=""):
  target = [r["rollout"]["states"][horizon] for r in rows]
  if prefix == "no_vision":
    xyz = np.concatenate([r["no_vision_xyz"] for r in rows])
    g = np.concatenate([r["no_vision_grasp"] for r in rows])
  else:
    xyz = np.stack([r[prefix+"xyz"][horizon] for r in rows])
    g = np.asarray([r[prefix+"grasp"][horizon] for r in rows])
  return outcome_metrics(xyz, g, truth_xyz[target], truth_g[target])


def persistence(results, z, p, xyz, grasp, dc, horizons):
  by_horizon = []
  for h in horizons:
    z_d, z_keep, p_d, p_keep, ee_errors = [], [], [], [], []
    rows = list(results.values())
    for row in rows:
      start, end = row["rollout"]["states"][0], row["rollout"]["states"][h]
      z_d.append(mse(row["z"][h], z[end], dc["z_std"].numpy()))
      z_keep.append(mse(z[start], z[end], dc["z_std"].numpy()))
      p_d.append(mse(row["p"][h], p[end], dc["p_std"].numpy()))
      p_keep.append(mse(p[start], p[end], dc["p_std"].numpy()))
      ee_errors.append(100*np.abs(row["p"][h, 14:17]-p[end, 14:17]))
    by_horizon.append({"horizon": h, "D_z_mse": float(np.mean(z_d)),
        "persistence_z_mse": float(np.mean(z_keep)), "D_p_mse": float(np.mean(p_d)),
        "persistence_p_mse": float(np.mean(p_keep)),
        "fraction_rollouts_D_beats_persistence_z": float(np.mean(np.asarray(z_d)<z_keep)),
        "D_end_effector_xyz_mae_cm": np.mean(ee_errors, axis=0).tolist(),
        "D_Q_outcomes": endpoint_metrics(rows, xyz, grasp, h),
        "Q_constant_z_with_same_predicted_p": endpoint_metrics(rows, xyz, grasp, h, "persistent_q_"),
        "Q_on_real_future": endpoint_metrics(rows, xyz, grasp, h, "real_q_")})
  return {"by_horizon": by_horizon}


def action_test(results, z, truth_xyz, dc, horizon, minimum_cm):
  pairs = {}
  for row in results.values():
    r = row["rollout"]
    pairs.setdefault((r["scene"], r["placement"]), {})[r["branch"]] = row
  rows = []
  for (scene, placement), pair in pairs.items():
    a, b = pair["close_lift"], pair["open_lift"]
    ra, rb = a["rollout"], b["rollout"]
    ia, ib = ra["states"][horizon], rb["states"][horizon]
    real_effect = 100*(truth_xyz[ia, 2]-truth_xyz[ib, 2])
    predicted_effect = 100*(a["xyz"][horizon, 2]-b["xyz"][horizon, 2])
    scale = dc["z_std"].numpy()
    a_match, b_match = mse(a["z"][horizon], z[ia], scale), mse(b["z"][horizon], z[ib], scale)
    a_swap, b_swap = mse(a["z"][horizon], z[ib], scale), mse(b["z"][horizon], z[ia], scale)
    matched, swapped = (a_match+b_match)/2, (a_swap+b_swap)/2
    rows.append({"scene": scene, "placement": placement,
      "eligible": bool(abs(real_effect) >= minimum_cm),
      "real_height_effect_cm": float(real_effect), "D_Q_height_effect_cm": float(predicted_effect),
      "effect_absolute_error_cm": float(abs(predicted_effect-real_effect)),
      "matched_z_mse": matched, "swapped_z_mse": swapped,
      "close_matched_swapped_z_mse": [a_match, a_swap], "open_matched_swapped_z_mse": [b_match, b_swap],
      "real_branch_z_separation_mse": mse(z[ia], z[ib], scale),
      "matched_beats_swapped": matched < swapped})
  eligible = [r for r in rows if r["eligible"]]
  return {"eligibility": f"absolute real A/B height effect >= {minimum_cm} cm",
          "eligible_pairs": len(eligible), "all_pairs": len(rows),
          "matched_beats_swapped_fraction": float(np.mean([r['matched_beats_swapped'] for r in eligible])) if eligible else None,
          "height_effect_mae_cm": float(np.mean([r['effect_absolute_error_cm'] for r in eligible])) if eligible else None,
          "pairs": rows}


def position_test(results, p, xyz, horizon, minimum_cm, tolerance):
  pairs = {}
  for row in results.values():
    r = row["rollout"]
    pairs.setdefault((r["scene"], r["branch"]), {})[r["placement"]] = row
  rows = []
  for (scene, branch), pair in pairs.items():
    near, far = pair["under"], pair["offset"]
    a, b = near["rollout"], far["rollout"]
    i, j = a["states"][horizon], b["states"][horizon]
    p_error = float(np.max(np.abs(p[a["states"][0]]-p[b["states"][0]])))
    same_actions = a["actions"][:horizon] == b["actions"][:horizon]
    if p_error > tolerance or not same_actions:
      raise ValueError(f"{scene}/{branch}: position test is confounded by robot state/actions")
    real = 100*(xyz[i, 2]-xyz[j, 2])
    predicted = 100*(near["xyz"][horizon, 2]-far["xyz"][horizon, 2])
    no_vision = 100*(near["no_vision_xyz"][0, 2]-far["no_vision_xyz"][0, 2])
    oracle = 100*(near["real_q_xyz"][horizon, 2]-far["real_q_xyz"][horizon, 2])
    rows.append({"scene": scene, "branch": branch, "source_p_max_difference": p_error,
        "identical_actions": same_actions, "eligible": bool(abs(real) >= minimum_cm),
        "real_position_height_effect_cm": float(real), "D_Q_position_height_effect_cm": float(predicted),
        "no_vision_position_height_effect_cm": float(no_vision), "real_Q_height_effect_cm": float(oracle),
        "D_Q_effect_error_cm": float(abs(real-predicted)), "no_vision_effect_error_cm": float(abs(real-no_vision))})
  eligible = [r for r in rows if r["eligible"]]
  return {"eligibility": f"absolute real under/offset height effect >= {minimum_cm} cm",
       "eligible_pairs": len(eligible), "all_pairs": len(rows),
       "D_Q_effect_mae_cm": float(np.mean([r['D_Q_effect_error_cm'] for r in eligible])) if eligible else None,
       "no_vision_effect_mae_cm": float(np.mean([r['no_vision_effect_error_cm'] for r in eligible])) if eligible else None,
       "pairs": rows}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("test", choices=("persistence", "action", "no-vision", "positions"))
  parser.add_argument("--run", required=True, type=pathlib.Path)
  parser.add_argument("--split", choices=("val", "test"), default="test")
  parser.add_argument("--horizon", type=int, default=24)
  parser.add_argument("--minimum-effect-cm", type=float, default=2.0)
  parser.add_argument("--tag", default="")
  args = parser.parse_args()
  if args.horizon < 1 or args.minimum_effect_cm <= 0:
    parser.error("horizon and minimum-effect-cm must be positive")
  if args.tag and (pathlib.Path(args.tag).name != args.tag or args.tag in (".", "..")):
    parser.error("tag must be a single directory name")
  manifest = load_run(args.run)
  z, _ = load_features(args.run, manifest)
  p, xyz, grasp = state_arrays(manifest)
  results, dc, qc = predictions(args.run, manifest, z, p, args.split, args.horizon, args.tag,
                               baseline=args.test in ("no-vision", "positions"))
  rows = list(results.values())
  ids = np.flatnonzero(np.asarray([s["split"] for s in manifest["states"]]) == args.split)
  q, _ = load_model(args.run, "readout", args.tag)
  real_q_xyz, real_q_g = readout(q, qc, z[ids], p[ids])
  real_q = outcome_metrics(real_q_xyz, real_q_g, xyz[ids], grasp[ids])
  output = args.run if not args.tag else args.run / "attempts" / args.tag
  validation = json.loads((output / "reports/readout_validation.json").read_text())
  gate = validation["q_gate_passed"] and q_gate(real_q, qc["q_limits"])
  if args.test == "persistence":
    report = persistence(results, z, p, xyz, grasp, dc, sorted({1, min(8, args.horizon), min(16, args.horizon), args.horizon}))
  elif args.test == "action":
    report = action_test(results, z, xyz, dc, args.horizon, args.minimum_effect_cm)
  elif args.test == "no-vision":
    qp, qpc = load_model(args.run, "readout_p", args.tag)
    future = [r["rollout"]["states"][args.horizon] for r in rows]
    pxyz, pg = readout(qp, qpc, z[future], p[future])
    report = {"D_Q": endpoint_metrics(rows, xyz, grasp, args.horizon),
              "no_vision_sequence_baseline": endpoint_metrics(rows, xyz, grasp, args.horizon, "no_vision"),
              "Q_on_real_future": endpoint_metrics(rows, xyz, grasp, args.horizon, "real_q_"),
              "Q_p_on_measured_future_p": outcome_metrics(pxyz, pg, xyz[future], grasp[future]),
              "baseline_contract": "direct supervised p0+action-prefix -> object xyz/grasp; "
                "no real future p. This is not an architecture-matched latent ablation."}
  else:
    report = position_test(results, p, xyz, args.horizon, args.minimum_effect_cm,
                           manifest["settings"]["pair_tolerance"])
  report = {"test": args.test, "split": args.split, "horizon": args.horizon,
            "scene_groups": len({r['rollout']['scene'] for r in rows}),
            "manifest_sha256": digest(args.run / "manifest.json"),
            "dynamics_checkpoint_sha256": digest(output / "models/dynamics.pt"),
            "Q_validation_gate_passed": validation["q_gate_passed"],
            "Q_on_all_real_evaluation_states": real_q,
            "Q_object_interpretation_gate_passed": gate,
            "note": "Q-derived imagined object results are diagnostic only when the Q gate fails. "
                    "No learned robot control, reward model or CEM is run.", **report}
  write_json(output / "reports" / f"{args.test}_{args.split}_h{args.horizon}.json", report)
  # One readable per-rollout table makes label/forecast mistakes easy to locate.
  fields = ("rollout", "source_height_cm", "real_height_cm", "predicted_height_cm", "real_held",
            "predicted_grasp_probability", "real_Q_height_cm", "real_Q_grasp_probability")
  with (output / "reports" / f"{args.test}_{args.split}_h{args.horizon}.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    for row in rows:
      r = row["rollout"]; end = r["states"][args.horizon]
      writer.writerow(dict(zip(fields, (r["id"], 100*xyz[r["states"][0], 2], 100*xyz[end, 2],
                        100*row["xyz"][args.horizon, 2], bool(grasp[end]), row["grasp"][args.horizon],
                        100*row["real_q_xyz"][args.horizon, 2], row["real_q_grasp"][args.horizon]))))
  print(f"{args.test}: {args.split}, {report['scene_groups']} held-out scene groups, {len(rows)} rollouts")
  print(f"Q interpretation gate: {'PASS' if gate else 'FAIL — object predictions are diagnostic only'}")
  print(json.dumps({k: v for k, v in report.items() if k not in ("pairs", "note")}, indent=2))
  if args.test in ("action", "positions") and report["eligible_pairs"] == 0:
    print("INCONCLUSIVE: no real outcomes meet the effect threshold; use a longer collected horizon.")
  print(f"saved report and per-rollout CSV under {output}/reports")


if __name__ == "__main__":
  main()
