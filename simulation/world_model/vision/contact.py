"""Diagnose robot coordinates and short visual forecasts reset around grasp transitions."""
import argparse
import csv
import json
import pathlib

import numpy as np

from .data import load_run, load_features, state_arrays, outcome_metrics, write_json, digest, provenance
from .train import load_model, imagine, readout, q_gate
from .test import predictions, mse


def phase(states, ids, t):
  """A grasp-transition proxy from recorded labels; not a finger-contact sensor."""
  current = states[ids[t]]
  held = current["held"]
  neighbours = [states[ids[k]]["held"] for k in (t-1, t+1) if 0 <= k < len(ids)]
  if any(value != held for value in neighbours) or (current["grasped_during_step"] and not held):
    return "transition"
  return "held" if held else "unheld"


def robot_metrics(predicted, actual, columns, scale):
  error = np.asarray(predicted)-np.asarray(actual)
  # Yaw is periodic; joints use their measured bounded-angle values.
  error[:, 17] = (error[:, 17]+np.pi) % (2*np.pi)-np.pi
  factors = np.asarray([180/np.pi]*14 + [100]*3 + [180/np.pi, 100, 1])
  units = ["degrees"]*7 + ["degrees/s"]*7 + ["cm"]*3 + ["degrees", "cm", "command"]
  return {"n": len(error), "coordinates": {
    name: {"mae": float(np.abs(error[:, i]).mean()*factors[i]), "unit": units[i],
           "normalized_mae": float(np.abs(error[:, i]/scale[i]).mean())}
    for i, name in enumerate(columns)},
    "command_sign_mismatch_fraction": float(np.mean((predicted[:, 19] > 0) != (actual[:, 19] > 0)))}


def robot_row(rollout, start, target, source_phase, mode, predicted, actual, columns):
  return {"rollout": rollout, "start_step": start, "target_step": target,
          "label_step": target-1, "source_phase": source_phase, "mode": mode,
          **{f"real_{name}": float(actual[i]) for i, name in enumerate(columns)},
          **{f"pred_{name}": float(predicted[i]) for i, name in enumerate(columns)}}


def diagnose(manifest, z, p, xyz, grasp, results, d, dc, q, qc, horizons):
  columns = manifest["proprio_columns"]
  scale = dc["p_std"].numpy()
  anchors, robot_rows, from_start = [], [], []
  maximum = max(horizons)
  for row in results.values():
    r, predicted = row["rollout"], row["p"]
    ids = r["states"]
    if len(ids) != len(r["actions"])+1:
      raise ValueError(f"unaligned state/action lengths: {r['id']}")
    for t in range(len(predicted)-1):
      source, target = manifest["states"][ids[t]], manifest["states"][ids[t+1]]
      if (source["split"] != "val" or target["split"] != "val"
          or source["scene"] != r["scene"] or target["scene"] != r["scene"]
          or target["action_from_previous"] != r["actions"][t]):
        raise ValueError(f"split/scene/action alignment mismatch: {r['id']} step {t}")
      if not np.isclose(target["time"]-source["time"], 1/manifest["control_hz"], rtol=0, atol=1e-6):
        raise ValueError(f"recorded time gap: {r['id']} step {t}")
    for t in range(1, len(predicted)):
      source_phase = phase(manifest["states"], ids, t-1)
      from_start.append((source_phase, predicted[t], p[ids[t]]))
      robot_rows.append(robot_row(r["id"], 0, t, source_phase, "from_start",
                                 predicted[t], p[ids[t]], columns))
    # Same anchors for every local horizon; keep all eligible states, including unheld controls.
    for t in range(len(predicted)-maximum):
      actual_ids = ids[t:t+maximum+1]
      actions = r["actions"][t:t+maximum]
      zz, pp = imagine(d, dc, z[ids[t]], p[ids[t]], actions)
      forced_z, _ = imagine(d, dc, z[ids[t]], p[ids[t]], actions, robot_states=p[actual_ids])
      if not all(np.isfinite(a).all() for a in (zz, pp, forced_z)):
        raise RuntimeError(f"nonfinite local prediction: {r['id']} step {t}")
      source_phase = phase(manifest["states"], ids, t)
      anchors.append({"rollout": r["id"], "start": t, "ids": actual_ids,
                      "phase": source_phase, "z": zz, "p": pp, "forced_z": forced_z,
                      "original_next_p": predicted[t+1]})
      robot_rows.append(robot_row(r["id"], t, t+1, source_phase, "reset_one_step",
                                 pp[1], p[actual_ids[1]], columns))
      if len(anchors) % 200 == 0:
        print(f"reset {len(anchors)} anchors; visual horizons={horizons}", flush=True)
  if not anchors:
    raise ValueError("no eligible reset anchors; shorten --reset-horizons")
  strata = ["all", "transition", "held", "unheld"]
  robot = {}
  for group in strata:
    full = [r for r in from_start if group == "all" or r[0] == group]
    local = [r for r in anchors if group == "all" or r["phase"] == group]
    robot[group] = {"from_start": robot_metrics(np.stack([r[1] for r in full]),
                                              np.stack([r[2] for r in full]), columns, scale) if full else None}
    if local:
      actual = np.stack([p[r["ids"][1]] for r in local])
      robot[group]["reset_one_step"] = robot_metrics(np.stack([r["p"][1] for r in local]), actual, columns, scale)
      robot[group]["from_start_same_targets"] = robot_metrics(
          np.stack([r["original_next_p"] for r in local]), actual, columns, scale)
  summaries, details = [], []
  for h in horizons:
    target = np.asarray([r["ids"][h] for r in anchors])
    actual_p, actual_z = p[target], z[target]
    predicted_z = np.stack([r["z"][h] for r in anchors])
    combinations = {
      "pred_z_pred_p": (predicted_z, np.stack([r["p"][h] for r in anchors])),
      "pred_z_real_p": (predicted_z, actual_p),
      "forced_z_real_p": (np.stack([r["forced_z"][h] for r in anchors]), actual_p),
      "unchanged_z_real_p": (np.stack([z[r["ids"][0]] for r in anchors]), actual_p),
      "real_z_real_p": (actual_z, actual_p)}
    for name, (zz, pp) in combinations.items():
      position, probability = readout(q, qc, zz, pp)
      if not np.isfinite(position).all() or not np.isfinite(probability).all():
        raise RuntimeError(f"nonfinite Q output at local horizon {h}: {name}")
      for group in strata:
        mask = np.asarray([group == "all" or r["phase"] == group for r in anchors])
        if not mask.any():
          continue
        held, guessed = grasp[target[mask]] >= .5, probability[mask] >= .5
        summaries.append({"horizon": h, "source_phase": group, "combination": name,
          "z_mse": mse(zz[mask], actual_z[mask], dc["z_std"].numpy()),
          **outcome_metrics(position[mask], probability[mask], xyz[target[mask]], grasp[target[mask]]),
          "true_held_detected": int((held & guessed).sum()),
          "false_held_predictions": int((~held & guessed).sum())})
      for anchor, end, pos, prob, latent in zip(anchors, target, position, probability, zz):
        details.append({"rollout": anchor["rollout"], "start_step": anchor["start"],
          "target_step": anchor["start"]+h, "source_phase": anchor["phase"], "horizon": h,
          "combination": name, "real_held": bool(grasp[end]), "predicted_grasp_probability": float(prob),
          "real_height_cm": float(100*xyz[end, 2]), "predicted_height_cm": float(100*pos[2]),
          "z_mse": mse(latent, z[end], dc["z_std"].numpy())})
  return {"anchor_count": len(anchors),
          "anchor_counts": {group: sum(r["phase"] == group for r in anchors) for group in strata[1:]},
          "robot_errors": robot, "local_forecasts": summaries}, robot_rows, details


def save_csv(path, rows):
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True, type=pathlib.Path)
  parser.add_argument("--tag", default="")
  parser.add_argument("--horizon", type=int, default=30, help="original rollout length")
  parser.add_argument("--reset-horizons", nargs="+", type=int, default=[1, 4, 8])
  args = parser.parse_args()
  horizons = sorted(set(args.reset_horizons) | {1})
  if min(horizons) < 1 or max(horizons) > args.horizon:
    parser.error("reset horizons must be positive and no longer than --horizon")
  if args.tag and (pathlib.Path(args.tag).name != args.tag or args.tag in (".", "..")):
    parser.error("tag must be one directory name")
  manifest = load_run(args.run)
  z, _ = load_features(args.run, manifest)
  p, xyz, grasp = state_arrays(manifest)
  if p.shape != (len(z), 20) or len(manifest["proprio_columns"]) != 20 or not np.isfinite(p).all():
    raise ValueError("expected finite 20-coordinate robot states")
  d, dc = load_model(args.run, "dynamics", args.tag)
  q, qc = load_model(args.run, "readout", args.tag)
  results, _, _ = predictions(args.run, manifest, z, p, "val", args.horizon, args.tag)
  print(f"contact diagnostic: val, {len(results)} original rollouts; no new collection/encoding/training", flush=True)
  report, robot_rows, details = diagnose(manifest, z, p, xyz, grasp, results, d, dc, q, qc, horizons)
  output = args.run if not args.tag else args.run / "attempts" / args.tag
  ids = [i for i, s in enumerate(manifest["states"]) if s["split"] == "val"]
  real_xyz, real_g = readout(q, qc, z[ids], p[ids])
  real_q = outcome_metrics(real_xyz, real_g, xyz[ids], grasp[ids])
  validation = json.loads((output / "reports/readout_validation.json").read_text())
  gate = validation["q_gate_passed"] and q_gate(real_q, qc["q_limits"])
  report.update({"test": "contact", "split": "val", "original_horizon": args.horizon,
    "reset_horizons": horizons, "scene_groups": len({r["rollout"]["scene"] for r in results.values()}),
    "Q_object_interpretation_gate_passed": gate, "Q_on_all_real_validation_states": real_q,
    "manifest_sha256": digest(args.run / "manifest.json"),
    "features_metadata_sha256": digest(args.run / "features/meta.json"),
    "dynamics_checkpoint_sha256": digest(output / "models/dynamics.pt"),
    "readout_checkpoint_sha256": digest(output / "models/readout.pt"), "provenance": provenance(),
    "diagnostic_source_sha256": digest(__file__),
    "contract": "Every reset begins with actual z/p; its actions start at the same recorded index. "
      "Forecast z remains recurrent; forced forecasts use recorded current/next p only. "
      "Transition strata use adjacent held changes or transient grasp flags, not exact finger contact. "
      "Labels group results and never enter D. Overlapping anchors are correlated, not independent trials. "
      "Real future p is offline-only. Failed Q gates make object interpretations diagnostic."})
  stem = f"contact_val_h{args.horizon}_reset{'-'.join(map(str, horizons))}"
  write_json(output / "reports" / f"{stem}.json", report)
  save_csv(output / "reports" / f"{stem}_robot.csv", robot_rows)
  save_csv(output / "reports" / f"{stem}_forecasts.csv", details)
  print(f"Q gate: {'PASS' if gate else 'FAIL — object metrics are diagnostic only'}; anchors={report['anchor_counts']}")
  print("Robot errors on identical targets: original rollout versus reset one-step")
  for group in ("transition", "held", "all"):
    metrics = report["robot_errors"][group]
    if "reset_one_step" not in metrics:
      print(f"{group}: no eligible anchors")
      continue
    print(f"{group}: {metrics['reset_one_step']['n']} anchors")
    for name, original in metrics["from_start_same_targets"]["coordinates"].items():
      reset = metrics["reset_one_step"]["coordinates"][name]
      print(f"  {name:<16} {original['mae']:.4f} -> {reset['mae']:.4f} {reset['unit']}")
    print("  command sign mismatch:", metrics["from_start_same_targets"]["command_sign_mismatch_fraction"],
          "->", metrics["reset_one_step"]["command_sign_mismatch_fraction"])
  print("step  source phase  Q inputs              z MSE  held-height cm  held detected  false held")
  for m in report["local_forecasts"]:
    if m["source_phase"] == "unheld":
      continue  # Full unheld control metrics remain in JSON and per-anchor CSV.
    height = "n/a" if m["height_mae_when_held_cm"] is None else f"{m['height_mae_when_held_cm']:.3f}"
    print(f"{m['horizon']:>4}  {m['source_phase']:<12}  {m['combination']:<20}  {m['z_mse']:.4f}  "
          f"{height:>14}  {m['true_held_detected']:>4}/{m['positive_labels']:<9}  {m['false_held_predictions']}")
  print(report["contract"])
  print(f"saved {stem}.json and robot/forecast CSVs under {output}/reports")


if __name__ == "__main__":
  main()
