"""Offline stable-lift decisions: actual outcomes, Q scoring, and frozen D alternatives."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import pathlib

import numpy as np

from .data import digest, load_features, load_run, outcome_metrics, state_arrays, write_json


def save_csv(path, rows):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
    writer.writeheader()
    writer.writerows(rows)


def action_key(actions):
  return tuple(tuple(a) for a in actions)


def decision_groups(manifest, horizons):
  """Use genuine branch points plus the original source; merge identical action prefixes."""
  scenes = defaultdict(list)
  for r in manifest["rollouts"]:
    if r["split"] in ("val", "test"):
      scenes[(r["scene"], r["placement"])].append(r)
  groups = []
  for (scene, placement), rows in sorted(scenes.items()):
    for start in range(min(len(r["actions"]) for r in rows)):
      cohorts = defaultdict(list)
      for r in rows:
        cohorts[action_key(r["actions"][:start])].append(r)
      for prefix, cohort in cohorts.items():
        if len(cohort) < 2 or (start and len({tuple(r["actions"][start]) for r in cohort}) < 2):
          continue
        suffix = hashlib.sha256(json.dumps(prefix).encode()).hexdigest()[:12]
        for h in horizons:
          if any(start+h > len(r["actions"]) for r in cohort):
            continue
          candidates = {}
          for r in sorted(cohort, key=lambda x: x["branch"]):
            actions = r["actions"][start:start+h]
            key = action_key(actions)
            if key not in candidates:
              candidates[key] = {"name": r["branch"], "aliases": [], "rollout": r,
                "ids": r["states"][start:start+h+1], "actions": actions}
            candidates[key]["aliases"].append(r["branch"])
          if len(candidates) > 1:
            groups.append({"id": f"{scene}/{placement}/s{start:02d}/h{h:02d}/{suffix}",
              "scene": scene, "placement": placement, "split": cohort[0]["split"],
              "start": start, "horizon": h, "candidates": list(candidates.values()), "cohort": cohort})
  return groups


def check_sources(root, manifest, z, groups):
  """Same observed starting state/history, and identical outcomes for aliased sequences."""
  hashes = {}
  def frames(state):
    result = []
    for name in state["frames"]:
      if name not in hashes:
        hashes[name] = digest(root / name)
      result.append(hashes[name])
    return result
  def equal(a, b):
    x, y = manifest["states"][a], manifest["states"][b]
    return (np.allclose(x["p"], y["p"], rtol=0, atol=1e-5)
      and np.allclose(x["object_xyz"], y["object_xyz"], rtol=0, atol=1e-6)
      and x["held"] == y["held"] and frames(x) == frames(y)
      and np.allclose(z[a], z[b], rtol=1e-5, atol=1e-5))
  for g in groups:
    source = g["candidates"][0]["ids"][0]
    for r in g["cohort"]:
      if not equal(source, r["states"][g["start"]]):
        raise ValueError(f"different candidate starting states/history: {g['id']}")
      candidate = next(c for c in g["candidates"] if r["branch"] in c["aliases"])
      for a, b in zip(candidate["ids"], r["states"][g["start"]:g["start"]+g["horizon"]+1]):
        if not equal(a, b):
          raise ValueError(f"identical actions have different recorded outcomes: {g['id']}")


def score_inputs(heights_cm, grasp, reference_cm, widths, goal_cm, stable_steps):
  heights_cm, grasp, widths = map(np.asarray, (heights_cm, grasp, widths))
  if (not all(np.isfinite(x).all() for x in (heights_cm, grasp, widths))
      or len(heights_cm) < stable_steps or np.any((grasp < 0) | (grasp > 1))):
    raise ValueError("invalid score trajectory")
  minimum = float(np.min(heights_cm[-stable_steps:]-reference_cm))
  # A 2 mm tolerance is predeclared; width is total opening, physical range 0..8 cm.
  valid = bool(np.min(widths) >= -.002 and np.max(widths) <= .082)
  return {"min_lift_cm": minimum, "min_held_proxy": float(np.min(grasp[-stable_steps:])),
          "height_goal_met": minimum >= goal_cm, "width_valid": valid,
          "min_width_cm": float(100*np.min(widths)), "max_width_cm": float(100*np.max(widths))}


def qualifies(row, threshold):
  return row["height_goal_met"] and row["width_valid"] and row["min_held_proxy"] >= threshold


def choose(rows, threshold):
  eligible = [r for r in rows if qualifies(r, threshold)]
  # No large-height/low-confidence product. If none qualifies, explicitly abstain.
  return max(eligible, key=lambda r: (r["min_held_proxy"], -r["action_cost"], r["candidate"])) if eligible else None


def calibrate(rows, required_precision, minimum):
  sweep = []
  for threshold in (.5, .7, .8, .9, .95, .99):
    proposed = [r for r in rows if qualifies(r, threshold)]
    tp = sum(r["real_success"] for r in proposed)
    positives = sum(r["real_success"] for r in rows)
    sweep.append({"threshold": threshold, "predicted_positive": len(proposed), "true_positive": tp,
      "true_positive_scene_groups": len({r["scene"] for r in proposed if r["real_success"]}),
      "precision": tp/len(proposed) if proposed else None, "recall": tp/positives if positives else None})
  supported = [s for s in sweep if s["true_positive"] >= minimum and s["true_positive_scene_groups"] >= 2
               and s["precision"] >= required_precision]
  selected = max(supported, key=lambda s: (s["recall"], -s["threshold"])) if supported else None
  return {"supported": bool(selected), "threshold": selected["threshold"] if selected else 1.01,
    "source": "fresh validation Q on actual future states; shared by all D variants", "sweep": sweep,
    "note": "Minimum held proxy is not a calibrated probability of joint stable-lift success. "
            "Unsupported calibration forces abstention; no test outcomes select a threshold."}


def summarize(rows, threshold):
  by_group = defaultdict(list)
  for row in rows:
    by_group[row["decision"]].append(row)
  decisions = []
  for group, candidates in by_group.items():
    chosen = choose(candidates, threshold)
    available = any(r["real_success"] for r in candidates)
    reference = next((r for r in candidates if "close_lift" in r["aliases"].split("|")), None)
    decisions.append({"decision": group, "scene": candidates[0]["scene"],
      "start_step": candidates[0].get("start_step"), "source_held": candidates[0].get("source_held"),
      "candidate_count": len(candidates),
      "successful_candidates": sum(r["real_success"] for r in candidates), "any_success": available,
      "chosen": chosen["candidate"] if chosen else "ABSTAIN", "abstained": chosen is None,
      "chosen_success": bool(chosen and chosen["real_success"]),
      "chosen_collision": bool(chosen and chosen["real_collision"]),
      "false_success": bool(chosen and not chosen["real_success"]),
      "regret": int(available)-int(chosen["real_success"]) if chosen else None,
      "reference_available": reference is not None,
      "always_close_lift_success": bool(reference and reference["real_success"]),
      "uniform_random_expected_success": sum(r["real_success"] for r in candidates)/len(candidates)})
  n, selected = len(decisions), [r for r in decisions if not r["abstained"]]
  possible = [r for r in decisions if r["any_success"]]
  reference = [r for r in decisions if r["reference_available"]]
  pred = [qualifies(r, threshold) for r in rows]
  tp = sum(v and r["real_success"] for v, r in zip(pred, rows))
  fp = sum(v and not r["real_success"] for v, r in zip(pred, rows))
  return {"decisions": n, "scene_groups": len({r["scene"] for r in decisions}),
    "candidate_success_coverage": len(possible)/n if n else None,
    "no_success_available": n-len(possible), "chosen_successes": sum(r["chosen_success"] for r in decisions),
    "chosen_success_rate_all_decisions": sum(r["chosen_success"] for r in decisions)/n if n else None,
    "chosen_success_rate_when_success_available": sum(r["chosen_success"] for r in possible)/len(possible) if possible else None,
    "abstention_rate": (n-len(selected))/n if n else None,
    "false_successes": sum(r["false_success"] for r in selected),
    "false_success_rate_selected": sum(r["false_success"] for r in selected)/len(selected) if selected else None,
    "selected_collision_count": sum(r["chosen_collision"] for r in selected),
    "mean_regret_selected": float(np.mean([r["regret"] for r in selected])) if selected else None,
    "candidate_success_precision": tp/(tp+fp) if tp+fp else None,
    "candidate_success_recall": tp/sum(r["real_success"] for r in rows) if any(r["real_success"] for r in rows) else None,
    "candidate_width_invalid_fraction": float(np.mean([not r["width_valid"] for r in rows])) if rows else None,
    "always_close_lift_decisions": len(reference),
    "always_close_lift_success_rate": sum(r["always_close_lift_success"] for r in reference)/len(reference) if reference else None,
    "chosen_success_rate_on_close_lift_decisions": sum(r["chosen_success"] for r in reference)/len(reference) if reference else None,
    "uniform_random_expected_success_rate": float(np.mean([r["uniform_random_expected_success"] for r in decisions])) if n else None,
    "oracle_candidate_success_rate": len(possible)/n if n else None}, decisions


def evaluate(args):
  import torch
  from .train import imagine, load_model, q_gate, readout
  torch.set_num_threads(min(torch.get_num_threads(), 4))
  manifest, original = load_run(args.run), load_run(args.models_run)
  z, meta = load_features(args.run, manifest)
  old_meta = json.loads((args.models_run / "features/meta.json").read_text())
  for k in ("model", "model_revision", "pooling", "camera", "clip_frames"):
    if not meta.get(k) or meta[k] != old_meta[k]:
      raise ValueError(f"encoder contract changed: {k}")
  for k in ("proprio_columns", "action_columns", "control_hz", "clip_padding"):
    if manifest[k] != original[k]:
      raise ValueError(f"simulation observation/action contract changed: {k}")
  old_seeds = {r["simulation_seed"] for r in original["rollouts"]}
  if old_seeds & {r["simulation_seed"] for r in manifest["rollouts"]}:
    raise ValueError("fresh collection reuses an original simulation seed")
  groups = decision_groups(manifest, args.horizons)
  if not groups or {g["split"] for g in groups} != {"val", "test"}:
    raise ValueError("need validation and test decisions with distinct candidates")
  check_sources(args.run, manifest, z, groups)
  p, xyz, held = state_arrays(manifest)
  outputs, summary_rows, lines = [], [], ["Stable-lift score / candidate-choice experiment (offline)"]
  model_files = {}
  for seed in args.seeds:
    tags = {"original": f"seed_{seed}", "robot": f"width_robot_seed_{seed}", "visual": f"width_visual_seed_{seed}"}
    q, qc = load_model(args.models_run, "readout", tags["original"])
    q_hash = digest(args.models_run / "attempts" / tags["original"] / "models/readout.pt")
    real_q_xyz, real_q_g = readout(q, qc, z, p)
    gates = {}
    for split in ("val", "test"):
      ids = [i for i, s in enumerate(manifest["states"]) if s["split"] == split]
      metrics = outcome_metrics(real_q_xyz[ids], real_q_g[ids], xyz[ids], held[ids])
      gates[split] = {"passed": q_gate(metrics, qc["q_limits"]), "metrics": metrics, "limits": qc["q_limits"]}
    models = {}
    for version, tag in tags.items():
      root = args.models_run / "attempts" / tag
      if digest(root / "models/readout.pt") != q_hash:
        raise ValueError(f"Q differs within seed {seed}: {tag}")
      models[version] = load_model(args.models_run, "dynamics", tag)
      for role in ("dynamics", "readout"):
        model_files[f"{tag}/{role}"] = digest(root / "models" / f"{role}.pt")
    candidates, steps = [], []
    for index, g in enumerate(groups, 1):
      for candidate in g["candidates"]:
        ids, actions = candidate["ids"], candidate["actions"]
        root_id = candidate["rollout"]["states"][0]
        actual = [manifest["states"][i] for i in ids[1:]]
        collision = any(s["obstacle_contact_during_step"] or s["table_contact_during_step"]
                        or s["obstacle_contact"] or s["table_contact"] for s in actual)
        answer = score_inputs(100*xyz[ids[1:], 2], held[ids[1:]], 100*xyz[root_id, 2],
                              p[ids[1:], 18], args.goal_cm, args.stable_steps)
        success = bool(answer["height_goal_met"] and answer["min_held_proxy"] == 1 and not collision)
        base = {"decision": g["id"], "scene": g["scene"], "placement": g["placement"], "split": g["split"],
          "start_step": g["start"], "horizon": g["horizon"], "candidate": candidate["name"],
          "aliases": "|".join(candidate["aliases"]), "action_cost": float(np.square(actions).mean()),
          "source_held": bool(held[ids[0]]),
          "actions_json": json.dumps(actions), "real_success": success, "real_collision": collision,
          "real_min_lift_cm": answer["min_lift_cm"], "real_min_held": answer["min_held_proxy"],
          "recorded_simulator_return": sum(s["reward"] for s in actual)}
        forecasts = {"real_Q_diagnostic": (real_q_xyz[ids], real_q_g[ids], p[ids])}
        for version, (d, dc) in models.items():
          zz, pp = imagine(d, dc, z[ids[0]], p[ids[0]], actions)
          predicted_xyz, probability = readout(q, qc, zz, pp)
          forecasts[version] = predicted_xyz, probability, pp
          if version == "original":
            persistent_xyz, persistent_g = readout(q, qc, np.repeat(z[ids[0]][None], len(ids), axis=0), pp)
            forecasts["unchanged_z_predicted_p"] = persistent_xyz, persistent_g, pp
        for version, (position, probability, predicted_p) in forecasts.items():
          scored = score_inputs(100*position[1:, 2], probability[1:], 100*real_q_xyz[root_id, 2],
                                predicted_p[1:, 18], args.goal_cm, args.stable_steps)
          candidates.append({**base, "version": version, **scored})
          for t, target in enumerate(ids[1:], 1):
            steps.append({"decision": g["id"], "candidate": candidate["name"], "version": version,
              "step": t, "real_height_cm": float(100*xyz[target, 2]), "real_held": bool(held[target]),
              "predicted_height_cm": float(100*position[t, 2]), "held_proxy": float(probability[t]),
              "real_width_cm": float(100*p[target, 18]), "predicted_width_cm": float(100*predicted_p[t, 18])})
      if index % 20 == 0:
        print(f"seed={seed} predicted {index}/{len(groups)} decision groups", flush=True)
    calibrations, metrics, decisions = {}, [], []
    for h in args.horizons:
      rows = [r for r in candidates if r["horizon"] == h and r["split"] == "val" and r["version"] == "real_Q_diagnostic"]
      if not rows:
        raise ValueError(f"no genuine validation choices at horizon {h}")
      calibration = calibrate(rows, args.calibration_precision, args.min_calibration_positives)
      calibrations[str(h)] = calibration
      for split in ("val", "test"):
        for version in forecasts:
          rows = [r for r in candidates if r["horizon"] == h and r["split"] == split and r["version"] == version]
          diagnostic = not (gates["val"]["passed"] and gates[split]["passed"] and calibration["supported"])
          common = {"seed": seed, "split": split, "horizon": h, "version": version,
            "threshold": calibration["threshold"], "calibration_supported": calibration["supported"],
            "Q_gate_passed": gates[split]["passed"], "diagnostic_only": diagnostic}
          result, choices = summarize(rows, calibration["threshold"])
          metrics.append({**common, "source_phase": "all", **result})
          for name, value in (("unheld", False), ("held", True)):
            subset = [r for r in rows if r["source_held"] == value]
            part, _ = summarize(subset, calibration["threshold"])
            metrics.append({**common, "source_phase": name, **part})
          decisions += [{**common, **r} for r in choices]
          lines.append(f"seed={seed} {split} h={h} {version}: chosen success={result['chosen_successes']}/{result['decisions']}; "
            f"available={result['candidate_success_coverage']}; abstention={result['abstention_rate']}; "
            f"false success={result['false_successes']}" + (" [DIAGNOSTIC]" if diagnostic else ""))
    folder = args.run.parent / "reports"
    save_csv(folder / f"seed_{seed}_candidates.csv", candidates)
    save_csv(folder / f"seed_{seed}_forecast_steps.csv", steps)
    save_csv(folder / f"seed_{seed}_decisions.csv", decisions)
    report = {"seed": seed, "calibrations": calibrations, "Q_gates": gates, "metrics": metrics,
      "goal_cm": args.goal_cm, "stable_steps": args.stable_steps,
      "manifest_sha256": digest(args.run / "manifest.json"), "models": tags,
      "contract": "Fresh fixed candidate sets; predicted future p only. Real_Q is an actual-future diagnostic, "
        "not a deployable policy. Actual collision labels enter answers only; no collision predictor exists. "
        "Abstention is no selection, not a simulated fallback. All-failure sets do not establish choice quality. "
        "No learned R, CEM or closed-loop pick/place. Overlapping decisions share scene groups."}
    write_json(folder / f"seed_{seed}.json", report)
    summary_rows += metrics
    outputs.append(report)
  folder = args.run.parent / "reports"
  save_csv(folder / "summary.csv", summary_rows)
  write_json(folder / "summary.json", {"reports": outputs, "model_hashes": model_files,
    "models_manifest_sha256": digest(args.models_run / "manifest.json"),
    "fresh_features_sha256": digest(args.run / "features/meta.json"), "horizons": args.horizons})
  (folder / "summary.txt").write_text("\n".join(lines)+"\n")
  print("\n".join(lines), flush=True)


def check():
  def row(height, confidence, success, candidate="a", valid=True):
    return {"decision": "one", "scene": "scene", "candidate": candidate, "aliases": candidate,
      "height_goal_met": height >= 5, "width_valid": valid, "min_held_proxy": confidence,
      "action_cost": .1, "real_success": success, "real_collision": False}
  assert choose([row(.2, .99, False), row(10, .2, False, "b")], .8) is None
  assert choose([row(6, .9, True), row(100, .2, False, "b")], .8)["candidate"] == "a"
  assert choose([row(6, .99, True, valid=False)], .8) is None
  assert not score_inputs([6, 6, 6], [1, 1, 0], 0, [.04]*3, 5, 3)["min_held_proxy"]
  result, _ = summarize([row(6, .9, False)], .8)
  assert result["candidate_success_coverage"] == 0 and result["chosen_success_rate_when_success_available"] is None
  assert result["false_successes"] == 1
  assert not calibrate([row(6, .99, False)], .9, 1)["supported"]
  supported = [{**row(6, .9, True), "scene": f"s{i//2}"} for i in range(4)]
  assert calibrate(supported, .9, 3)["threshold"] == .5
  assert not calibrate(supported[:2], .9, 1)["supported"]  # one scene is insufficient
  sample = {"rollouts": [{"id": name, "scene": "s", "placement": "under", "split": "val",
    "branch": name, "states": [0, 1, 2], "actions": [[0]*5, [0, 0, 0, 0, grip]]}
    for name, grip in (("close_lift", -1), ("alias", -1), ("open_lift", 1))]}
  groups = decision_groups(sample, [1, 2])
  assert len(groups) == 2 and all(len(g["candidates"]) == 2 for g in groups)
  print("PASS: risk/height rule, width rejection, stable window, all-failure sets, calibration and action aliases")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--check", action="store_true", help="small deterministic checks; no Torch/GPU/data needed")
  parser.add_argument("--models-run", type=pathlib.Path)
  parser.add_argument("--run", type=pathlib.Path)
  parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
  parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16, 30])
  parser.add_argument("--goal-cm", type=float, default=5)
  parser.add_argument("--stable-steps", type=int, default=3)
  parser.add_argument("--calibration-precision", type=float, default=.9)
  parser.add_argument("--min-calibration-positives", type=int, default=3)
  args = parser.parse_args()
  if args.check:
    check()
    return
  if (not args.run or not args.models_run or args.goal_cm <= 0 or args.stable_steps < 1
      or min(args.horizons) < args.stable_steps or max(args.horizons) > 30
      or len(set(args.horizons)) != len(args.horizons) or not 0 < args.calibration_precision <= 1
      or args.min_calibration_positives < 1 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)):
    parser.error("provide model/data runs, positive goal, distinct seeds/horizons, valid stable/calibration settings")
  args.horizons = sorted(args.horizons)
  evaluate(args)


if __name__ == "__main__":
  main()
