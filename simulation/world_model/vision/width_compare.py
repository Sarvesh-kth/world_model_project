"""Compare frozen baseline and matched robot/visual width corrections on validation only."""
import argparse
import csv
import json
import pathlib
import sys

from .data import digest, write_json


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True, type=pathlib.Path)
  parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  parser.add_argument("--execute", action="store_true", help="train missing width variants and run their validation stages")
  parser.add_argument("--epochs", type=int, default=60, help="same budget for both width variants")
  args = parser.parse_args()
  if args.epochs < 1 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
    parser.error("epochs must be positive and seeds distinct/nonnegative")
  args.run = args.run.resolve()
  rows, sources = [], {}
  for seed in args.seeds:
    reference, budget = None, None
    for variant, tag in (("baseline", f"seed_{seed}"), ("robot", f"width_robot_seed_{seed}"),
                         ("visual", f"width_visual_seed_{seed}")):
      root = args.run / "attempts" / tag / "reports"
      if args.execute and variant != "baseline":
        from .pipeline import run_command
        attempt = root.parent
        training_path = root / "dynamics_training.json"
        if not attempt.exists():
          run_command([sys.executable, "-m", "world_model.vision.train", "width", "--run", str(args.run),
            "--from-tag", f"seed_{seed}", "--tag", tag, "--seed", str(seed), "--width-input", variant,
            "--epochs", str(args.epochs), "--rollout-steps", "8"], args.run / "logs" / f"{tag}_train.txt")
        if not training_path.is_file() or not (attempt / "models/dynamics.pt").is_file():
          parser.error(f"incomplete attempt {attempt}; preserve it elsewhere before retrying")
        training = json.loads(training_path.read_text())
        if (training["settings"]["epochs"] != args.epochs or training["width_input"] != variant
            or training["seed"] != seed or training["settings"]["rollout_steps"] != 8
            or training["parent_checkpoint_sha256"] != baseline_hash
            or training["checkpoint_sha256"] != digest(attempt / "models/dynamics.pt")):
          parser.error(f"existing {tag} has different settings/weights; preserve this experiment")
        for name, checksum in training["frozen_companions"].items():
          if not (attempt / name).is_file() or digest(attempt / name) != checksum:
            parser.error(f"frozen companion changed: {attempt / name}")
        jobs = [("contact", ["contact", "--horizon", "30", "--reset-horizons", "1", "4", "8"],
                 ["contact_val_h30_reset1-4-8.json", "contact_val_h30_reset1-4-8_robot.csv", "contact_val_h30_reset1-4-8_forecasts.csv"])]
        jobs += [(test, ["test", test, "--split", "val", "--horizon", "30"],
                  [f"{test}_val_h30.json", f"{test}_val_h30.csv"]) for test in ("persistence", "action", "positions")]
        for name, command, files in jobs:
          if all((root / f).is_file() for f in files):
            print(f"SKIP existing {tag} {name}", flush=True)
            continue
          run_command([sys.executable, "-m", "world_model.vision."+command[0], *command[1:],
                       "--run", str(args.run), "--tag", tag], args.run / "logs" / f"{tag}_{name}.txt")
      reports = {}
      for name in ("contact_val_h30_reset1-4-8", "persistence_val_h30", "action_val_h30", "positions_val_h30"):
        path = root / f"{name}.json"
        if not path.is_file():
          parser.error(f"missing {path}; finish this tag's validation commands first")
        reports[name.split("_")[0]] = json.loads(path.read_text())
        sources[str(path)] = digest(path)
      c, persistence, action, positions = (reports[k] for k in ("contact", "persistence", "action", "positions"))
      signature = (c["manifest_sha256"], c["readout_checkpoint_sha256"], c["anchor_counts"],
                   c["original_horizon"], c["reset_horizons"], c["scene_groups"])
      if reference is None:
        reference = signature
        baseline_hash = c["dynamics_checkpoint_sha256"]
        if args.execute:
          for role, checksum in (("dynamics", baseline_hash), ("readout", c["readout_checkpoint_sha256"])):
            if digest(root.parent / "models" / f"{role}.pt") != checksum:
              parser.error(f"baseline {tag}/{role} changed since its contact diagnostic")
      elif signature != reference:
        parser.error(f"{tag} differs in data, Q or evaluation anchors from its baseline")
      if variant != "baseline":
        training = json.loads((root / "dynamics_training.json").read_text())
        if (training["parent_checkpoint_sha256"] != baseline_hash or training["width_input"] != variant
            or training["seed"] != seed or training["checkpoint_sha256"] != c["dynamics_checkpoint_sha256"]):
          parser.error(f"{tag} is not the expected frozen width correction")
        settings = training["settings"]
        current_budget = tuple(settings[k] for k in ("epochs", "batch_size", "lr", "rollout_steps", "seed", "from_tag"))
        if budget is not None and current_budget != budget:
          parser.error(f"{tag} differs in training budget from the robot-only correction")
        budget = current_budget
        sources[str(root / "dynamics_training.json")] = digest(root / "dynamics_training.json")
      for report in reports.values():
        if (report["split"] != "val" or report["manifest_sha256"] != c["manifest_sha256"]
            or report["dynamics_checkpoint_sha256"] != c["dynamics_checkpoint_sha256"]):
          parser.error(f"mixed validation reports for {tag}")
      local = next(m for m in c["local_forecasts"] if m["horizon"] == 8
                   and m["source_phase"] == "held" and m["combination"] == "pred_z_pred_p")
      final = next(m for m in persistence["by_horizon"] if m["horizon"] == 30)["D_Q_outcomes"]
      robot = c["robot_errors"]
      rows.append({"seed": seed, "variant": variant, "tag": tag,
        "q_gate": c["Q_object_interpretation_gate_passed"],
        "held_width_rollout_cm": robot["held"]["from_start_same_targets"]["coordinates"]["gripper_width"]["mae"],
        "transition_width_reset_cm": robot["transition"]["reset_one_step"]["coordinates"]["gripper_width"]["mae"],
        "h8_held_height_mae_cm": local["height_mae_when_held_cm"],
        "h8_grasp_f1": local["grasp_f1"], "h8_held_detected": local["true_held_detected"],
        "h8_held_targets": local["positive_labels"], "h8_false_held": local["false_held_predictions"],
        "h30_held_height_mae_cm": final["height_mae_when_held_cm"], "h30_grasp_f1": final["grasp_f1"],
        "action_height_effect_mae_cm": action["height_effect_mae_cm"],
        "position_height_effect_mae_cm": positions["D_Q_effect_mae_cm"]})
  output = args.run / "reports/width_comparison.json"
  write_json(output, {"rows": rows, "source_report_hashes": sources,
    "contract": "Validation only. Compare visual with robot at matched seed, data, frozen baseline and Q. "
      "Robot width correction sees zero z; original D_z still uses vision in both variants. "
      "Normal forecast metrics use predicted future p, not teacher forcing. "
      "A failed Q gate makes object interpretations diagnostic. Training fits share evaluation scene groups."})
  with output.with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print("seed variant  width-held cm  width-transition cm  h8 held height cm  held detected  FP  h30 held height cm")
  for r in rows:
    print(f"{r['seed']:>4} {r['variant']:<8} {r['held_width_rollout_cm']:>13.3f} "
          f"{r['transition_width_reset_cm']:>20.3f} {r['h8_held_height_mae_cm']:>18.3f} "
          f"{r['h8_held_detected']:>5}/{r['h8_held_targets']:<7} {r['h8_false_held']:>3} "
          f"{r['h30_held_height_mae_cm']:>19.3f}" + ("  Q gate FAIL" if not r["q_gate"] else ""))
  print(f"saved {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
  main()
