"""Run the existing visual experiment stages sequentially and export compact evidence."""
import argparse
import csv
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

from .data import digest, provenance, write_json

TESTS = ("persistence", "action", "no-vision", "positions")
SIMULATION = pathlib.Path(__file__).resolve().parents[2]


def stages(args):
  root = args.run
  def command(module, *options):
    return [sys.executable, "-u", "-m", "world_model.vision."+module, "--run", str(root), *map(str, options)]
  result = [
    ("collect", command("collect", "--profile", args.profile, "--seed", args.seed,
      "--train-scenes", args.train_scenes, "--val-scenes", args.val_scenes, "--test-scenes", args.test_scenes),
      [root / "manifest.json"]),
    ("audit", command("audit"), [root / "reports/audit.json"]),
    ("encode", command("encode"), [root / "features/meta.json", root / "features/latents.npy"])]
  # Fixed settings for every seed; no selection or adaptation using final test results.
  for seed in args.training_seeds:
    tag = f"seed_{seed}"
    output = root / "attempts" / tag
    for role in ("readout", "dynamics", "baseline"):
      model_roles = {"readout": ("readout", "readout_p"), "dynamics": ("dynamics",), "baseline": ("no_vision",)}[role]
      files = [output / "models" / f"{r}.pt" for r in model_roles]
      files += [output / "reports" / f"{r}_training.json" for r in model_roles]
      if role == "readout":
        files.append(output / "reports/readout_validation.json")
      result.append((f"{tag}_train_{role}", command("train", role, "--tag", tag, "--seed", seed,
         "--epochs", args.epochs, "--rollout-steps", args.rollout_steps), files))
    for split in ("val", "test"):
      for test in TESTS:
        stem = f"{test}_{split}_h{args.horizon}"
        result.append((f"{tag}_{stem}", command("test", test, "--tag", tag, "--split", split, "--horizon", args.horizon),
                       [output / "reports" / (stem+suffix) for suffix in (".json", ".csv")]))
  return result


def run_command(command, log):
  env = os.environ.copy()
  env["PYTHONUNBUFFERED"] = "1"
  if sys.platform.startswith("linux"):
    # Preserve explicit backend choices; the course container uses the existing OSMesa bootstrap.
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open("a") as stream:
    stream.write("\nCOMMAND " + json.dumps(command) + "\n"); stream.flush()
    child = subprocess.Popen(command, cwd=SIMULATION, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
      for line in child.stdout:
        print(line, end="", flush=True); stream.write(line); stream.flush()
      code = child.wait()
    except BaseException:
      child.terminate(); child.wait()
      raise
  if code:
    raise RuntimeError(f"stage exited {code}; inspect {log}")


def archive_training(files, root):
  existing = [p for p in files if p.exists()]
  if existing:
    folder = root / "logs/interrupted" / str(time.time_ns())
    for p in existing:
      target = folder / p.relative_to(root)
      target.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(str(p), target)
    print(f"Preserved incomplete training attempt under {folder}; restarting this stage", flush=True)


def metric_rows(root, seeds, horizon):
  rows = []
  for seed in seeds:
    folder = root / "attempts" / f"seed_{seed}" / "reports"
    for role in ("readout", "readout_p", "dynamics", "no_vision"):
      training = json.loads((folder / f"{role}_training.json").read_text())
      rows.append({"seed": seed, "split": "val", "test": "training_"+role,
                   "best_epoch": training["best_epoch"], "best_validation_loss": training["best_validation_loss"]})
    validation = json.loads((folder / "readout_validation.json").read_text())
    for role in ("readout", "readout_p"):
      rows.append({"seed": seed, "split": "val", "test": role, "q_validation_gate": validation["q_gate_passed"],
                   **validation[role]})
    for split in ("val", "test"):
      for test in TESTS:
        report = json.loads((folder / f"{test}_{split}_h{horizon}.json").read_text())
        base = {"seed": seed, "split": split, "test": test, "scene_groups": report["scene_groups"],
                "q_interpretation_gate": report["Q_object_interpretation_gate_passed"]}
        if test == "persistence":
          for point in report["by_horizon"]:
            rows.append({**base, **{k: v for k, v in point.items() if not isinstance(v, dict)},
              **{"D_Q_"+k: v for k, v in point["D_Q_outcomes"].items()},
              **{"constant_z_"+k: v for k, v in point["Q_constant_z_with_same_predicted_p"].items()}})
        elif test == "no-vision":
          for model in ("D_Q", "no_vision_sequence_baseline", "Q_on_real_future", "Q_p_on_measured_future_p"):
            rows.append({**base, "model": model, "horizon": horizon, **report[model]})
        else:
          rows.append({**base, "horizon": horizon, **{k: v for k, v in report.items()
             if k in ("eligible_pairs", "all_pairs", "matched_beats_swapped_fraction", "height_effect_mae_cm",
                      "D_Q_effect_mae_cm", "no_vision_effect_mae_cm")}})
          if test == "action" and report.get("additional_action_comparisons", {}).get("all_pairs"):
            rows.append({**base, "test": "additional_actions", "horizon": horizon,
                         **{k: v for k, v in report["additional_action_comparisons"].items() if k != "pairs"}})
  return rows


def export_results(args, state):
  target = args.export_dir
  if target.exists():
    info = target / "run_info.json"
    if not info.exists() or json.loads(info.read_text())["campaign"] != state["campaign"]:
      raise ValueError(f"export folder belongs to another campaign: {target}")
  target.mkdir(parents=True, exist_ok=True)
  # Explicit whitelist: no frames, latent arrays, checkpoints, agent notes or interrupted weights.
  for folder in [args.run / "reports", args.run / "logs"] + [
      args.run / "attempts" / f"seed_{s}" / "reports" for s in args.training_seeds]:
    for source in folder.glob("*"):
      if source.is_file() and source.suffix in (".json", ".csv", ".txt"):
        destination = target / source.relative_to(args.run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
  manifest = json.loads((args.run / "manifest.json").read_text())
  scenes = {}
  for rollout in manifest["rollouts"]:
    key = rollout["scene"]+"/"+rollout["placement"]
    scenes.setdefault(key, {"split": rollout["split"], "anchor_xy": rollout.get("anchor_xy"),
      "simulation_seed": rollout.get("simulation_seed"), "parameters": rollout.get("parameters"),
      "initial_robot_state": manifest["states"][rollout["states"][0]]["p"],
      "initial_object_xyz": manifest["states"][rollout["states"][0]]["object_xyz"]})
  write_json(target / "scene_settings.json", scenes)
  write_json(target / "run_info.json", {"campaign": state["campaign"], "provenance": state["provenance"],
    "manifest_sha256": digest(args.run / "manifest.json"), "collection_settings": manifest["settings"],
    "collection_config": manifest["config"], "rejected_attempts": manifest.get("rejected_attempts", []),
    "features": {k: v for k, v in json.loads((args.run / "features/meta.json").read_text()).items() if k != "keys"},
    "pipeline": state, "note": "Observed evaluations; Q failures remain diagnostic. No reward model or learned control."})
  rows = metric_rows(args.run, args.training_seeds, args.horizon)
  with (target / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
    writer.writeheader(); writer.writerows(rows)
  lines = ["Visual consequence campaign", f"profile={args.profile}; collection seed={args.seed}",
           f"train/val/test scenes={args.train_scenes}/{args.val_scenes}/{args.test_scenes}",
           f"epochs={args.epochs}; recursive training steps={args.rollout_steps}; evaluation steps={args.horizon}",
           "Q gate failures are diagnostic results, not software failures. Lower MAE/MSE is better."]
  for row in rows:
    prefix = f"seed={row['seed']} {row['split']}"
    if row["test"] == "persistence" and row["horizon"] == args.horizon:
      lines.append(f"{prefix}: latent MSE D={row['D_z_mse']:.4f}, unchanged={row['persistence_z_mse']:.4f}; "
                   f"D+Q height MAE={row['D_Q_height_mae_cm']:.3f} cm, grasp F1={row['D_Q_grasp_f1']:.3f}")
    elif row["test"] == "action":
      lines.append(f"{prefix}: close/open matched-beats-swapped={row['matched_beats_swapped_fraction']}; "
                   f"height-effect MAE={row['height_effect_mae_cm']} cm; eligible={row['eligible_pairs']}/{row['all_pairs']}")
    elif row["test"] == "no-vision" and row["model"] in ("D_Q", "no_vision_sequence_baseline"):
      lines.append(f"{prefix} {row['model']}: height MAE={row['height_mae_cm']:.3f} cm, grasp F1={row['grasp_f1']:.3f}")
    if row["test"] == "positions":
      lines.append(f"seed={row['seed']} {row['split']}: Q gate={row['q_interpretation_gate']}; "
                   f"position-effect MAE D+Q={row['D_Q_effect_mae_cm']} cm; blind={row['no_vision_effect_mae_cm']} cm; "
                   f"eligible={row['eligible_pairs']}/{row['all_pairs']}")
  (target / "summary.txt").write_text("\n".join(lines)+"\n")
  write_json(target / "files.json", {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob("*"))
             if p.is_file() and p.name != "files.json"})
  print("\n"+"\n".join(lines), flush=True)
  print(f"Full logs: {args.run}/logs\nShareable reports: {target}\nRead summary.csv and the individual JSON/CSV reports.")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--run", required=True, type=pathlib.Path)
  p.add_argument("--profile", choices=("controlled", "varied"), default="varied")
  p.add_argument("--train-scenes", type=int, default=60)
  p.add_argument("--val-scenes", type=int, default=12)
  p.add_argument("--test-scenes", type=int, default=12)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
  p.add_argument("--epochs", type=int, default=60)
  p.add_argument("--rollout-steps", type=int, default=8)
  p.add_argument("--horizon", type=int, help="defaults to the collected sequence length (varied 30, controlled 24)")
  p.add_argument("--export-dir", type=pathlib.Path)
  p.add_argument("--resume", action="store_true")
  p.add_argument("--plan", action="store_true", help="print ordered stages without running or creating files")
  args = p.parse_args()
  if args.horizon is None:
    args.horizon = 30 if args.profile == "varied" else 24
  length = 30 if args.profile == "varied" else 24
  if min(args.train_scenes, args.val_scenes, args.test_scenes, args.epochs, args.rollout_steps, args.horizon) < 1 or max(args.rollout_steps, args.horizon) > length:
    p.error(f"counts/epochs must be positive; rollout-steps and horizon must be between 1 and {length}")
  if len(set(args.training_seeds)) != len(args.training_seeds) or min(args.training_seeds) < 0:
    p.error("training-seeds must be distinct nonnegative integers")
  args.run = args.run.resolve()
  args.export_dir = (args.export_dir or SIMULATION.parent / "results" / args.run.name).resolve()
  if args.export_dir == args.run or args.run in args.export_dir.parents or args.export_dir in args.run.parents:
    p.error("export directory must be separate from the raw run directory")
  ordered = stages(args)
  if args.plan:
    for i, (name, command, _) in enumerate(ordered, 1):
      print(f"{i:02d} {name}: "+json.dumps(command))
    print(f"Export: {args.export_dir}")
    return
  campaign = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items() if k not in ("resume", "plan")}
  if args.export_dir.exists():
    info = args.export_dir / "run_info.json"
    if not info.exists() or json.loads(info.read_text())["campaign"] != campaign:
      p.error("export folder exists for another campaign; use a new run/export name")
  path = args.run / "pipeline.json"
  if args.resume:
    state = json.loads(path.read_text())
    if state["campaign"] != campaign:
      p.error("resume configuration differs; use the original command with --resume")
  else:
    if args.run.exists() and any(args.run.iterdir()):
      p.error("run directory is not empty; use --resume for this pipeline or a new run name")
    state = {"campaign": campaign, "provenance": provenance(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "stages": {}, "complete": False}
    write_json(path, state)
  try:
    if any(not state["stages"].get(name, {}).get("complete") and (name == "encode" or "_train_" in name)
           for name, _, _ in ordered):
      run_command([sys.executable, "-c", "import torch, transformers, cv2, mujoco; "
        "assert torch.cuda.is_available(), 'CUDA unavailable; run on the notebook'; "
        "print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__)"], args.run / "logs/preflight.txt")
    for i, (name, command, files) in enumerate(ordered, 1):
      previous = state["stages"].get(name, {})
      if previous.get("complete"):
        if any(not pathlib.Path(f).exists() or digest(f) != checksum for f, checksum in previous["artifacts"].items()):
          raise ValueError(f"completed {name} artifacts changed; preserve this run and start a new one")
        print(f"[{i}/{len(ordered)}] SKIP completed {name}", flush=True)
        continue
      if previous and "_train_" in name:
        archive_training(files, args.run)
      if name == "collect" and (args.run / "manifest.json").exists():
        if json.loads((args.run / "manifest.json").read_text()).get("complete"):
          command = None
        else:
          command += ["--resume"]
      if name == "encode" and (args.run / "features").exists():
        if (args.run / "features/meta.json").exists():
          command = None
        else:
          command += ["--resume"]
      print(f"\n[{i}/{len(ordered)}] START {name}", flush=True)
      state["stages"][name] = {"complete": False, "command": command}
      write_json(path, state)
      started = time.monotonic()
      if command:
        run_command(command, args.run / "logs" / (name+".txt"))
      state["stages"][name].update(complete=True, seconds=time.monotonic()-started,
                                   artifacts={str(f): digest(f) for f in files})
      write_json(path, state)
    state.update(complete=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state.pop("error", None)
    write_json(path, state)
    export_results(args, state)
  except BaseException as error:
    state["complete"] = False
    state["error"] = str(error) or type(error).__name__
    write_json(path, state)
    print(f"\nSTOPPED: {state['error']}\nLogs: {args.run}/logs\nRerun the same command with --resume.", flush=True)
    raise


if __name__ == "__main__":
  main()
