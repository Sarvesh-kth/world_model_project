"""Collect fresh alternatives, encode, evaluate stable-lift decisions, and export evidence."""
import argparse
import datetime
import json
import pathlib
import shutil
import sys
import time

from .data import digest, load_features, load_run, provenance, write_json
from .pipeline import SIMULATION, run_command


def inputs(args):
  manifest = load_run(args.models_run)
  _, features = load_features(args.models_run, manifest)
  if not manifest.get("complete") or manifest["settings"]["profile"] != "varied":
    raise ValueError("use the completed varied vision_v2 run and its frozen width corrections")
  if not features.get("model_revision"):
    raise ValueError("source encoder revision missing; cannot pin the fresh cache")
  if args.seed in {r["simulation_seed"] for r in manifest["rollouts"]}:
    raise ValueError("choose a fresh collection seed")
  frozen = {}
  for seed in args.seeds:
    base = args.models_run / "attempts" / f"seed_{seed}"
    baseline = digest(base / "models/dynamics.pt")
    q_hash = digest(base / "models/readout.pt")
    for variant, tag in (("original", f"seed_{seed}"), ("robot", f"width_robot_seed_{seed}"),
                         ("visual", f"width_visual_seed_{seed}")):
      root = args.models_run / "attempts" / tag
      training = json.loads((root / "reports/dynamics_training.json").read_text())
      checkpoint = digest(root / "models/dynamics.pt")
      if training["checkpoint_sha256"] != checkpoint or training["seed"] != seed:
        raise ValueError(f"training/checkpoint mismatch: {tag}")
      if digest(root / "models/readout.pt") != q_hash:
        raise ValueError(f"Q differs from the original: {tag}")
      if variant != "original" and (training["parent_checkpoint_sha256"] != baseline
                                    or training["width_input"] != variant):
        raise ValueError(f"not the expected frozen width correction: {tag}")
      frozen[tag] = {"dynamics_sha256": checkpoint, "readout_sha256": q_hash,
                     "selected_epoch": training["best_epoch"]}
  first = manifest["rollouts"][0]
  replay = args.models_run / "episodes" / first["scene"] / first["placement"] / "replay.json"
  layout = json.loads(replay.read_text())["layout"]
  campaign = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items() if k != "resume"}
  code = [SIMULATION / "world_model/vision" / f"{name}.py" for name in
          ("decision_pipeline", "decision", "collect", "encode", "audit", "data", "train")]
  code += list((SIMULATION / "environment").glob("*.py"))
  code += [SIMULATION / "world_model/train_dynamics.py", SIMULATION / "world_model/prepare.py",
           SIMULATION / "data_collection/scripted_policy.py", SIMULATION / "world_model/vision/pipeline.py"]
  return {"campaign": campaign, "frozen_models": frozen,
    "source_manifest_sha256": digest(args.models_run / "manifest.json"),
    "source_features_sha256": digest(args.models_run / "features/meta.json"),
    "encoder": {k: features[k] for k in ("model", "model_revision", "pooling", "camera", "clip_frames")},
    "config": manifest["config"], "layout": layout,
    "collection": {k: manifest["settings"][k] for k in
                    ("hold_steps", "lift_steps", "lift_action", "offset", "pair_tolerance")},
    "code_sha256": {str(p.relative_to(SIMULATION)): digest(p) for p in sorted(code)}}


def export(args, state):
  target = args.export_dir
  if target.exists() and (not (target / "run_info.json").is_file()
      or json.loads((target / "run_info.json").read_text())["inputs"] != state["inputs"]):
    raise ValueError(f"export belongs to a different experiment: {target}")
  target.mkdir(parents=True, exist_ok=True)
  for folder in (args.out / "reports", args.out / "logs"):
    for source in folder.glob("*"):
      if source.is_file() and source.suffix in (".json", ".csv", ".txt"):
        destination = target / source.relative_to(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
  fresh = args.out / "observations"
  shutil.copyfile(fresh / "reports/audit.json", target / "collection_audit.json")
  manifest = load_run(fresh)
  scenes = {}
  for r in manifest["rollouts"]:
    key = r["scene"]+"/"+r["placement"]
    scenes.setdefault(key, {"split": r["split"], "seed": r["simulation_seed"],
      "anchor_xy": r["anchor_xy"], "parameters": r["parameters"], "actions": {}})["actions"][r["branch"]] = r["actions"]
  write_json(target / "scene_settings.json", scenes)
  write_json(target / "run_info.json", {**state, "manifest_sha256": digest(fresh / "manifest.json"),
    "features": {k: v for k, v in json.loads((fresh / "features/meta.json").read_text()).items() if k != "keys"},
    "note": "Frozen models, fresh scene groups, no learned R/CEM/control. "
            "Collection's train split is unused: retained for the shared collector/audit contract."})
  write_json(target / "files.json", {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob("*"))
                                   if p.is_file() and p.name != "files.json"})
  print((args.out / "reports/summary.txt").read_text(), flush=True)
  print(f"COMPLETE: reports/logs exported to {target}\nRaw frames/cache remain under {fresh}", flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("data/vision_v2"))
  parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/decision_v1"))
  parser.add_argument("--export-dir", type=pathlib.Path)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--seed", type=int, default=20261004, help="fresh collection seed; distinct from model fitting seeds")
  parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="frozen model fitting seeds")
  parser.add_argument("--val-scenes", type=int, default=6)
  parser.add_argument("--test-scenes", type=int, default=12)
  parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16, 30])
  parser.add_argument("--goal-cm", type=float, default=5)
  parser.add_argument("--stable-steps", type=int, default=3)
  parser.add_argument("--calibration-precision", type=float, default=.9)
  parser.add_argument("--min-calibration-positives", type=int, default=3)
  args = parser.parse_args()
  if (min(args.val_scenes, args.test_scenes, args.stable_steps, args.min_calibration_positives) < 1
      or min(args.horizons) < args.stable_steps or max(args.horizons) > 30 or args.goal_cm <= 0
      or not 0 < args.calibration_precision <= 1 or args.seed < 0 or min(args.seeds) < 0
      or len(set(args.seeds)) != len(args.seeds) or len(set(args.horizons)) != len(args.horizons)):
    parser.error("invalid scene/seed/goal/horizon/calibration settings")
  args.models_run, args.out = args.models_run.resolve(), args.out.resolve()
  args.export_dir = (args.export_dir or SIMULATION.parent / "results" / args.out.name).resolve()
  args.horizons = sorted(args.horizons)
  if args.out == args.models_run or args.models_run in args.out.parents:
    parser.error("keep the new experiment outside the source model run")
  signature = inputs(args)
  path = args.out / "pipeline.json"
  if path.exists():
    if not args.resume:
      parser.error("experiment already exists; rerun the same command with --resume")
    state = json.loads(path.read_text())
    if state["inputs"] != signature:
      parser.error("inputs/settings/code changed; preserve this run and use a new --out name")
  else:
    if args.out.exists() and any(args.out.iterdir()):
      parser.error("output directory is not empty; preserve it and use a new name")
    state = {"inputs": signature, "provenance": provenance(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "stages": {}, "complete": False}
    write_json(path, state)
  config, layout = args.out / "config.json", args.out / "layout.json"
  write_json(config, signature["config"])
  write_json(layout, signature["layout"])
  fresh = args.out / "observations"
  collection = signature["collection"]
  collect = [sys.executable, "-u", "-m", "world_model.vision.collect", "--run", str(fresh),
    "--profile", "varied", "--decision-branches", "--seed", str(args.seed), "--config", str(config),
    "--layout", str(layout), "--train-scenes", "4", "--val-scenes", str(args.val_scenes),
    "--test-scenes", str(args.test_scenes), "--camera", signature["encoder"]["camera"]]
  for name, value in collection.items():
    collect += ["--"+name.replace("_", "-"), str(value)]
  evaluate = [sys.executable, "-u", "-m", "world_model.vision.decision", "--models-run", str(args.models_run),
              "--run", str(fresh), "--seeds", *map(str, args.seeds), "--horizons", *map(str, args.horizons)]
  for name in ("goal_cm", "stable_steps", "calibration_precision", "min_calibration_positives"):
    evaluate += ["--"+name.replace("_", "-"), str(getattr(args, name))]
  files = [args.out / "reports" / name for name in ("summary.json", "summary.csv", "summary.txt")]
  files += [args.out / "reports" / f"seed_{seed}{suffix}" for seed in args.seeds
            for suffix in (".json", "_candidates.csv", "_forecast_steps.csv", "_decisions.csv")]
  jobs = [("collect", collect, [fresh / "manifest.json"]),
          ("audit", [sys.executable, "-u", "-m", "world_model.vision.audit", "--run", str(fresh)], [fresh / "reports/audit.json"]),
          ("encode", [sys.executable, "-u", "-m", "world_model.vision.encode", "--run", str(fresh),
                      "--model", signature["encoder"]["model"], "--revision", signature["encoder"]["model_revision"]],
                      [fresh / "features/meta.json", fresh / "features/latents.npy"]),
          ("decisions", evaluate, files)]
  try:
    if not state["stages"].get("encode", {}).get("complete"):
      run_command([sys.executable, "-c", "import torch, transformers, cv2, mujoco; "
        "assert torch.cuda.is_available(), 'Use the CUDA notebook environment'; "
        "print('GPU:', torch.cuda.get_device_name(0), 'CUDA:', torch.version.cuda)"], args.out / "logs/preflight.txt")
    for i, (name, command, artifacts) in enumerate(jobs, 1):
      previous = state["stages"].get(name, {})
      if previous.get("complete"):
        if any(not pathlib.Path(p).is_file() or digest(p) != h for p, h in previous["artifacts"].items()):
          raise ValueError(f"completed {name} artifacts changed; preserve this experiment")
        print(f"[{i}/{len(jobs)}] SKIP {name}", flush=True)
        continue
      if name == "collect" and (fresh / "manifest.json").exists():
        if load_run(fresh).get("complete"):
          command = None
        else:
          command += ["--resume"]
      if name == "encode" and (fresh / "features").exists():
        if (fresh / "features/meta.json").exists():
          command = None
        else:
          command += ["--resume"]
      state["stages"][name] = {"complete": False, "command": command}
      write_json(path, state)
      print(f"[{i}/{len(jobs)}] START {name}", flush=True)
      started = time.monotonic()
      if command:
        run_command(command, args.out / "logs" / f"{name}.txt")
      state["stages"][name].update(complete=True, seconds=time.monotonic()-started,
                                   artifacts={str(p): digest(p) for p in artifacts})
      write_json(path, state)
    if inputs(args) != signature:
      raise ValueError("source models/data/code changed during execution; preserve this experiment")
    state.update(complete=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state.pop("error", None)
    export(args, state)
    # Publish completion only after every report and the export exist.
    write_json(path, state)
  except BaseException as error:
    state["complete"] = False
    state["error"] = str(error) or type(error).__name__
    write_json(path, state)
    print(f"STOPPED: {state['error']}\nFull logs: {args.out}/logs\nRerun the identical command with --resume.", flush=True)
    raise


if __name__ == "__main__":
  main()
