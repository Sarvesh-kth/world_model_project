"""Two fresh policies using Calle's trainer/data, then the same camera benchmark.

python -u -m controller.sac_scratch --steps 40000 --episodes 30
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import sac_obstacles as common


def fit(args, state, label, obstacles):
    runs = state.setdefault("training_runs", {})
    training = Path(runs[label]) if label in runs else None
    if training is None or not (training / "imagined_training.json").is_file():
        if training is not None and training.exists() and not args.restart_training:
            raise RuntimeError(f"{label} training was interrupted; rerun with --restart-training to start a new preserved attempt")
        if shutil.disk_usage(args.out).free < args.min_free_gb * 1024 ** 3:
            raise OSError(f"need at least {args.min_free_gb} GiB free before training")
        attempts = args.out / "training" / label
        attempts.mkdir(parents=True, exist_ok=True)
        training = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):03d}"
        runs[label] = str(training)
        state["complete"] = False
        state.pop("error", None)
        command = common.training_command(args, state["settings"], training, obstacles=obstacles)
        state.setdefault("training_commands", {})[label] = command
        common.write(args.out / "pipeline.json", state)
        log = args.out / "logs" / f"{label}_{training.name}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"TRAIN {label}: {command}", flush=True)
        with log.open("a") as output, subprocess.Popen(command, cwd=common.SIMULATION, stdout=subprocess.PIPE,
                                                     stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                output.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
    record = common.read(training / "imagined_training.json")
    if record.get("initial_checkpoint_sha256") is not None or record["settings"].get("initialize_run") is not None:
        raise ValueError(f"{label} did not train from scratch")
    if bool(record["settings"].get("obstacles")) != obstacles or not record.get("terminal_observation_copy_fix"):
        raise ValueError(f"{label} has the wrong training filter or trainer version")
    for key in (*common.RECIPE, "steps"):
        if record["settings"][key] != state["settings"][key]:
            raise ValueError(f"{label} training setting differs: {key}")
    actor_hash = common.file_hash(common.require_file(training / "models/sac_best.zip"))
    hashes = state.setdefault("trained_actor_hashes", {})
    if hashes.get(label, actor_hash) != actor_hash:
        raise ValueError(f"{label} checkpoint changed after training")
    hashes[label] = actor_hash
    state.setdefault("selected_checkpoints", {})[label] = record["best"]
    common.write(args.out / "pipeline.json", state)
    print(f"SELECTED {label}: {record['best']}", flush=True)
    return training


def execute(args):
    import torch
    import stable_baselines3
    from . import imagined_sac
    from .control_pipeline import run_settings, scene_layout
    if stable_baselines3.__version__ != "2.7.1" or not torch.cuda.is_available():
        raise RuntimeError("need stable-baselines3==2.7.1 and CUDA; use scripts/setup_sac_obstacles.sh")
    # Read the trainer's actual defaults; only the documented 40k budget/seed/thread options are explicit here.
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(imagined_sac.arguments().parse_args([])).items()}
    settings.update(steps=args.steps, seed=args.seed, threads=args.threads)
    inputs = common.input_record(args)
    audit = common.audit_data(args)
    cfg, rest_z = run_settings(args.models_run, "")
    options = {"mode": "two_fresh_Calle_recipe_policies", "episodes": args.episodes, "steps": args.steps,
               "seed": args.seed, "threads": args.threads, "keep_frames": args.keep_frames,
               "position_jitter": .01, "obstacle_count": [1, 2], "empty_seed": 20495010,
               "obstacle_seed": 20496010, "python": sys.version,
               "rendering": {name: os.environ.get(name) for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM")},
               "versions": {name: importlib.metadata.version(name) for name in
                            ("torch", "torchvision", "stable-baselines3", "gymnasium", "mujoco", "transformers", "numpy")}}
    signature = common.digest({"inputs": inputs, "settings": settings, "options": options, "config": cfg, "rest_z": rest_z})
    path = args.out / "pipeline.json"
    state = common.read(path) if path.exists() else {"signature": signature, "complete": False, "settings": settings, "options": options}
    if state["signature"] != signature:
        raise ValueError("code, inputs, recipe or runtime changed; use a new --out instead of mixing runs")
    common.write(args.out / "inputs.json", {"hashes": inputs, "data_audit": audit, "recipe": settings,
                                          "options": options, "initialization": "fresh MlpPolicy [128,128], then original BC",
                                          "code_base": "calle-v3c:1ad414633aeceac32f057c0d653281c178684291",
                                          "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=common.REPO, text=True).strip()})
    common.write(path, state)
    scenes = [{"condition": condition, "seed": base + i,
               "layout": scene_layout(None, cfg, base + i, obstacles, random_ends=True)}
              for condition, base, obstacles in (("empty", 20495010, None), ("obstacles", 20496010, [1, 2]))
              for i in range(args.episodes)]
    recorded = {common.digest(row["layout"]) for row in common.read(args.run / "scene_settings.json")}
    if any(common.digest(scene["layout"]) in recorded for scene in scenes):
        raise ValueError("an evaluation layout duplicates a recorded scene")
    scene_path = args.out / "scenes.json"
    if scene_path.exists() and common.read(scene_path) != scenes:
        raise ValueError("scene manifest changed; choose a new --out")
    common.write(scene_path, scenes)
    print(f"PREFLIGHT OK: {torch.cuda.get_device_name()}; fresh policies, {args.steps} transitions each; "
          f"{audit['states_by_split']}", flush=True)
    if args.preflight_only:
        return
    try:
        for label, obstacles in (("train_empty", False), ("train_mixed", True)):
            training = fit(args, state, label, obstacles)
            common.evaluate(args, state, training, label, cfg, rest_z)
        state.update(complete=True, finished_unix=time.time())
        state.pop("error", None)
        common.write(path, state)
        common.export(args, state)
        print(f"COMPLETE: reports={common.REPO / 'results' / args.out.name}; GIFs={args.out / 'evaluation'}", flush=True)
    except Exception as error:
        state.update(complete=False, error=f"{type(error).__name__}: {error}")
        common.write(path, state)
        common.export(args, state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("data/episodes_v3"))
    p.add_argument("--models-run", type=Path, default=Path("models/v3c"))
    p.add_argument("--out", type=Path, default=Path("data/calle_sac_scratch_v1"))
    p.add_argument("--steps", type=int, default=40000, help="imagined transitions per fresh policy, matching the reported 40k recipe")
    p.add_argument("--episodes", type=int, default=30, help="episodes per condition per policy; 120 total by default")
    p.add_argument("--seed", type=int, default=20474010, help="same initialization/training seed for both policies")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--min-free-gb", type=float, default=5)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--restart-training", action="store_true", help="preserve interrupted fit and start a new attempt from scratch")
    args = p.parse_args()
    if min(args.steps, args.episodes, args.threads) < 1 or args.min_free_gb < 0:
        p.error("steps, episodes and threads must be positive; free space cannot be negative")
    if Path.cwd().resolve() != common.SIMULATION:
        p.error(f"run from {common.SIMULATION}")
    for name in ("run", "models_run", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if common.SIMULATION / "data" not in args.out.parents or any(args.out.is_relative_to(source) or source.is_relative_to(args.out)
                                                               for source in (args.run, args.models_run)):
        p.error("choose a separate output folder below simulation/data/")
    args.imagined_run = None
    args.policy_labels = ("train_empty", "train_mixed")
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.out / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("another process owns this run")
        execute(args)


if __name__ == "__main__":
    main()
