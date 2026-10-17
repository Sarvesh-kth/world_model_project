"""Matched camera evaluation -> obstacle-inclusive imagination training -> camera evaluation.

Run from simulation/: python -u -m controller.sac_obstacles
No simulator-state actor, MPC, world-model fitting or new policy inputs are introduced.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

SIMULATION = Path(__file__).resolve().parents[1]
REPO = SIMULATION.parent

# Inherit the actual source run's training recipe instead of silently taking new defaults.
RECIPE = ("envs", "horizon", "updates_per_step", "bc_epochs", "bc_weight", "td3bc_alpha",
          "critic_warmup", "prefill", "learning_rate", "ent_coef", "penalty_scale", "grasp_check",
          "eval_every", "validation_starts", "validation_horizon", "exact_checks", "seed", "threads")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. See scripts/setup_sac_obstacles.sh and controller/SAC_OBSTACLES.md")
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"{path} is an LFS pointer; run the selective LFS setup first")
    return path


def input_record(args):
    paths = [args.run / "manifest.json",
             args.run / "scene_settings.json", args.run / "features/meta.json",
             args.run / "features/latents.npy", args.models_run / "info.json",
             args.models_run / "features/meta.json", args.models_run / "features/pca.npz"]
    if args.imagined_run is not None:
        paths += [args.imagined_run / name for name in ("models/sac_best.zip", "pipeline.json", "imagined_training.json")]
    paths += [args.models_run / f"models/{role}.pt" for role in ("dynamics", "readout", "reward", "obstacle")]
    # Source + config + mesh hashes reject resuming across an environment/code change.
    paths += [p for directory in ("controller", "rl", "environment", "world_model", "data_collection", "configs")
              for p in (SIMULATION / directory).rglob("*") if p.is_file() and
              "__pycache__" not in p.parts and p.suffix in (".py", ".yml", ".yaml", ".xml", ".obj", ".stl", ".png")]
    paths.append(SIMULATION / "offscreen.py")
    return {str(p.resolve()): file_hash(require_file(p)) for p in sorted(set(paths))}


def audit_data(args):
    import numpy as np
    manifest, scenes = read(args.run / "manifest.json"), read(args.run / "scene_settings.json")
    if manifest.get("schema") != "vision_consequences_v1":
        raise ValueError("unexpected recorded-data schema")
    states = manifest["states"]
    latents = np.load(args.run / "features/latents.npy", mmap_mode="r")
    meta = read(args.models_run / "features/meta.json")
    data_meta = read(args.run / "features/meta.json")
    for key in ("model", "model_revision", "pooling", "cameras", "clip_frames", "spatial_pool", "pca_dims", "layout"):
        if data_meta.get(key) != meta.get(key):
            raise ValueError(f"recorded features and frozen model differ: {key}")
    if latents.shape != (len(states), sum(meta["layout"].values())) or not np.issubdtype(latents.dtype, np.floating):
        raise ValueError(f"latent shape/type does not match manifest/models: {latents.shape}, {latents.dtype}")
    for begin in range(0, len(states), 2048):
        if not np.isfinite(latents[begin:begin + 2048]).all():
            raise ValueError("recorded latents contain nonfinite values")
    if data_meta.get("keys") != [s["key"] for s in states]:
        raise ValueError("latent row keys do not match the manifest; do not mix dataset versions")
    scene_map = {s["rollout"]: s for s in scenes}
    split_by_scene, counts = {}, {}
    for state in states:
        scene, split = state["scene"], state["split"]
        if scene in split_by_scene and split_by_scene[scene] != split:
            raise ValueError("one scene appears in multiple splits")
        split_by_scene[scene] = split
        condition = "obstacles" if scene_map[scene]["layout"].get("obstacles") else "empty"
        name = f"{split}/{condition}"
        counts[name] = counts.get(name, 0) + 1
    for split in ("train", "val"):
        for condition in ("empty", "obstacles"):
            if not counts.get(f"{split}/{condition}"):
                raise ValueError(f"no {split}/{condition} starts; cannot run the mixed-scene experiment")
    return {"recorded_states": len(states), "latent_shape": list(latents.shape), "states_by_split": counts}


def training_command(args, settings, output, *, obstacles=True):
    command = [sys.executable, "-u", "-m", "controller.imagined_sac", "--run", str(args.run),
               "--models-run", str(args.models_run),
               "--out", str(output), "--steps", str(args.steps or settings["steps"]), "--device", "cuda"]
    if args.imagined_run is not None:
        command.extend(["--initialize-run", str(args.imagined_run)])
    if obstacles:
        command.append("--obstacles")
    for key in RECIPE:
        if key == "grasp_check":
            if not settings[key]:
                command.append("--no-grasp-check")
        else:
            command.extend(["--" + key.replace("_", "-"), str(settings[key])])
    return command


def summaries(rows, labels=("before", "after")):
    import statistics
    table = []
    for policy in labels:
        for condition in ("empty", "obstacles"):
            group = [r for r in rows if r["policy"] == policy and r["condition"] == condition]
            if not group:
                continue
            failures = [r["final_goal_distance_cm"] for r in group if not r["task_success"]]
            table.append({"policy": policy, "condition": condition, "episodes": len(group),
                          "successes": sum(bool(r["task_success"]) for r in group),
                          "failures": len(failures),
                          "mean_final_goal_distance_cm": statistics.mean(r["final_goal_distance_cm"] for r in group),
                          "failure_mean_goal_distance_cm": statistics.mean(failures) if failures else None,
                          "obstacle_contact_steps": sum(r["obstacle_contact_steps"] for r in group)})
    return table


def export(args, state):
    from world_model.common import save_csv
    target = REPO / "results" / args.out.name
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for marker in sorted((args.out / "evaluation").glob("*/*/episodes/sac_imagined/seed_*/complete.json")):
        completion = read(marker)
        if completion["run_signature"] != state["signature"]:
            raise ValueError(f"stale completion marker: {marker}")
        result = read(marker.with_name("result.json"))
        result.update({"policy": marker.relative_to(args.out / "evaluation").parts[0],
                       "condition": marker.relative_to(args.out / "evaluation").parts[1],
                       "folder": str(marker.parent.relative_to(args.out)),
                       "actor_sha256": completion["actor_sha256"], "scene_sha256": completion["scene_sha256"],
                       "initial_state_sha256": completion["initial_state_sha256"]})
        rows.append(result)
        for source in (marker.with_name("result.json"), marker.with_name("steps.csv"), marker):
            destination = target / source.relative_to(args.out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    labels = getattr(args, "policy_labels", ("before", "after"))
    table = summaries(rows, labels)
    compact = [{k: r[k] for k in ("policy", "condition", "seed", "task_success", "final_goal_distance_cm",
                                  "obstacle_contact_steps", "folder", "actor_sha256", "scene_sha256", "initial_state_sha256")} for r in rows]
    if compact:
        save_csv(target / "episodes.csv", compact)
        save_csv(target / "summary.csv", table)
    closest = {f"{policy}/{condition}": sorted([r for r in compact if r["policy"] == policy and r["condition"] == condition],
                                              key=lambda r: r["final_goal_distance_cm"])[:10]
               for policy in labels for condition in ("empty", "obstacles")}
    write(target / "closest_10.json", closest)
    paired = []
    for before in [r for r in compact if r["policy"] == labels[0]]:
        after = next((r for r in compact if r["policy"] == labels[1] and r["condition"] == before["condition"] and r["seed"] == before["seed"]), None)
        if after:
            if any(after[key] != before[key] for key in ("scene_sha256", "initial_state_sha256")):
                raise ValueError("paired policies have different scenes or initial simulator states")
            paired.append({"condition": before["condition"], "seed": before["seed"],
                           f"{labels[0]}_success": before["task_success"], f"{labels[1]}_success": after["task_success"],
                           "distance_improvement_cm": before["final_goal_distance_cm"] - after["final_goal_distance_cm"]})
    if paired:
        save_csv(target / "paired.csv", paired)
    write(target / "summary.json", {"complete": state["complete"], "summary": table, "paired": paired})
    for name in ("pipeline.json", "scenes.json", "inputs.json"):
        shutil.copyfile(args.out / name, target / name)
    if state.get("training"):
        record = Path(state["training"]) / "imagined_training.json"
        if record.is_file():
            shutil.copyfile(record, target / "imagined_training.json")
    for label, run in state.get("training_runs", {}).items():
        record = Path(run) / "imagined_training.json"
        if record.is_file():
            destination = target / "training" / label / "imagined_training.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record, destination)
    for log in (args.out / "logs").glob("*.txt"):
        destination = target / "logs" / log.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log, destination)
    lines = [f"Calle imagined SAC; complete={state['complete']}; {len(rows)}/{4 * args.episodes} camera episodes"]
    for row in table:
        lines.append(f"{row['policy']}/{row['condition']}: placements {row['successes']}/{row['episodes']}; "
                     f"mean B={row['mean_final_goal_distance_cm']:.2f}cm; contact steps={row['obstacle_contact_steps']}")
    lines.append("The actor still has 41 inputs and no explicit obstacle map. No success is implied by completion.")
    (target / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def evaluate(args, state, policy_run, label, cfg, rest_z):
    import torch
    from controller import control_pipeline as cp
    policy = cp.Policy(policy_run)
    if policy.heightmap or policy.sac.observation_space.shape != (41,) or policy.sac.action_space.shape != (5,):
        raise ValueError("expected Calle's unchanged 41-input, 5-action imagined SAC")
    actor_hash = file_hash(policy_run / "models/sac_best.zip")
    models = None
    opts = argparse.ArgumentParser()
    cp.run_arguments(opts)
    options = opts.parse_args(["--headless", "--threads", str(args.threads)])
    options.position_jitter = .01
    options.penalty_scale = state["settings"]["penalty_scale"]
    torch.set_num_threads(args.threads)
    for scene in read(args.out / "scenes.json"):
        root = args.out / "evaluation" / label / scene["condition"]
        folder = root / "episodes/sac_imagined" / f"seed_{scene['seed']}"
        expected = {"run_signature": state["signature"], "actor_sha256": actor_hash, "scene_sha256": digest(scene)}
        if (folder / "complete.json").is_file():
            saved = read(folder / "complete.json")
            if any(saved.get(key) != value for key, value in expected.items()) or not (folder / "actual.gif").is_file():
                raise ValueError(f"completed episode identity changed: {folder}")
            continue
        if shutil.disk_usage(args.out).free < args.min_free_gb * 1024 ** 3:
            raise OSError(f"less than {args.min_free_gb} GiB free; safely stopped before the next episode")
        if models is None:
            models = cp.WorldModels(args.models_run, "", options)
        print(f"EVALUATE {label}/{scene['condition']} seed={scene['seed']}", flush=True)
        result = cp.episode(root, "sac_imagined", scene["seed"], copy.deepcopy(cfg), copy.deepcopy(scene["layout"]),
                            rest_z, options, policy, models)
        if digest(result["layout"]) != digest(scene["layout"]):
            raise ValueError("the episode changed its requested scene layout")
        import numpy as np
        with np.load(folder / "trajectory.npz") as trajectory:
            expected["initial_state_sha256"] = digest({key: trajectory[key][0].tolist()
                                                       for key in ("qpos", "p", "object_xyz", "held")})
        write(folder / "complete.json", expected)
        # GIF + trajectory.qpos reproduce the movements; retain JPEGs only when requested.
        if not args.keep_frames:
            shutil.rmtree(folder / "frames")
        export(args, state)
    del models, policy
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagined-run", type=Path, default=Path("data/sac_imagined_v1"))
    parser.add_argument("--run", type=Path, default=Path("data/episodes_v3"))
    parser.add_argument("--models-run", type=Path, default=Path("models/v3c"))
    parser.add_argument("--out", type=Path, default=Path("data/calle_sac_obstacles_v1"))
    parser.add_argument("--episodes", type=int, default=30, help="episodes per condition and policy: 120 total at 30")
    parser.add_argument("--steps", type=int, default=None, help="additional imagined transitions; default source run's steps")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=5)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--restart-training", action="store_true", help="start a new attempt if training was interrupted; preserves old attempt")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="fill the frozen checkpoint results first; rerun without this flag to train")
    args = parser.parse_args()
    if args.episodes < 1 or (args.steps is not None and args.steps < 1) or args.threads < 1 or args.min_free_gb < 0:
        parser.error("episodes, steps and threads must be positive; min-free-gb must be nonnegative")
    if Path.cwd().resolve() != SIMULATION:
        parser.error(f"run this command from {SIMULATION}")
    for name in ("imagined_run", "run", "models_run", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if SIMULATION / "data" not in args.out.parents or any(args.out.is_relative_to(source) or source.is_relative_to(args.out)
                                                        for source in (args.imagined_run, args.run, args.models_run)):
        parser.error("choose a new --out folder below this checkout's simulation/data/")
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.out / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another process already owns this run")
        execute(args)


def execute(args):
    for name in ("models/sac_best.zip", "pipeline.json", "imagined_training.json"):
        require_file(args.imagined_run / name)
    inputs = input_record(args)
    source = read(args.imagined_run / "imagined_training.json")
    settings = source["settings"]
    missing = set((*RECIPE, "steps")) - settings.keys()
    if missing or read(args.imagined_run / "pipeline.json").get("kind") != "imagined_sac":
        raise ValueError(f"need the original imagined SAC training record; missing settings={sorted(missing)}")
    if settings.get("obstacles"):
        raise ValueError("the before policy was already obstacle-trained; supply Calle's original empty-trained run")
    if settings.get("tag") or Path(settings["models_run"]).name != args.models_run.name:
        raise ValueError("source SAC used a different world-model bundle; obtain its matching assets first")
    import torch
    import stable_baselines3
    from controller.control_pipeline import Policy, run_settings, scene_layout
    if stable_baselines3.__version__ != "2.7.1" or not torch.cuda.is_available():
        raise RuntimeError("need stable-baselines3==2.7.1 and a working CUDA PyTorch; use the setup script")
    frozen = Policy(args.imagined_run)
    if frozen.heightmap or frozen.sac.observation_space.shape != (41,) or frozen.sac.action_space.shape != (5,):
        raise ValueError("the supplied checkpoint is not the 41-input imagined SAC")
    del frozen
    audit = audit_data(args)
    cfg, rest_z = run_settings(args.models_run, "")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "stable-baselines3", "gymnasium", "mujoco", "transformers", "numpy")}
    options = {"episodes": args.episodes, "steps": args.steps or settings["steps"], "threads": args.threads,
               "position_jitter": .01, "obstacle_count": [1, 2], "empty_seed": 20495010, "obstacle_seed": 20496010,
               "keep_frames": args.keep_frames, "versions": versions, "python": sys.version}
    options["rendering"] = {name: os.environ.get(name) for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}
    signature = digest({"inputs": inputs, "options": options, "config": cfg, "rest_z": rest_z})
    path = args.out / "pipeline.json"
    state = read(path) if path.exists() else {"signature": signature, "complete": False, "settings": settings, "options": options}
    if state["signature"] != signature:
        raise ValueError("inputs/code/settings/runtime changed; use a new --out instead of mixing results")
    write(args.out / "inputs.json", {"hashes": inputs, "data_audit": audit,
                                   "source_training": source, "options": options,
                                   "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                                   "code_base": "calle-v3c:1ad414633aeceac32f057c0d653281c178684291"})
    write(path, state)
    scenes = [{"condition": condition, "seed": base + i,
               "layout": scene_layout(None, cfg, base + i, obstacles, random_ends=True)}
              for condition, base, obstacles in (("empty", 20495010, None), ("obstacles", 20496010, [1, 2]))
              for i in range(args.episodes)]
    recorded_scenes = read(args.run / "scene_settings.json")
    used_layouts = {digest(s["layout"]) for s in recorded_scenes}
    if any(digest(s["layout"]) in used_layouts for s in scenes):
        raise ValueError("an evaluation layout duplicates a recorded training/validation/test scene")
    scene_path = args.out / "scenes.json"
    if scene_path.exists() and read(scene_path) != scenes:
        raise ValueError("saved scene manifest differs; choose a new --out")
    write(scene_path, scenes)
    print(f"PREFLIGHT OK: CUDA={torch.cuda.get_device_name()}; source actor SHA256="
          f"{inputs[str(args.imagined_run / 'models/sac_best.zip')]}; {audit['states_by_split']}", flush=True)
    if args.preflight_only:
        return
    try:
        evaluate(args, state, args.imagined_run, "before", cfg, rest_z)
        if args.evaluate_only:
            export(args, state)
            return
        training = Path(state["training"]) if state.get("training") else None
        if training is None or not (training / "imagined_training.json").is_file():
            if training is not None and training.exists() and not args.restart_training:
                raise RuntimeError("training was interrupted. Rerun with --restart-training for a fresh attempt from the frozen source; old files stay intact")
            attempts = args.out / "training"
            attempts.mkdir(exist_ok=True)
            training = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):03d}"
            state.update(training=str(training), complete=False)
            command = training_command(args, settings, training)
            state["training_command"] = command
            state.pop("error", None)
            write(path, state)
            log = args.out / "logs" / f"{training.name}.txt"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"TRAIN: {command}; log={log}", flush=True)
            with log.open("a") as output, subprocess.Popen(command, cwd=SIMULATION, stdout=subprocess.PIPE,
                                                         stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output.write(line)
                    output.flush()
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
        training_record = read(training / "imagined_training.json")
        if training_record.get("initial_checkpoint_sha256") != inputs[str(args.imagined_run / "models/sac_best.zip")]:
            raise ValueError("trained actor did not start from the frozen source checkpoint")
        if not training_record["settings"].get("obstacles") or not training_record.get("terminal_observation_copy_fix"):
            raise ValueError("training record is not the corrected obstacle-inclusive fit")
        trained_hash = file_hash(require_file(training / "models/sac_best.zip"))
        if state.get("trained_actor_sha256", trained_hash) != trained_hash:
            raise ValueError("the completed training checkpoint changed; do not mix actor versions")
        state["trained_actor_sha256"] = trained_hash
        state["selected_checkpoint"] = training_record["best"]
        print(f"SELECTED: {training_record['best']}; source actor remains preserved", flush=True)
        write(path, state)
        evaluate(args, state, training, "after", cfg, rest_z)
        state.update(complete=True, finished_unix=time.time())
        state.pop("error", None)
        write(path, state)
        export(args, state)
        print(f"COMPLETE: reports={REPO / 'results' / args.out.name}; GIFs/trajectories={args.out / 'evaluation'}", flush=True)
    except Exception as error:
        state.update(complete=False, error=f"{type(error).__name__}: {error}")
        write(path, state)
        export(args, state)
        raise


if __name__ == "__main__":
    main()
