"""Four live rank-selection interventions from one successful recorded MPC scene.

Run from simulation/: python -u -m controller.mpc_counterfactual
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


SIMULATION = Path(__file__).resolve().parents[1]
CASES = (("rank_05", 5, None, 5), ("rank_25", 25, None, 25),
         ("rank_64", 64, None, 64), ("best_60_then_64", 1, 60, 64))


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(args):
    # Import the original offscreen selection before replay/MuJoCo; reuse the existing controller unchanged.
    from . import control_pipeline as cp
    from . import replay
    from world_model.common import write_json
    import numpy as np
    import torch

    source = args.source_episode
    result = read(source / "result.json")
    if result["method"] != "mpc" or not result["task_success"]:
        raise ValueError("Choose an actual successful MPC episode as the reference")
    source_options = read(source.parents[2] / "settings.json")
    saved_config = read(source / "mpc_settings.json")
    if saved_config["POPULATION"] != 64 or saved_config["RETREAT_AFTER_PLACE"]:
        raise ValueError("This experiment requires the source's 64-candidate search and no post-placement retreat rule")
    with np.load(source / "trajectory.npz", allow_pickle=False) as trajectory:
        reference = {key: trajectory[key].copy() for key in ("qpos", "p", "object_xyz", "held", "actions")}
    if len(reference["actions"]) < 60:
        raise ValueError("The reference must contain at least 60 executed actions")
    models_run = Path(source_options["models_run"]).resolve()
    hashes = {str(path): file_hash(path) for path in (
        source / "result.json", source / "trajectory.npz", source / "mpc_settings.json",
        source.parents[2] / "settings.json")}
    for folder in (models_run, SIMULATION / "controller", SIMULATION / "environment", SIMULATION / "rl",
                   SIMULATION / "world_model", SIMULATION / "configs", SIMULATION / "data_collection"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in (".pt", ".npz", ".json", ".py", ".xml", ".obj", ".stl", ".yml", ".yaml"):
                hashes[str(path.resolve())] = file_hash(path)
    hashes[str(SIMULATION / "offscreen.py")] = file_hash(SIMULATION / "offscreen.py")
    identity = {"inputs": hashes, "cases": CASES, "size": args.size, "speed": args.speed,
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "mujoco", "numpy", "transformers", "stable-baselines3")},
                "backend": {name: os.environ.get(name) for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    state_path = args.out / "counterfactual.json"
    if state_path.exists() and read(state_path)["signature"] != signature:
        raise ValueError("Inputs/code/render settings changed; choose a new --out to preserve this run")
    state = {"signature": signature, "complete": False, "source_episode": str(source),
             "ranking": "1-based descending score within the CEM round containing the normal winner; ties keep candidate order",
             "invalid_rank": "Keep the original invalid-plan hold-still fallback; never substitute a different rank",
             "switch": "First 60 actions choose rank 1; action 61 onward chooses rank 64",
             "limits": "One scene; fresh live perception. Source prefix equality is measured, not assumed.", **identity}
    write_json(state_path, state)
    frozen_config = args.out / "source_mpc_config.py"
    frozen_config.write_text("\n".join(f"{key} = {value!r}" for key, value in saved_config.items()
                                      if key.isupper() and key != "CONFIG_FILE") + "\n")
    torch.set_num_threads(source_options["threads"])
    cfg, rest_z = cp.run_settings(models_run, source_options["tag"])
    opts = argparse.Namespace(**source_options)
    opts.headless, opts.save_all_plans = True, True
    opts.models_run, opts.mpc_config = models_run, frozen_config
    opts.mpc_rank, opts.mpc_rank_after, opts.mpc_rank_then = 1, None, 1
    models = cp.WorldModels(models_run, source_options["tag"], opts)
    rows = []
    for label, rank, switch, then in CASES:
        root = args.out / label
        folder = root / "episodes/mpc" / f"seed_{result['seed']}"
        marker = root / "complete.json"
        if marker.exists():
            completed = read(marker)
            if completed["signature"] != signature:
                raise ValueError(f"Stale variant identity: {marker}")
            row = completed["metrics"]
            for name in ("result.json", "trajectory.npz", "plans.npz"):
                if file_hash(folder / name) != completed["episode_hashes"][name]:
                    raise ValueError(f"Completed recording changed: {folder / name}")
        else:
            root.mkdir(parents=True, exist_ok=True)
            opts.mpc_rank, opts.mpc_rank_after, opts.mpc_rank_then = rank, switch, then
            settings = {**source_options, "models_run": str(models_run), "mpc_config": str(frozen_config),
                        "headless": True, "save_all_plans": True, "mpc_rank": rank,
                        "mpc_rank_after": switch, "mpc_rank_then": then, "out": str(root)}
            write_json(root / "settings.json", settings)
            print(f"RUN {label}: rank={rank}; after={switch}; then={then}", flush=True)
            actual = cp.episode(root, "mpc", result["seed"], copy.deepcopy(cfg), copy.deepcopy(result["layout"]),
                                rest_z, opts, models=models)
            with np.load(folder / "trajectory.npz", allow_pickle=False) as trajectory:
                for key in ("qpos", "p", "object_xyz", "held"):
                    if not np.array_equal(trajectory[key][0], reference[key][0]):
                        raise ValueError(f"Different initial {key} from the source episode: {label}")
                prefix_differences = {}
                if switch is not None:
                    n = min(switch, len(trajectory["actions"]))
                    for key, original in reference.items():
                        count = n if key == "actions" else n + 1
                        prefix_differences[key] = float(np.abs(trajectory[key][:count].astype(float) - original[:count].astype(float)).max(initial=0.))
                    prefix_matches = n == switch and all(v <= 1e-6 for v in prefix_differences.values())
            with (folder / "steps.csv").open() as stream:
                decisions = list(csv.DictReader(stream))
            row = {"variant": label, "task_success": actual["task_success"], "steps": actual["steps"],
                   "final_goal_distance_cm": actual["final_goal_distance_cm"],
                   "obstacle_contact_steps": actual["obstacle_contact_steps"],
                   "fallback_steps": sum(r["planner_fallback"].lower() in ("true", "1") for r in decisions),
                   "source_first_60_matches_within_1e-6": prefix_matches if switch is not None else None,
                   "source_prefix_max_differences": prefix_differences,
                   "initial_state_matches": True}
            write_json(marker, {"signature": signature, "metrics": row,
                                "episode_hashes": {name: file_hash(folder / name) for name in
                                                   ("result.json", "trajectory.npz", "plans.npz")}})
        print(f"RENDER {label}: placed={row['task_success']}; B={row['final_goal_distance_cm']:.2f}cm; "
              f"fallbacks={row['fallback_steps']}", flush=True)
        ep = replay.load_episode(folder, models_run, source_options["tag"])
        ep["show_all_paths"], ep["label"] = True, label
        replay.video(ep, args.size, args.out / "gifs" / label, "overview", 1, args.speed)
        rows.append(row)
        write_json(args.out / "summary.json", {"source": result, "variants": rows, "complete": len(rows) == len(CASES)})
    state["complete"] = True
    write_json(state_path, state)
    print(f"COMPLETE: four GIFs under {args.out / 'gifs'}", flush=True)
    for row in rows:
        print(f"{row['variant']}: placed={row['task_success']}; B={row['final_goal_distance_cm']:.2f}cm; "
              f"contacts={row['obstacle_contact_steps']}; fallback steps={row['fallback_steps']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode", type=Path,
                        default=Path("data/mpc_allpaths_seed20496039/episodes/mpc/seed_20496039"))
    parser.add_argument("--out", type=Path, default=Path("data/mpc_counterfactual_seed20496039"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--speed", type=float, default=.5)
    args = parser.parse_args()
    if Path.cwd().resolve() != SIMULATION:
        parser.error(f"Run from {SIMULATION}")
    if not 0 < args.speed <= 10 or args.size < 64:
        parser.error("speed must be in (0,10] and size at least 64")
    args.source_episode, args.out = args.source_episode.resolve(), args.out.resolve()
    if (SIMULATION / "data" not in args.out.parents or args.out.is_relative_to(args.source_episode)
            or args.source_episode.is_relative_to(args.out)):
        parser.error("Choose a separate output under simulation/data; preserve the reference episode")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process owns this experiment")
        execute(args)


if __name__ == "__main__":
    main()
