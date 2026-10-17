"""Collect compact controller results/replays and selected SAC weights; exclude training data/media.

.venv/bin/python scripts/share_control_results.py --stage
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results/control_release_v1"
RUNS = ("calle_sac_scratch_v1", "calle_mpc_matched_empty30", "calle_mpc_matched_obstacles30",
        "mpc_allpaths_seed20496039", "mpc_counterfactual_seed20496039")
SMALL = {".json", ".csv", ".txt", ".svg", ".png", ".pdf"}
CODE = {".py", ".sh", ".md", ".yml", ".yaml", ".toml", ".txt", ".xml"}
PREFIXES = ("scripts/", "presentation/", "simulation/controller/", "simulation/world_model/",
            "simulation/rl/", "simulation/environment/", "simulation/configs/", "simulation/data_collection/")


def git(*args, **kwargs):
    return subprocess.check_output(["git", *args], cwd=ROOT, **kwargs)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true", help="stage only collected files and source changes; never commit/push")
    args = parser.parse_args()
    if args.stage and git("diff", "--cached", "--name-only").strip():
        raise RuntimeError("The Git index already contains staged files. Review/commit those first to avoid mixing them into this export.")
    import numpy as np
    staged, inventory = [], []

    def record(path, source=None, note="copied"):
        staged.append(str(path.relative_to(ROOT)))
        inventory.append({"file": str(path.relative_to(BUNDLE)), "bytes": path.stat().st_size,
                          "sha256": sha(path), "source": str(source) if source else None, "note": note})

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        record(destination, source)

    run_status = {}
    for name in RUNS:
        run = ROOT / "simulation/data" / name
        if not run.is_dir():
            print(f"Not present, skipped: {name}")
            continue
        count = 0
        for source in sorted(run.rglob("*")):
            relative = source.relative_to(run)
            if not source.is_file() or "frames" in relative.parts or "gifs" in relative.parts:
                continue
            target = BUNDLE / "runs" / name / relative
            # Metadata only from training; checkpoints are collected separately below.
            if source.suffix in {".json", ".csv", ".txt"} and "logs" not in relative.parts:
                copy(source, target)
            elif source.name in ("trajectory.npz", "plans.npz") and "training" not in relative.parts:
                if not (source.parent / "result.json").is_file():
                    continue  # A result is written last, after the episode arrays finish saving.
                target.parent.mkdir(parents=True, exist_ok=True)
                with np.load(source, allow_pickle=False) as arrays:
                    keys = ("qpos", "p", "object_xyz", "held", "actions") if source.name == "trajectory.npz" else arrays.files
                    np.savez_compressed(target, **{key: arrays[key] for key in keys if key in arrays})
                record(target, source, "Replay arrays only; cached JEPA z excluded" if source.name == "trajectory.npz" else "All saved imagined paths")
                count += source.name == "trajectory.npz"
        run_status[name] = {"recorded_episodes": count}
        for marker in ("pipeline.json", "counterfactual.json"):
            if (run / marker).is_file():
                run_status[name]["complete"] = json.loads((run / marker).read_text()).get("complete")
        reports = ROOT / "results" / name
        if reports.is_dir():
            for source in sorted(reports.rglob("*")):
                if (source.is_file() and source.suffix in SMALL and
                        not {"logs", "frames", "gifs"}.intersection(source.relative_to(reports).parts)):
                    copy(source, BUNDLE / "reports" / name / source.relative_to(reports))

    state = json.loads((ROOT / "simulation/data/calle_sac_scratch_v1/pipeline.json").read_text())
    if not state["complete"]:
        raise RuntimeError("The SAC campaign must be complete before publishing its selected weights")
    for label, path in state["training_runs"].items():
        source = Path(path)
        if not source.is_absolute():
            source = ROOT / "simulation" / source
        if sha(source / "models/sac_best.zip") != state["trained_actor_hashes"][label]:
            raise ValueError(f"Selected checkpoint identity changed: {label}")
        for name in ("models/sac_best.zip", "pipeline.json", "imagined_training.json"):
            copy(source / name, BUNDLE / "policies" / label / name)

    readme = BUNDLE / "README.md"
    readme.write_text("""# Controller results release

Selected SAC policies: policies/train_empty and policies/train_mixed each contain
models/sac_best.zip, pipeline.json and imagined_training.json. These folders work
with controller.control_pipeline --methods sac_imagined --imagined-run <folder>.
Frozen D/Q/R/O/PCA/encoder metadata remain at simulation/models/v3c (already in Git LFS).

Reports include actual episode outcomes, settings, CSVs and any generated plots.
Runs may be partial: consult inventory.json and their recorded completion flags.
Replay copies contain evaluation states and imagined paths, not training examples.
trajectory.npz omits z; original files remain unchanged. Original source hashes in
metadata describe originals; inventory hashes describe these exported copies.

From simulation/, regenerate a replay, with no controller evaluation or training:

    ../.venv/bin/python -m controller.replay ../results/control_release_v1/runs/calle_mpc_matched_obstacles30/episodes/mpc/seed_20496039 --video --models-run models/v3c --tag '' --speed 0.5

For runs/mpc_allpaths_seed20496039, add --all-paths to draw every recorded proposal.
Counterfactual variant episodes are under runs/mpc_counterfactual_seed20496039/<variant>/episodes/mpc/seed_20496039.

No raw frames, GIF/MP4/video archives, training datasets, replay buffers, virtualenvs
or cached JEPA latents are included. Source scripts/notebooks are in the repository.
""")
    record(readme, note="Export instructions")
    manifest = BUNDLE / "inventory.json"
    manifest.write_text(json.dumps({"source_commit": git("rev-parse", "HEAD", text=True).strip(),
                                    "runs": run_status, "files": inventory}, indent=2) + "\n")
    staged.append(str(manifest.relative_to(ROOT)))
    print(json.dumps(run_status, indent=2))
    print(f"Collected {len(staged)} files, {sum(r['bytes'] for r in inventory) / 1024**2:.1f} MiB under {BUNDLE}")
    if args.stage:
        for pattern in ("results/control_release_v1/**/*.zip", "results/control_release_v1/**/*.npz"):
            git("lfs", "track", pattern)
        # Force-add only the enumerated export files, never a whole data/results directory.
        for start in range(0, len(staged), 100):
            git("add", "-f", "--", *staged[start:start+100])
        git("add", "--", ".gitattributes")
        changed = git("ls-files", "-m", "-o", "--exclude-standard", "-z").decode().split("\0")
        for name in changed:
            path = ROOT / name
            source_path = name.startswith(PREFIXES) or ("/" not in name and path.suffix in CODE)
            if not name or not source_path or path.is_symlink():
                continue
            if not path.exists() and path.suffix in CODE | {".ipynb"}:
                git("add", "-u", "--", name)
                continue
            if not path.is_file():
                continue
            if path.suffix == ".ipynb":
                notebook = json.loads(path.read_text())
                notebook.get("metadata", {}).pop("widgets", None)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        cell["outputs"], cell["execution_count"] = [], None
                blob = git("hash-object", "-w", "--stdin", input=(json.dumps(notebook, indent=1) + "\n").encode()).decode().strip()
                git("update-index", "--add", "--cacheinfo", f"100644,{blob},{name}")
            elif path.suffix in CODE or name.startswith("simulation/configs/"):
                git("add", "--", name)
        print("Staged export and source changes. Notebook outputs remain in your working copy only.")


if __name__ == "__main__":
    main()
