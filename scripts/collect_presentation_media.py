"""Copy selected historical/probe GIFs for the presentation; never run controllers.

From repo root: .venv/bin/python scripts/collect_presentation_media.py --stage
Add --source-root /path/to/Sarveshs/archive when the original runs are elsewhere.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "presentation/history_media"
REQUESTS = (
    ("v1_success", "q_clutter_v1/control/scene_0019_empty_normal/mixed/episodes/rl_q/seed_20464024/actual.gif", "Pooled V1; JEPA perception + simulator-trained SAC; placed 3.10 cm from B"),
    ("v1_failure", "q_clutter_v1/control/scene_0016_clutter_drop_late/mixed/episodes/rl_q/seed_20464021/actual.gif", "Pooled V1; forced late drop; failed placement, 7.84 cm"),
    ("v2_success", "control_final_full20/episodes/jepa_mpc/seed_20494012/actual.gif", "Spatial Sarvesh V2; SAC-guided MPC; placed 0.45 cm"),
    ("v2_failure", "control_final_full20/episodes/jepa_mpc/seed_20494022/actual.gif", "Spatial Sarvesh V2; SAC-guided MPC; failed placement, 7.49 cm"),
    ("v3_success", "control_final_20/episodes/jepa_mpc/seed_20494022/actual.gif", "Historical Sarvesh V3 combined-data bundle; placed 0.70 cm"),
    ("v3_failure", "control_final_20/episodes/jepa_mpc/seed_20494011/actual.gif", "Historical Sarvesh V3 combined-data bundle; failed placement, 53.34 cm"),
    ("rank_05", "mpc_counterfactual_seed20496039/gifs/rank_05.gif", "Calle MPC rank-5 intervention; placed 2.39 cm in 93 steps"),
    ("rank_25", "mpc_counterfactual_seed20496039/gifs/rank_25.gif", "Calle MPC rank-25 intervention; placed 1.36 cm in 153 steps"),
    ("rank_64", "mpc_counterfactual_seed20496039/gifs/rank_64.gif", "Calle MPC rank-64 intervention; placed 6.38 cm in 171 steps, one fallback"),
    ("best_60_then_64", "mpc_counterfactual_seed20496039/gifs/best_60_then_64.gif", "Calle MPC best first 60 actions then rank 64; failed, 13.18 cm at 300 steps"),
    ("mpc_allpaths", "mpc_allpaths_seed20496039/episodes/mpc/seed_20496039/replay.gif", "Full-proposal overlay of the normal best-plan source; placed 4.36 cm in 125 steps"),
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=Path, default=[], help="extra project/archive root containing simulation/data, data, or run folders")
    parser.add_argument("--stage", action="store_true", help="stage only the selected GIFs, manifest and LFS attributes; never commit/push")
    args = parser.parse_args()
    if args.stage and subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT).strip():
        raise SystemExit("Commit/review existing staged changes first; this command must not mix them with the media selection.")
    roots = [ROOT, ROOT.parent / "world_model_project", *args.source_root]
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records, copied = [], []
    for name, relative, description in REQUESTS:
        candidates = [base / prefix / relative for base in roots for prefix in ("simulation/data", "data", "")]
        if name in ("rank_05", "rank_25", "rank_64", "best_60_then_64"):
            candidates += [base / "gifs" / f"{name}.gif" for base in roots]
        if name == "mpc_allpaths":
            candidates = [base / "presentation/mpc_allpaths_seed20496039.gif" for base in roots] + candidates
        source = next((path for path in candidates if path.is_file()), None)
        record = {"id": name, "description": description, "requested_run_path": relative}
        if source is None:
            print(f"MISSING {name}: {relative}")
            records.append({**record, "status": "missing"})
            continue
        with source.open("rb") as stream:
            if stream.read(6) not in (b"GIF87a", b"GIF89a"):
                raise SystemExit(f"Not actual GIF bytes (possibly an LFS pointer): {source}")
        destination = DESTINATION / f"{name}.gif"
        sha = digest(source)
        if destination.exists() and digest(destination) != sha:
            raise SystemExit(f"Refusing to overwrite a different clip: {destination}")
        if not destination.exists():
            shutil.copyfile(source, destination)
        copied.append(destination)
        records.append({**record, "status": "copied", "source": str(source.resolve()),
                        "destination": str(destination.relative_to(ROOT)), "sha256": sha,
                        "bytes": destination.stat().st_size})
        print(f"COPIED {name}: {destination.stat().st_size / 1024**2:.1f} MiB")
    manifest = DESTINATION / "selection.json"
    manifest.write_text(json.dumps({"purpose": "Presentation clips only; no new evaluation or training",
                                   "clips": records}, indent=2) + "\n")
    print(f"Collected {len(copied)}/{len(REQUESTS)} clips; missing entries retain presentation placeholders.")
    if args.stage:
        subprocess.run(["git", "lfs", "track", "presentation/history_media/*.gif"], cwd=ROOT, check=True)
        paths = [str(path.relative_to(ROOT)) for path in (*copied, manifest)]
        subprocess.run(["git", "add", "-f", "--", *paths], cwd=ROOT, check=True)
        subprocess.run(["git", "add", "--", ".gitattributes"], cwd=ROOT, check=True)
        print("Selected GIFs staged. Review, commit and push; no MP4s, datasets, weights or raw frames were staged.")


if __name__ == "__main__":
    main()
