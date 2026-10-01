"""Train a rough JEPA dynamics model locally with M2's own pipeline, for M3's Level E tests.

M2's trained weights aren't in git, so M3 reproduces M2's Grade E recipe (README of the M2_Kuba
branch) on the Mac: M1's collector -> M2's prepare -> M2's action branches -> V-JEPA features
(controller/experiments/encode_features.py, because M2's encoder needs CUDA) -> M2's
train_dynamics with the split architecture. M2's scripts run unchanged, from simulation/.
The result is M3's local stand-in, not M2's official model.

    .venv/bin/python -m controller.experiments.jepa_pipeline [--skip-collect]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from controller.adapters.m1_adapter import SIM_DIR
from controller.config import Paths

ROOT = (Paths().data_dir / "jepa" / "grade_e").resolve()
CHECKPOINT = ROOT / "dynamics_split" / "best.pt"


def run(args, cwd=SIM_DIR):
    print("$", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--max-per-episode", type=int, default=10)
    p.add_argument("--branch-sources", type=int, default=120)
    p.add_argument("--skip-collect", action="store_true", help="episodes are already in data/jepa/grade_e/episodes")
    args = p.parse_args()
    py, episodes = sys.executable, ROOT / "episodes"
    if not args.skip_collect:
        run([py, "-m", "data_collection.collect", "--config", "configs/grade_e.yml",
             "--layout", "configs/grade_e_layout.json", "--episodes", args.episodes,
             "--workers", 3, "--seed", 34, "--out", episodes])
    run([py, "-m", "world_model.prepare", "--episodes", episodes, "--out", ROOT / "manifest.json",
         "--max-per-episode", args.max_per_episode])
    run([py, "-m", "world_model.collect_branches", "--manifest", ROOT / "manifest.json",
         "--out", ROOT / "manifest_branches.json", "--max-sources", args.branch_sources])
    run([py, "-m", "controller.experiments.encode_features", "--manifest", ROOT / "manifest_branches.json",
         "--out", ROOT / "features_branches"], cwd=Path.cwd())
    run([py, "-m", "world_model.train_dynamics", "--manifest", ROOT / "manifest_branches.json",
         "--features", ROOT / "features_branches", "--architecture", "split", "--epochs", 30,
         "--out", ROOT / "dynamics_split"])
    run([py, "-m", "world_model.probe", "--manifest", ROOT / "manifest_branches.json",
         "--features", ROOT / "features_branches"])
    print(f"checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
