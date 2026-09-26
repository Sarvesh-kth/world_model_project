"""Hyperparameters, paths and device choice. Every configurable value lives here."""

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CEMConfig:
    horizon: int = 10  # H, steps imagined per plan (1 s at 10 Hz)
    n_samples: int = 300  # N, action sequences sampled per iteration
    n_elites: int = 30  # K, about 10% of N
    n_iters: int = 4  # refits of mean and std per plan
    init_std: float = 0.5  # starting std, in action units (actions live in [-1, 1])
    min_std: float = 0.05  # std floor, stops the Gaussian collapsing too early
    execute_steps: int = 3  # actions executed per plan before replanning; 1 = MPC (Grade C)
    warm_start: bool = True  # start each search from the unexecuted rest of the previous plan


@dataclass
class Paths:
    """Output folders, relative to the working directory (the repo root) unless absolute.

    All under data/, which the repo's .gitignore already keeps out of git.
    """

    data_dir: Path = Path("data")
    runs_dir: Path = Path("data/runs")
    checkpoints_dir: Path = Path("data/checkpoints")


def m1_sim_dir() -> Path:
    """M1's simulation/ folder (holds their `environment` and `data_collection` packages).

    It sits next to controller/ in the repo, so it's found from this file's location, like M1's
    own config.py finds configs/. Set $M1_SIM_DIR to use a copy somewhere else.
    """
    path = Path(os.environ.get("M1_SIM_DIR") or Path(__file__).resolve().parents[1] / "simulation")
    if not (path / "environment" / "env.py").is_file():
        raise FileNotFoundError(f"M1's simulation folder not found at {path}. Set M1_SIM_DIR.")
    return path


def get_device(name: str | None = None) -> torch.device:
    """The given device, else the best available one: cuda, then mps (Apple GPU), then cpu."""
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
