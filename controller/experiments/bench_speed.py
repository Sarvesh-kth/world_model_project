"""Task 6: how long one CEM planning decision takes, by latent shape, N, H and iterations.

Mock latent models with random weights (speed doesn't depend on training):
  pooled    [1024]        z + MLP([z, a])                     (one vector per clip)
  tokens    [256, 1024]   z + per-token MLP([z_token, a])     (patch tokens of one frame)
  m2_split  [1044]        M2's SplitDynamics on [z | p]       (M2's actual choice: pooled + proprio)
Also reports the memory of the imagined trajectory CEM keeps: N x (H+1) x latent x 4 bytes.

    .venv/bin/python -m controller.experiments.bench_speed [--devices cpu mps]
"""

import argparse
import itertools
import json
import time

import torch
from torch import nn

from controller.cem import cem_plan
from controller.config import Paths, get_device
from controller.eval import write_csv


class MockLatent(nn.Module):
    def __init__(self, d=1024, a=5, width=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + a, width), nn.GELU(), nn.Linear(width, d))

    def forward(self, z, a):  # z [N, D] or [N, T, D]
        a = a.view(a.shape[0], *([1] * (z.dim() - 2)), a.shape[-1]).expand(*z.shape[:-1], a.shape[-1])
        return z + self.net(torch.cat([z, a], dim=-1))


def make(name, device):
    if name == "m2_split":
        from controller.adapters.m1_adapter import SIM_DIR  # noqa: F401  (simulation/ on the path)
        from world_model.train_dynamics import SplitDynamics

        m = SplitDynamics(1024, 20, 5).to(device).eval()
        return (lambda s, a: torch.cat(m(s[..., :1024], s[..., 1024:], a), dim=-1)), (1044,)
    m = MockLatent().to(device).eval()
    return m, ((1024,) if name == "pooled" else (256, 1024))


@torch.inference_mode()
def time_plan(fn, shape, device, n, h, iters, repeats=3):
    z0, goal = torch.zeros(shape, device=device), torch.ones(shape, device=device)
    cost = lambda tr, g: (tr[:, 1:] - g).square().flatten(2).mean(dim=(1, 2))  # noqa: E731
    low, high = -torch.ones(5, device=device), torch.ones(5, device=device)
    kwargs = dict(horizon=h, n_samples=n, n_elites=max(2, n // 10), n_iters=iters, action_low=low, action_high=high)
    cem_plan(z0, goal, fn, cost, **kwargs)  # warm-up
    sync = torch.mps.synchronize if device.type == "mps" else (torch.cuda.synchronize if device.type == "cuda" else (lambda: None))
    times = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        cem_plan(z0, goal, fn, cost, **kwargs)
        sync()
        times.append(time.perf_counter() - t)
    return min(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", nargs="+", default=["cpu", str(get_device())])
    args = p.parse_args()
    rows = []
    grid = {"pooled": [(n, h, i) for n, h, i in itertools.product((100, 300, 1000), (5, 10, 20), (3, 5))],
            "m2_split": [(n, h, i) for n, h, i in itertools.product((100, 300, 1000), (5, 10, 20), (3, 5))],
            "tokens": [(n, h, 3) for n, h in itertools.product((100, 300), (5, 10))]}
    for dev_name in dict.fromkeys(args.devices):
        device = torch.device(dev_name)
        for name, combos in grid.items():
            fn, shape = make(name, device)
            for n, h, iters in combos:
                mem_gb = n * (h + 1) * torch.tensor(shape).prod().item() * 4 / 1e9
                if mem_gb > 4:  # would not fit next to everything else in 16 GB
                    rows.append({"device": dev_name, "model": name, "N": n, "H": h, "iters": iters,
                                 "seconds": None, "trajectory_GB": round(mem_gb, 2), "note": "skipped: > 4 GB"})
                    continue
                try:
                    sec, note = round(time_plan(fn, shape, device, n, h, iters), 4), ""
                except RuntimeError as e:  # e.g. the Mac GPU failing under memory pressure
                    sec, note = None, f"failed: {type(e).__name__}"
                rows.append({"device": dev_name, "model": name, "N": n, "H": h, "iters": iters,
                             "seconds": sec, "trajectory_GB": round(mem_gb, 3), "note": note})
                print(rows[-1], flush=True)
    out = Paths().runs_dir / "bench_speed"
    write_csv(rows, out / "bench_speed.csv")
    (out / "bench_speed.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
