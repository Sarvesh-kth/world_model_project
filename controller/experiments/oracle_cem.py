"""Task 4: CEM with the simulator as a perfect world model, on the fixed Grade E scene.

Three stages (reach, lift, place), each with its own state-based cost, 20 seeds each (the seed
changes the cube's starting yaw and the planner's sampling), plus random and scripted baselines.
This is the upper bound for any learned world model.

    .venv/bin/python -m controller.experiments.oracle_cem [--seeds 20] [--workers 6]

Writes data/runs/oracle/<label>.csv, summary.json and success.png.
"""

import argparse
import functools
import json

from controller.adapters.m1_adapter import grade_e_adapter
from controller.agents import OracleCEMAgent, RandomAgent, ScriptedAgent
from controller.config import CEMConfig, Paths
from controller.costs import STAGE_COSTS
from controller.eval import plot_runs, run_many, summarize, write_csv

# Small N because every imagined step is a real MuJoCo step (~2.5 ms each)
ORACLE_CFG = CEMConfig(horizon=8, n_samples=64, n_elites=8, n_iters=4, init_std=0.3, min_std=0.05, execute_steps=2)
MAX_STEPS = {"reach": 40, "lift": 80, "place": 200}


def oracle_agent(task, seed):
    return OracleCEMAgent(STAGE_COSTS[task], ORACLE_CFG, seed=seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tasks", nargs="+", default=list(MAX_STEPS))
    args = p.parse_args()
    out = Paths().runs_dir / "oracle"
    seeds = list(range(args.seeds))
    runs, summary = {}, {"cem_config": ORACLE_CFG.__dict__, "max_steps": MAX_STEPS}
    for task in args.tasks:
        for name, make_agent in [
            ("random", RandomAgent),
            ("scripted", ScriptedAgent),
            ("oracle_cem", functools.partial(oracle_agent, task, 0)),
        ]:
            label = f"{task}/{name}"
            rows = run_many(grade_e_adapter, make_agent, seeds, task, MAX_STEPS[task], workers=args.workers, label=label)
            write_csv(rows, out / f"{task}_{name}.csv")
            runs[label] = rows
            summary[label] = summarize(rows)
            print(label, summary[label], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_runs(runs, out / "success.png", title="Grade E scene, 20 seeds: CEM with the simulator as model")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
