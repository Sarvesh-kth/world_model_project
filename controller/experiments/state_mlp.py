"""Task 5: learn a state-based world model from M1's episodes and plan with it.

1. Train StateMLP on the Grade E episodes (20% of episodes held out).
2. Open-loop prediction error vs horizon (1..10 steps) on the held-out episodes.
3. CEM with the MLP on reach / lift / place, same costs and CEM settings as the oracle (task 4).
4. Audit: replay every chosen plan in the real sim and log predicted vs real cost, to catch the
   planner exploiting model errors.

    .venv/bin/python -m controller.experiments.state_mlp [--seeds 20] [--workers 6]

Needs data/jepa/grade_e/episodes (collected by controller.experiments.jepa_pipeline).
"""

import argparse
import csv
import functools
import json

import numpy as np

from controller.adapters.m1_adapter import grade_e_adapter
from controller.agents import StateMLPAgent
from controller.config import Paths
from controller.costs import STAGE_COSTS
from controller.dynamics import state_mlp
from controller.eval import plot_runs, run_episode, run_many, summarize, write_csv
from controller.experiments.oracle_cem import MAX_STEPS, ORACLE_CFG

EPISODES = Paths().data_dir / "jepa" / "grade_e" / "episodes"
MODEL_PATH = Paths().checkpoints_dir / "state_mlp.pt"
OUT = Paths().runs_dir / "state_mlp"


def mlp_agent(task, seed):
    return StateMLPAgent(str(MODEL_PATH), STAGE_COSTS[task], ORACLE_CFG, seed=seed)


def plot_horizon(errors, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = sorted(errors)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(h, [errors[k][0] for k in h], marker="o", label="gripper position")
    ax.plot(h, [errors[k][1] for k in h], marker="o", label="object position")
    ax.set_xlabel("steps predicted open-loop")
    ax.set_ylabel("mean error (cm)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("State MLP: held-out multi-step error")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--audit-episodes", type=int, default=3)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. train
    episodes = state_mlp.load_episodes(EPISODES)
    names = sorted(episodes)
    val = set(np.random.default_rng(0).choice(names, size=max(1, len(names) // 5), replace=False))
    model, history = state_mlp.train(episodes, val)
    state_mlp.save(model, MODEL_PATH)
    write_csv(history, OUT / "training.csv")

    # 2. multi-step error on held-out episodes
    errors = state_mlp.rollout_errors(model, {k: episodes[k] for k in val})
    write_csv([{"horizon": h, "ee_err_cm": round(e, 3), "obj_err_cm": round(o, 3)} for h, (e, o) in errors.items()],
              OUT / "horizon_error.csv")
    plot_horizon(errors, OUT / "horizon_error.png")
    print("held-out error (cm) by horizon:", {h: tuple(round(x, 2) for x in v) for h, v in errors.items()}, flush=True)

    # 3. CEM with the MLP, every stage
    summary, runs = {"val_episodes": sorted(val), "final_val_mse": history[-1]}, {}
    for task in MAX_STEPS:
        rows = run_many(grade_e_adapter, functools.partial(mlp_agent, task, 0), list(range(args.seeds)), task,
                        MAX_STEPS[task], workers=args.workers, label=f"{task}/state_mlp_cem")
        write_csv(rows, OUT / f"{task}_state_mlp_cem.csv")
        runs[f"{task}/state_mlp_cem"] = rows
        summary[task] = summarize(rows)
        print(task, summary[task], flush=True)
    plot_runs(runs, OUT / "success.png", title="Grade E scene: CEM with the learned state MLP")

    # 4. audit: predicted vs real cost of every chosen plan, full task
    audit = []
    for seed in range(args.audit_episodes):
        adapter, agent = grade_e_adapter(), StateMLPAgent(str(MODEL_PATH), STAGE_COSTS["place"], ORACLE_CFG, audit=True)
        row = run_episode(adapter, agent, seed, "place", MAX_STEPS["place"])
        audit += [{"seed": seed, **r} for r in agent.audit_log]
        adapter.close()
        print(f"audit seed {seed}: success={row['success']} steps={row['steps']}", flush=True)
    write_csv(audit, OUT / "audit.csv")
    gaps = np.array([r["real_cost"] - r["predicted_cost"] for r in audit])
    false_grasps = sum(r["pred_grasped"] and not r["real_grasped"] for r in audit)
    summary["audit"] = {
        "plans": len(audit), "mean_real_minus_predicted_cost": round(float(gaps.mean()), 3),
        "plans_much_worse_than_predicted (gap > 1)": int((gaps > 1).sum()),
        "plans_predicting_a_grasp_that_did_not_happen": int(false_grasps),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary["audit"], indent=2))


if __name__ == "__main__":
    main()
