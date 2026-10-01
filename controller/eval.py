"""Run any agent on any env for many seeds, log one CSV row per episode, summarize and plot.

    rows = run_many(make_adapter, make_agent, seeds, task="place", max_steps=200, workers=8)
    write_csv(rows, "data/runs/oracle/place.csv"); print(summarize(rows))

make_adapter / make_agent are zero-argument factories (top-level functions or functools.partial),
because every worker process builds its own env and agent.
"""

import csv
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from controller.interfaces import F_EE, F_GRASPED, F_OBJ, F_REST_Z, F_TARGET

REACH_TOL = 0.03  # gripper within 3 cm of the object centre
LIFT_TOL = 0.05  # object held at least 5 cm above its resting height


def task_success(task, features, info):
    """Has `task` (reach / lift / place) succeeded in this step? place uses M1's own success flag."""
    if task == "reach":
        return np.linalg.norm(features[F_EE] - features[F_OBJ]) < REACH_TOL
    if task == "lift":
        return bool(features[F_GRASPED]) and features[F_OBJ][2] - features[F_REST_Z] > LIFT_TOL
    if task == "place":
        return bool(info["success"])
    raise ValueError(f"unknown task {task}")


def run_episode(adapter, agent, seed, task, max_steps):
    """One episode; stops at success, at the env's own end, or after max_steps."""
    obs, _ = adapter.reset(seed=seed)
    agent.reset(adapter, obs)
    step_times, success, steps, collisions = [], False, 0, 0
    min_reach, max_lift, grasped_ever = np.inf, 0.0, False
    for steps in range(1, max_steps + 1):
        t0 = time.perf_counter()
        action = agent.act(adapter, obs)
        step_times.append(time.perf_counter() - t0)
        obs, _, terminated, truncated, info = adapter.step(action)
        f = adapter.task_features()
        collisions += int(info["obstacle_contact"])
        min_reach = min(min_reach, float(np.linalg.norm(f[F_EE] - f[F_OBJ])))
        grasped_ever |= bool(f[F_GRASPED])
        if f[F_GRASPED]:
            max_lift = max(max_lift, float(f[F_OBJ][2] - f[F_REST_Z]))
        if task_success(task, f, info):
            success = True
            break
        if terminated or truncated:
            break
    return {
        "seed": seed, "task": task, "success": int(success), "steps": steps,
        "final_obj_target_xy": round(float(np.linalg.norm(f[F_OBJ][:2] - f[F_TARGET][:2])), 4),
        "min_ee_obj": round(min_reach, 4), "grasped_ever": int(grasped_ever), "max_lift": round(max_lift, 4),
        "collisions": collisions,
        "time_per_step_s": round(float(np.mean(step_times)), 4),
        "max_time_per_step_s": round(float(np.max(step_times)), 4),
    }


def _run_one(job):
    make_adapter, make_agent, seed, task, max_steps = job
    adapter = make_adapter()
    try:
        return run_episode(adapter, make_agent(), seed, task, max_steps)
    finally:
        adapter.close()


def run_many(make_adapter, make_agent, seeds, task, max_steps, workers=1, label=None):
    """run_episode for every seed, in `workers` processes; rows come back sorted by seed."""
    jobs = [(make_adapter, make_agent, s, task, max_steps) for s in seeds]
    if workers <= 1:
        rows = [_run_one(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_run_one, jobs)
    for r in rows:
        r["label"] = label or task
    return sorted(rows, key=lambda r: r["seed"])


def summarize(rows):
    """Success rate and averages over a list of episode rows."""
    ok = [r for r in rows if r["success"]]
    return {
        "episodes": len(rows), "success_rate": round(len(ok) / len(rows), 3),
        "steps_to_success_mean": round(float(np.mean([r["steps"] for r in ok])), 1) if ok else None,
        "final_obj_target_xy_mean": round(float(np.mean([r["final_obj_target_xy"] for r in rows])), 4),
        "grasp_rate": round(float(np.mean([r["grasped_ever"] for r in rows])), 3),
        "time_per_step_s_mean": round(float(np.mean([r["time_per_step_s"] for r in rows])), 3),
        "max_time_per_step_s": round(float(np.max([r["max_time_per_step_s"] for r in rows])), 3),
    }


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path):
    with Path(path).open(newline="") as f:
        return [{k: (v if k in ("task", "label") else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def plot_runs(runs, path, title=""):
    """Bar chart of success rate and time per step for {label: rows}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(runs)
    success = [100 * np.mean([r["success"] for r in runs[k]]) for k in labels]
    times = [np.mean([r["time_per_step_s"] for r in runs[k]]) for k in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(6, 1.6 * len(labels) + 3), 3.6))
    ax1.bar(labels, success, color="#3b6ea5")
    ax1.set_ylabel("success (%)")
    ax1.set_ylim(0, 105)
    for i, v in enumerate(success):
        ax1.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=9)
    ax2.bar(labels, times, color="#8a8a8a")
    ax2.set_ylabel("mean time per step (s)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
