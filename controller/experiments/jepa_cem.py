"""Level E deliverable: CEM on the JEPA world model, closed loop in M1's env (Grade E scene).

The agent sees only the static camera (64-frame clips through frozen V-JEPA 2) and proprio, and
plans with D toward the goal state of an encoded goal clip. Default D: the rough model trained
locally by controller.experiments.jepa_pipeline with M2's code (pass M2's own with --checkpoint).

Offline checks first (cached dataset, no sim):
  goal_cost    latent distance to the goal along M1's successful recorded episodes (should fall)
  actions      D's predicted gripper move for unit +-x/y/z actions vs the real one (~4 cm)
  horizon      gripper error of D's open-loop rollouts, 1..10 steps, on held-out episodes
Then closed loop, two cost variants: z+p (M2's training loss terms) and z only (pure visual goal).

    .venv/bin/python -m controller.experiments.jepa_cem [--seeds 5] [--max-steps 150]
"""

import argparse
import csv
import json

import numpy as np
import torch

from controller.adapters.m1_adapter import grade_e_adapter
from controller.adapters.m2_adapter import JEPADynamics, JEPAEncoder, jpeg_roundtrip
from controller.agents import JEPACEMAgent
from controller.config import CEMConfig, Paths
from controller.eval import run_episode, summarize, write_csv
from controller.experiments.jepa_pipeline import CHECKPOINT, ROOT

JEPA_CFG = CEMConfig(horizon=10, n_samples=300, n_elites=30, n_iters=4, init_std=0.5, min_std=0.05, execute_steps=2)
VARIANTS = {"jepa_z_and_p": (1.0, 1.0), "jepa_z_only": (1.0, 0.0)}
OUT = Paths().runs_dir / "jepa"


def load_cache():
    manifest = json.loads((ROOT / "manifest_branches.json").read_text())
    meta = json.loads((ROOT / "features_branches" / "meta.json").read_text())
    latents = np.load(ROOT / "features_branches" / "latents.npy")
    return manifest, meta["index"], latents, (ROOT / manifest["episodes_dir"]).resolve()


def episode_rows(episodes_dir, name):
    with (episodes_dir / name / "data.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def check_goal_cost(d, z_goal, manifest, index, latents, episodes_dir):
    """Normalized latent MSE to the goal at each cached serial of successful episodes."""
    goal = (torch.as_tensor(z_goal) - d.z_mean) / d.z_std
    by_ep = {}
    for s in manifest["samples"]:
        if "branch_action" in s:
            continue
        by_ep.setdefault(s["episode"], set()).add(s["source"])
    curves = []
    for ep, serials in sorted(by_ep.items()):
        meta = json.loads((episodes_dir / ep / "meta.json").read_text())
        if not meta["success"]:
            continue
        n = meta["steps"]
        for t in sorted(serials):
            z = (torch.as_tensor(latents[index[f"{ep}:{t}"]]) - d.z_mean) / d.z_std
            curves.append((t / n, (z - goal).square().mean().item()))
    curves.sort()
    bins = np.linspace(0, 1, 6)
    means = [float(np.mean([v for x, v in curves if lo <= x < hi or (hi == 1 and x == 1)])) for lo, hi in zip(bins[:-1], bins[1:])]
    return {"episode_fraction_bins": [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(bins[:-1], bins[1:])],
            "z_goal_mse": [round(m, 3) for m in means]}


@torch.no_grad()
def check_actions(d, manifest, index, latents):
    """Predicted gripper displacement (cm) for unit actions on held-out states."""
    val = [s for s in manifest["samples"] if s["split"] == "val" and "branch_action" not in s][:40]
    out = {}
    for axis, i in (("x", 0), ("y", 1), ("z", 2)):
        moves, dz = [], []
        for s in val:
            st = d.state(latents[index[f"{s['episode']}:{s['source']}"]], s["p"])
            preds = []
            for sign in (1.0, -1.0):
                a = torch.zeros(1, 5)
                a[0, i], a[0, 4] = sign, s["p"][-1]
                preds.append(d(st[None], a)[0])
            p_plus, p_minus = d.proprio(preds[0]), d.proprio(preds[1])
            moves.append(100 * float(p_plus[14 + i] - p_minus[14 + i]) / 2)
            dz.append(float((preds[0][: d.z_dim] - preds[1][: d.z_dim]).norm() / (preds[0][: d.z_dim] - st[: d.z_dim]).norm().clamp_min(1e-6)))
        out[axis] = {"predicted_move_cm": round(float(np.mean(moves)), 2), "latent_plus_vs_minus_rel": round(float(np.mean(dz)), 3)}
    return out


@torch.no_grad()
def check_horizon(d, manifest, index, latents, episodes_dir, max_h=10):
    """Gripper position error (cm) after h open-loop steps of D, starting from cached clips."""
    val = [s for s in manifest["samples"] if s["split"] == "val" and "branch_action" not in s]
    errs = {h: [] for h in range(1, max_h + 1)}
    cache = {}
    for s in val:
        rows = cache.setdefault(s["episode"], episode_rows(episodes_dir, s["episode"]))
        t = s["source"]
        if t + max_h > len(rows):
            continue
        st = d.state(latents[index[f"{s['episode']}:{t}"]], s["p"])[None]
        for h in range(1, max_h + 1):
            r = rows[t + h - 1]  # serial t+h is row index t+h-1
            a = torch.tensor([[float(r[c]) for c in ("action_dx", "action_dy", "action_dz", "action_dyaw", "action_gripper")]])
            st = d(st, a)
            ee = d.proprio(st[0])[14:17].numpy()
            errs[h].append(100 * float(np.linalg.norm(ee - [float(r["ee_x"]), float(r["ee_y"]), float(r["ee_z"])])))
    return {h: round(float(np.mean(v)), 2) for h, v in errs.items() if v}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--skip-offline", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    encoder = JEPAEncoder()
    d = JEPADynamics(args.checkpoint)
    adapter = grade_e_adapter()
    adapter.reset(seed=0)
    goal_frames, goal_p, ok = adapter.goal_frames(n=d.clip_frames, camera=d.camera)
    z_goal = encoder.encode_clip(np.stack([jpeg_roundtrip(f) for f in goal_frames]))
    summary = {"checkpoint": args.checkpoint, "architecture": d.architecture, "cem_config": JEPA_CFG.__dict__,
               "goal_expert_succeeded": bool(ok), "goal_ee": [round(float(x), 3) for x in goal_p[14:17]]}

    if not args.skip_offline:
        manifest, index, latents, episodes_dir = load_cache()
        summary["goal_cost_along_successful_episodes"] = check_goal_cost(d, z_goal, manifest, index, latents, episodes_dir)
        summary["action_sensitivity"] = check_actions(d, manifest, index, latents)
        summary["gripper_error_cm_by_horizon"] = check_horizon(d, manifest, index, latents, episodes_dir)
        for k in ("goal_cost_along_successful_episodes", "action_sensitivity", "gripper_error_cm_by_horizon"):
            print(k, summary[k], flush=True)

    for name in args.variants:
        zw, pw = VARIANTS[name]
        rows, plans = [], []
        for seed in range(args.seeds):
            agent = JEPACEMAgent(args.checkpoint, encoder, JEPA_CFG, z_weight=zw, p_weight=pw, seed=seed)
            row = run_episode(adapter, agent, seed, "place", args.max_steps)
            row["label"] = name
            rows.append(row)
            plans += [{"seed": seed, **r} for r in agent.plan_log]
            print(name, {k: row[k] for k in ("seed", "success", "steps", "min_ee_obj", "grasped_ever", "final_obj_target_xy", "final_ee")}, flush=True)
        write_csv(rows, OUT / f"{name}.csv")
        write_csv(plans, OUT / f"{name}_plans.csv")
        summary[name] = summarize(rows)
        print(name, summary[name], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    adapter.close()


if __name__ == "__main__":
    main()
