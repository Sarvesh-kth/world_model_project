"""Opt-in live A-to-B control: real-state SAC versus Q-observed SAC and JEPA MPC.

The online V-JEPA encoder uses the pooling, clip length and PCA basis recorded in the feature cache (encode.py).
The planner is a CEM anchored to the SAC guide: the guide is candidate 0, the gripper follows it, part of the noise
is shared over the horizon, the penalty head's imagined contacts are added to the score, and a candidate only
replaces the guide when it beats the guide's own forecast by --guide-margin."""
import argparse
import copy
import datetime
import json
import os
import pathlib
import shutil
import sys
import time

# cameras render offscreen through EGL; must be set before mujoco is imported
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np

from environment.config import Config, load_config
from data_collection.scripted_policy import ScriptedPickPlace
from .data import digest, load_run, outcome_metrics, provenance, write_json
from .decision import save_csv
from .pipeline import SIMULATION, run_command
from .task_control import GoalReward, TaskSession, policy_observation, predicted_reward

METHODS = ("scripted", "rl_true", "rl_q", "jepa_mpc")


class WorldModels:
    def __init__(self, root, tag, encoder, dynamics_tag=None, blind=False):
        import torch
        from .train import load_model
        from .encode import Encoder, PINNED_REVISION
        if not torch.cuda.is_available():
            raise RuntimeError("live JEPA encoding needs CUDA")
        self.torch = torch
        self.d, self.dc = load_model(root, "dynamics", tag if dynamics_tag is None else dynamics_tag)
        # blind = the proprio-only twin readout_p: a control for how much the image contributes
        self.q, self.qc = load_model(root, "readout_p" if blind else "readout", tag)
        # optional learned penalty head R(z,p) -> proximity, collision, table_hit (reward stage of train.py)
        self.r, self.rc = (load_model(root, "reward", tag) if (pathlib.Path(root) / "attempts" / tag / "models/reward.pt").exists()
                           else (None, None))
        self.use_penalties = self.r is not None
        self.penalty_scale = 1.0
        self.width_gate = True
        # same frozen weights, pooling, PCA basis and clip settings as the cached features
        pooling = encoder["pooling"].split("_pca")[0]
        pca = np.load(pathlib.Path(root) / "features/pca.npz") if pooling != "mean_all_encoder_tokens" else None
        if pca is not None and digest(pathlib.Path(root) / "features/pca.npz") != encoder["pca_sha256"]:
            raise ValueError("PCA basis changed after the features were encoded")
        # offline HF configs do not expose a commit hash; the weights were loaded with the pinned revision
        revision = encoder["model_revision"] or PINNED_REVISION
        self.encoder = Encoder(encoder["model"], revision, "mean_all" if pca is None else pooling,
                               pca=pca, clip_frames=encoder["clip_frames"], stride=encoder.get("frame_stride", 1),
                               dtype=encoder.get("dtype", "bf16"), spatial_pool=encoder.get("spatial_pool", 1))
        if (self.encoder.revision or revision) != revision:
            raise ValueError("encoder revision changed")

    def encode(self, frames):
        z = self.encoder.encode_frames(frames)
        if z.shape != (self.dc["z_dim"],) or not np.isfinite(z).all():
            raise ValueError("invalid online JEPA vector")
        return z

    def predict(self, z, p, actions):
        from .train import normalized
        with self.torch.inference_mode():
            zz, pp = self.d(normalized(z, self.dc, "z"), normalized(p, self.dc, "p"),
                            self.torch.as_tensor(actions, dtype=self.torch.float32))
        return (zz.numpy()*self.dc["z_std"].numpy()+self.dc["z_mean"].numpy(),
                pp.numpy()*self.dc["p_std"].numpy()+self.dc["p_mean"].numpy())

    def read(self, z, p):
        from .train import readout
        xyz, held = readout(self.q, self.qc, z, p)
        # fingers closed to under 2 cm hold nothing (the cube is 4.5 cm wide); without this Q kept reporting
        # "held" after a missed grasp and the policy carried air to B in every failed episode
        if self.width_gate:
            held = held*(np.asarray(p)[:, 18] > .02)
        return xyz, held

    def penalty(self, z, p, weights):
        """Weighted imagined penalty per candidate, same sign convention as the reward formula (<= 0)."""
        from .train import normalized
        if self.r is None or not self.use_penalties:
            return np.zeros(len(z), np.float32)
        with self.torch.inference_mode():
            y = self.r(self.torch.cat((normalized(z, self.rc, "z"), normalized(p, self.rc, "p")), -1)).numpy()
        proximity = np.clip(y[:, 0], -1, 0)
        collision, table = 1/(1+np.exp(-y[:, 1])), 1/(1+np.exp(-y[:, 2]))
        # scaled above 1 the planner leaves the guide earlier, before the forecast reaches the wall
        return (self.penalty_scale*(weights["proximity"]*proximity - weights["collision"]*collision
                                    - weights["table_hit"]*table)).astype(np.float32)


def terminal_values(policy, observations):
    """SAC's critic estimates continuation value; this is not our visual Q readout."""
    import torch
    with torch.inference_mode():
        obs, _ = policy.policy.obs_to_tensor(np.asarray(observations, np.float32))
        actions = policy.actor(obs, deterministic=True)
        values = torch.cat(policy.critic(obs, actions), dim=1).min(dim=1).values
    return values.cpu().numpy()


def forecasts(models, policy, z, p, actions, memory, goal, rest_z, step, cfg, terminal_weight):
    n, horizon, _ = actions.shape
    zs = [np.repeat(z[None], n, 0)]
    ps = [np.repeat(p[None], n, 0)]
    xyz, held = models.read(zs[0], ps[0])
    xyzs, grasps = [xyz], [held]
    memories = [copy.deepcopy(memory) for _ in range(n)]
    rewards = np.zeros((n, horizon), np.float32)
    penalties = np.zeros((n, horizon), np.float32)
    valid = np.ones(n, bool)
    rejected = [set() for _ in range(n)]

    def require(ok, reason):
        valid[:] &= ok
        for i in np.flatnonzero(~ok):
            rejected[i].add(reason)

    finished = np.zeros(n, bool)
    for t in range(horizon):
        zz, pp = models.predict(zs[-1], ps[-1], actions[:, t])
        finite = np.isfinite(zz).all(1) & np.isfinite(pp).all(1)
        require(finite, "D_nonfinite")
        # Freeze nonfinite candidates to permit a report; they stay disqualified.
        zz[~finite], pp[~finite] = zs[-1][~finite], ps[-1][~finite]
        xyz, held = models.read(zz, pp)
        finite = np.isfinite(xyz).all(1) & np.isfinite(held)
        require(finite, "Q_nonfinite")
        # the open gripper measures up to 0.083; D overshoots it by a few mm during the carry, which is not a fault
        require((pp[:, 18] >= -.002) & (pp[:, 18] <= .09), "finger_width")
        require((np.abs(pp[:, 14:17]) < 5).all(1), "ee_bounds")
        require((np.abs(xyz) < 5).all(1), "object_bounds")
        require((np.abs(pp) < 100).all(1), "robot_bounds")
        require((np.abs(zz) < 1e4).all(1), "latent_bounds")
        xyz[~finite], held[~finite] = xyzs[-1][~finite], grasps[-1][~finite]
        zs.append(zz); ps.append(pp); xyzs.append(xyz); grasps.append(held)
        # learned obstacle/table penalties on the imagined state (zero when no penalty head is trained)
        imagined_penalty = models.penalty(zz, pp, cfg.rewards.weights)
        penalties[:, t] = imagined_penalty
        for i in range(n):
            if finished[i]:
                continue
            failed = xyz[i, 2] < cfg.table.height-cfg.episode.fall_margin or step+t+1 >= cfg.episode.max_steps
            rewards[i, t], _ = predicted_reward(memories[i], pp[i], xyz[i], held[i], actions[i, t],
                                                goal, rest_z, failed)
            rewards[i, t] += imagined_penalty[i]
            finished[i] = memories[i].task_succeeded or failed
    observations = [policy_observation(ps[-1][i], xyzs[-1][i], grasps[-1][i], memories[i],
                    goal, rest_z, max(0, 1-(step+horizon)/cfg.episode.max_steps)) if valid[i]
                    else policy_observation(p, xyzs[0][i], grasps[0][i], memory, goal, rest_z, 0)
                    for i in range(n)]
    continuation = terminal_values(policy, observations) if terminal_weight else np.zeros(n)
    continuation[finished | ~valid] = 0.0
    discounted = rewards @ np.power(policy.gamma, np.arange(horizon))
    scores = discounted + terminal_weight*policy.gamma**horizon*continuation
    require(np.isfinite(scores), "score_nonfinite")
    scores[~valid] = -1e9
    return {"actions": actions, "z": np.stack(zs, 1), "p": np.stack(ps, 1),
            "xyz": np.stack(xyzs, 1), "held": np.stack(grasps, 1), "rewards": rewards,
            "scores": scores, "valid": valid, "terminal_critic": continuation, "penalties": penalties,
            "discounted_reward": discounted,
            "invalid_reasons": np.asarray([";".join(sorted(r)) for r in rejected])}


def actor_sequence(models, policy, z, p, memory, goal, rest_z, step, cfg, horizon):
    memory = copy.deepcopy(memory)
    z, p = z.copy()[None], p.copy()[None]
    actions = []
    for t in range(horizon):
        xyz, held = models.read(z, p)
        obs = policy_observation(p[0], xyz[0], held[0], memory, goal, rest_z,
                                 max(0, 1-(step+t)/cfg.episode.max_steps))
        action, _ = policy.predict(obs, deterministic=True)
        actions.append(action)
        z, p = models.predict(z, p, action[None])
        if not np.isfinite(z).all() or not np.isfinite(p).all():
            actions.extend([action.copy()]*(horizon-t-1))
            break
        xyz, held = models.read(z, p)
        if not np.isfinite(xyz).all() or not np.isfinite(held).all():
            actions.extend([action.copy()]*(horizon-t-1))
            break
        predicted_reward(memory, p[0], xyz[0], held[0], action, goal, rest_z)
    return np.asarray(actions, np.float32)


def plan(models, policy, z, p, memory, goal, rest_z, step, cfg, args, rng, previous=None):
    horizon = min(args.horizon, cfg.episode.max_steps-step)
    guide = actor_sequence(models, policy, z, p, memory, goal, rest_z, step, cfg, horizon)
    # CEM noise around the SAC guide; the original hardcoded 0.6 lets the search leave the policy's support
    std_floor = np.full(5, min(.1, args.cem_std), np.float32)
    # the gripper is the policy's call: D and Q never saw a gripper closed on nothing, so a candidate that
    # closes early looks like a grasp in imagination and the critic rewards it (every 0/5 run failed this way)
    free_gripper = getattr(args, "free_gripper", False)
    if not free_gripper:
        std_floor[-1] = 0
    gripper = np.where(guide[:, -1] > 0, 1, -1).astype(np.float32)
    mean, std = guide.copy(), np.full_like(guide, args.cem_std)
    std[:, -1] = args.cem_std if free_gripper else 0
    best = None
    for iteration in range(args.iterations):
        # part of each candidate's noise is shared across its horizon: independent per-step noise averages out
        # (under 1 cm of lateral spread over 8 steps), so no candidate could ever route around an obstacle
        noise = rng.normal(size=(args.population, horizon, 5))
        smooth = getattr(args, "cem_smooth", 0.0)
        if smooth > 0:
            noise = np.sqrt(smooth)*rng.normal(size=(args.population, 1, 5)) + np.sqrt(1-smooth)*noise
        actions = np.clip(mean + std*noise, -1, 1).astype(np.float32)
        actions[..., -1] = np.where(actions[..., -1] > 0, 1, -1)
        if not free_gripper:
            actions[..., -1] = gripper
        actions[0] = guide
        if previous is not None:
            shifted = np.concatenate((previous[1:], previous[-1:]), axis=0)
            actions[1] = shifted[:horizon]
            if not free_gripper:
                actions[1, :, -1] = gripper
        pool = forecasts(models, policy, z, p, actions, memory, goal, rest_z, step, cfg, args.terminal_weight)
        selected = int(np.argmax(pool["scores"]))
        if best is None or pool["scores"][selected] > best["pool"]["scores"][best["selected"]]:
            best = {"pool": pool, "selected": selected, "iteration": iteration}
        elite_ids = np.argsort(pool["scores"])[-args.elites:]
        elites = actions[elite_ids]
        mean, std = elites.mean(0), np.maximum(elites.std(0), std_floor)
    pool, selected = best["pool"], best["selected"]
    best["fallback"] = not bool(pool["valid"][selected])
    # candidate 0 is always the guide, so its forecast is scored with the same D/Q/critic as everything else;
    # a candidate only replaces the guide when it beats that forecast by more than the model noise
    best["guide_score"] = float(pool["scores"][0])
    best["best_score"] = float(pool["scores"][selected])
    margin = getattr(args, "guide_margin", 0.0)
    best["kept_guide"] = best["fallback"] or (bool(pool["valid"][0]) and best["best_score"]-best["guide_score"] <= margin)
    if best["kept_guide"]:
        selected = 0
    best["selected"] = selected
    best["action"] = guide[0] if best["fallback"] else pool["actions"][selected, 0]
    best["sequence"] = guide if best["fallback"] else pool["actions"][selected]
    return best


def save_frame(session, folder, serial, history):
    frame = session.sim.render("static")
    path = folder / "frames" / f"static_{serial:04d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, session.cfg.data.jpeg_quality]):
        raise OSError(f"failed to save {path}")
    # Encode the same JPEG representation used by the existing offline cache.
    history.append(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    del history[:-64]


def viewer_process(cfg, layout, queue):
    """Separate process that owns the MuJoCo window: the episode process renders its cameras with EGL and a window
    cannot share that context, so this one rebuilds the same scene and mirrors the robot state it is sent."""
    import mujoco
    import mujoco.viewer
    from environment import scene, EpisodeLayout
    from environment.config import Config
    model = scene.build_scene(Config.nested(cfg), EpisodeLayout.from_dict(layout))
    data = mujoco.MjData(model)
    previous = target = None
    arrived = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            try:
                state = queue.get(timeout=1/60)
                if state is None:
                    break
                previous, target, arrived = (target if target is not None else state[0]), state[0], time.monotonic()
                gap = state[1]
            except Exception:
                pass
            if target is None:
                continue
            # the planner delivers 2-3 states a second; glide between the last two so the window moves smoothly
            alpha = min(1.0, (time.monotonic()-arrived)/max(gap, 1e-3))
            data.qpos[:] = previous+alpha*(target-previous)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1/60)
    # the viewer thread can hang or crash on a normal exit (see play.py), so leave the hard way
    os._exit(0)


class LiveViewer:
    """Mirror of the simulation in a MuJoCo window, unless --headless or there is no display (batch runs)."""
    def __init__(self, session, args, config):
        self.queue = self.process = None
        if getattr(args, "headless", False) or not os.environ.get("DISPLAY"):
            return
        import multiprocessing
        context = multiprocessing.get_context("spawn")
        self.queue = context.Queue(maxsize=4)
        self.process = context.Process(target=viewer_process, daemon=True,
                                       args=(config, session.sim.layout.to_dict(), self.queue))
        # the child reads these at its first `import mujoco`: no EGL there, and no NVIDIA PRIME offload
        # variables either (they leave the window blank, see play.py); this process keeps its own values
        saved = {k: os.environ.pop(k) for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "__NV_PRIME_RENDER_OFFLOAD",
                 "__GLX_VENDOR_LIBRARY_NAME", "__EGL_VENDOR_LIBRARY_FILENAMES") if k in os.environ}
        try:
            self.process.start()
        finally:
            os.environ.update(saved)

    def sync(self, session):
        if self.process is None or not self.process.is_alive():
            return
        now = time.monotonic()
        gap, self.last = (now-self.last if getattr(self, "last", None) else .1), now
        try:
            # the window glides from its last state to this one over the time the step took
            self.queue.put_nowait((session.sim.data.qpos.copy(), gap))
        except Exception:
            pass   # the window is behind; drop this frame

    def close(self):
        if self.process is not None and self.process.is_alive():
            try:
                self.queue.put(None, timeout=1)
            except Exception:
                pass
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()


def episode(root, signature, args, *, session_factory=TaskSession, models=None, policy=None):
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(args.threads)
    cfg = Config.nested(signature["config"])
    seed = args.episode_seed
    folder = root / "episodes" / args.method / f"seed_{seed}"
    if folder.exists():
        # Preserve interrupted evidence instead of mixing two trajectories.
        backup = root / "interrupted" / f"{args.method}_{seed}_{time.time_ns()}"
        backup.parent.mkdir(exist_ok=True)
        folder.rename(backup)
    folder.mkdir(parents=True)
    session = session_factory(cfg, signature["layout"], args.position_jitter)
    checkpoint = getattr(args, "policy_checkpoint", root / "models/sac_best.zip")
    checkpoint_sha = None if args.method == "scripted" else digest(checkpoint)
    frozen = signature.get("frozen_baseline")
    if frozen and args.method != "scripted" and checkpoint_sha != frozen["checkpoint_sha256"]:
        raise ValueError("controller does not use the frozen baseline checkpoint")
    policy = None if args.method == "scripted" else policy if policy is not None else SAC.load(checkpoint, device="cpu")
    visual = args.method in ("rl_q", "jepa_mpc")
    models = (models if models is not None else WorldModels(args.models_run, args.tag, signature["encoder"],
                                                             blind=getattr(args, "blind", False))) if visual else None
    if models is not None and getattr(args, "no_penalties", False):
        models.use_penalties = False
    if models is not None:
        models.width_gate = not getattr(args, "no_width_gate", False)
        models.penalty_scale = getattr(args, "penalty_scale", 1.0)
    goal = np.array([*signature["layout"]["place"], signature["known_rest_z"]], np.float32)
    memory = GoalReward(cfg)
    histories, rows, predictions, candidates = [], [], [], []
    states_p, states_xyz, states_held, states_z, executed_actions = [], [], [], [], []
    previous = None
    started = time.monotonic()
    viewer = None
    try:
        obs = session.reset(seed)
        viewer = LiveViewer(session, args, signature["config"])
        step_time = 1/(cfg.control.hz*max(getattr(args, "speed", 1.0), 1e-3))
        scripted = ScriptedPickPlace(session.sim, np.random.default_rng(seed)) if args.method == "scripted" else None
        save_frame(session, folder, 0, histories)
        z = models.encode(histories) if visual else None
        states_p.append(session.obs["proprio"].copy())
        states_xyz.append(session.obs["state"][:3].copy())
        states_held.append(bool(session.sim._check_contacts()[0]))
        if visual:
            states_z.append(z.copy())
        if visual:
            xyz, held = models.read(z[None], session.obs["proprio"][None])
            xyz, held = xyz[0], float(held[0])
        for step in range(cfg.episode.max_steps):
            p = session.obs["proprio"].copy()
            decision = None
            decision_start = time.monotonic()
            if scripted is not None:
                action = scripted.act()
            elif args.method in ("rl_true", "bc_true"):
                action, _ = policy.predict(obs, deterministic=True)
            elif args.method == "rl_q":
                estimate = policy_observation(p, xyz, held, memory, goal, signature["known_rest_z"],
                                              1-step/cfg.episode.max_steps)
                action, _ = policy.predict(estimate, deterministic=True)
            else:
                decision = plan(models, policy, z, p, memory, goal, signature["known_rest_z"], step,
                                cfg, args, np.random.default_rng(seed*1000+step), previous)
                action, previous = decision["action"], decision["sequence"]
            decision_seconds = time.monotonic()-decision_start
            obs, reward, done, timeout, info = session.step(action)
            if viewer.process is not None:
                # play at --speed times real time while the window is open
                viewer.sync(session)
                time.sleep(max(0, step_time-(time.monotonic()-decision_start)))
            proposed_action = np.asarray(action).copy()
            action = np.asarray(info.get("executed_action", action))
            intervened = bool(info.get("forced_release", False))
            if intervened:
                previous = None
            if scripted is not None and info.get("restart_scripted"):
                scripted = ScriptedPickPlace(session.sim, np.random.default_rng(seed+step+1))
            actual_xyz, actual_p = session.obs["state"][:3].copy(), session.obs["proprio"].copy()
            states_p.append(actual_p); states_xyz.append(actual_xyz)
            states_held.append(info["held_endpoint"]); executed_actions.append(np.asarray(action).copy())
            save_frame(session, folder, step+1, histories)
            row = {"step": step+1, "time_s": float(session.sim.data.time),
                   "true_original_reward": info["original_reward"], "control_reward": reward,
                   "actual_held": info["held_endpoint"],
                   "actual_goal_distance_cm": float(100*np.linalg.norm(actual_xyz[:2]-goal[:2])),
                   "actual_lift_cm": float(100*(actual_xyz[2]-session.rest_z)),
                   "task_success": info["task_success"], "table_contact": info["table_contact"],
                   "obstacle_contact": info["obstacle_contact"], "decision_seconds": decision_seconds}
            row.update({f"action_{k}": float(v) for k, v in zip(("dx", "dy", "dz", "dyaw", "gripper"), action)})
            if "forced_release" in info:
                row.update({"forced_release": intervened, "phase": info["phase"],
                            "proposed_gripper": float(proposed_action[-1])})
            row.update({f"actual_object_{k}": float(v) for k, v in zip("xyz", actual_xyz)})
            row.update({f"true_component_{k}": float(v) for k, v in info["reward_components"].items()})
            row.update({f"control_component_{k}": float(v) for k, v in info["control_reward_components"].items()})
            if visual:
                next_z = models.encode(histories)
                states_z.append(next_z.copy())
                next_xyz, next_held = models.read(next_z[None], actual_p[None])
                next_xyz, next_held = next_xyz[0], float(next_held[0])
                row.update({"Q_actual_observation_held_probability": next_held,
                            "Q_actual_observation_xyz_mae_cm": float(100*np.abs(next_xyz-actual_xyz).mean())})
                row.update({f"Q_actual_observation_{k}": float(v) for k, v in zip("xyz", next_xyz)})
                failed = (next_xyz[2] < cfg.table.height-cfg.episode.fall_margin or step+1 >= cfg.episode.max_steps)
                estimated_reward, _ = predicted_reward(memory, actual_p, next_xyz, next_held, action,
                                                       goal, signature["known_rest_z"], failed)
                row["Q_observed_supported_reward"] = estimated_reward
                if decision is not None:
                    pool, selected = decision["pool"], decision["selected"]
                    row.update({"predicted_score": float(pool["scores"][selected]),
                                "predicted_penalty_sum": float(pool["penalties"][selected].sum()),
                                "planner_fallback_to_RL_Q": decision["fallback"],
                                "guide_score": decision["guide_score"], "best_candidate_score": decision["best_score"],
                                "kept_guide": decision["kept_guide"],
                                "valid_candidates": int(pool["valid"].sum())})
                    if not decision["fallback"] and not intervened:
                        predicted_xyz = pool["xyz"][selected, 1]
                        row.update({"D_Q_one_step_xyz_mae_cm": float(100*np.abs(predicted_xyz-actual_xyz).mean()),
                            "D_Q_one_step_height_mae_cm": float(100*abs(predicted_xyz[2]-actual_xyz[2])),
                            "D_one_step_latent_normalized_mse": float(np.square((pool["z"][selected, 1]-next_z)/models.dc["z_std"].numpy()).mean()),
                            "D_one_step_ee_mae_cm": float(100*np.abs(pool["p"][selected, 1, 14:17]-actual_p[14:17]).mean()),
                            "predicted_first_supported_reward": float(pool["rewards"][selected, 0])})
                    archive = folder / "forecasts" / f"step_{step+1:04d}.npz"
                    archive.parent.mkdir(exist_ok=True)
                    np.savez_compressed(archive, actions=pool["actions"], p=pool["p"], xyz=pool["xyz"],
                        held=pool["held"], rewards=pool["rewards"], scores=pool["scores"], valid=pool["valid"],
                        invalid_reasons=pool["invalid_reasons"],
                        current_z=z, selected_z=pool["z"][selected], selected=selected,
                        fallback=decision["fallback"], intervention=intervened)
                    for i in range(len(pool["scores"])):
                        candidates.append({"step": step+1, "candidate": i, "selected": i == selected,
                            "executed_selected": i == selected and not decision["fallback"] and not intervened,
                            "valid": bool(pool["valid"][i]), "score": float(pool["scores"][i]),
                            "invalid_reasons": str(pool["invalid_reasons"][i]),
                            "discounted_reward": float(pool["discounted_reward"][i]),
                            "terminal_SAC_critic": float(pool["terminal_critic"][i])})
                    if not decision["fallback"]:
                        for t in range(1, pool["xyz"].shape[1]):
                            predictions.append({"decision_step": step+1, "imagined_step": t,
                                "predicted_x": float(pool["xyz"][selected, t, 0]),
                                "predicted_y": float(pool["xyz"][selected, t, 1]),
                                "predicted_z": float(pool["xyz"][selected, t, 2]),
                                "held_probability": float(pool["held"][selected, t]),
                                "supported_reward": float(pool["rewards"][selected, t-1])})
                z, xyz, held = next_z, next_xyz, next_held
            rows.append(row)
            if (step+1) % 10 == 0 or done or timeout:
                save_csv(folder / "steps.csv", rows)
                print(f"{args.method} seed={seed} step={step+1}: held={info['held_endpoint']} "
                      f"goal_distance={row['actual_goal_distance_cm']:.1f}cm lift={row['actual_lift_cm']:.1f}cm "
                      f"success={info['task_success']}", flush=True)
            if done or timeout:
                break
        save_csv(folder / "steps.csv", rows)
        if predictions:
            save_csv(folder / "selected_forecasts.csv", predictions)
        if candidates:
            save_csv(folder / "candidates.csv", candidates)
        arrays = {"p": np.asarray(states_p), "object_xyz": np.asarray(states_xyz),
                  "held": np.asarray(states_held), "actions": np.asarray(executed_actions),
                  "true_rewards": np.asarray([r["true_original_reward"] for r in rows]),
                  "control_rewards": np.asarray([r["control_reward"] for r in rows])}
        if visual:
            arrays["z"] = np.asarray(states_z)
        np.savez_compressed(folder / "trajectory.npz", **arrays)
        write_json(folder / "trajectory_meta.json", {"states": len(states_p), "actions": len(executed_actions),
            "alignment": "state[t] -> actions[t] -> state[t+1]; frame serial is state index; pad first frame to 64",
            "encoder": signature["encoder"] if visual else None,
            "inputs": "p, action, camera-derived z; object xyz/held are evaluator/training labels only for JEPA methods"})
        from PIL import Image
        images = [Image.open(p).convert("RGB").resize((256, 256)) for p in sorted((folder / "frames").glob("*.jpg"))[::2]]
        images[0].save(folder / "actual.gif", save_all=True, append_images=images[1:],
                       duration=200, loop=0)
        result = {"method": args.method, "seed": seed, **session.result(),
                  "SAC_checkpoint_sha256": checkpoint_sha,
                  "seconds": time.monotonic()-started,
                  "model_fallback_steps": sum(bool(r.get("planner_fallback_to_RL_Q")) for r in rows),
                  "guide_kept_steps": sum(bool(r.get("kept_guide")) for r in rows),
                  "input_contract": "true state for scripted/rl_true; causal JEPA clip + measured p + explicit B for rl_q/jepa_mpc"}
        if visual:
            result["Q_on_actual_observations"] = outcome_metrics(
                [[r[f"Q_actual_observation_{k}"] for k in "xyz"] for r in rows],
                [r["Q_actual_observation_held_probability"] for r in rows], states_xyz[1:], states_held[1:])
        forecasts_compared = [r for r in rows if "D_Q_one_step_height_mae_cm" in r]
        if forecasts_compared:
            result["D_Q_one_step_height_mae_cm"] = float(np.mean([r["D_Q_one_step_height_mae_cm"] for r in forecasts_compared]))
            result["D_one_step_latent_normalized_mse"] = float(np.mean([r["D_one_step_latent_normalized_mse"] for r in forecasts_compared]))
        write_json(folder / "result.json", result)
        print(f"FINISHED {args.method}: success={result['task_success']} goal={result['final_goal_distance_cm']:.1f}cm", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        session.close()


def frozen_baseline(root, cfg, layout, jitter):
    """Verify an already evaluated RL actor; newer integration code may reuse it."""
    state = json.loads((root / "pipeline.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if not state.get("complete") or not summary.get("complete") or not summary.get("gate_passed"):
        raise ValueError("baseline must have completed and passed rl_baseline")
    for name, sha in state["trained"].items():
        if digest(root / "models" / name) != sha:
            raise ValueError(f"baseline training artifact changed: {name}")
    evaluations = root / "evaluations.json"
    if digest(evaluations) != state["evaluations_sha256"]:
        raise ValueError("baseline evaluations changed")
    rows = json.loads(evaluations.read_text())["rl_true"]["episodes"]
    if (len(rows) < 100 or len({r["seed"] for r in rows}) != len(rows)
            or sum(r["task_success"] for r in rows)/len(rows) < .9
            or state["inputs"]["options"]["validation_episodes"] < 20):
        raise ValueError("baseline needs >=90% strict placement over >=100 fresh episodes and >=20 validation episodes")
    training = root / "models/rl_training.json"
    sha = digest(root / "models/sac_best.zip")
    if (sha != summary["SAC_checkpoint_sha256"]
            or sha != json.loads(training.read_text())["checkpoint_sha256"]):
        raise ValueError("baseline model does not match its success report")
    source = state["inputs"]
    contracts = list((SIMULATION / "environment").glob("*.py")) + [SIMULATION / "world_model/vision/task_control.py"]
    for path in contracts:
        if digest(path) != source["code_sha256"][str(path.relative_to(SIMULATION))]:
            raise ValueError("baseline physics/observation/reward code changed; keep its exact runtime contract")
    if layout != source["layout"] or jitter != source["options"]["position_jitter"]:
        raise ValueError("use the baseline's layout and position-jitter for a matched comparison")
    for section in ("sim", "control", "robot", "table", "rewards", "episode"):
        if cfg[section] != source["config"][section]:
            raise ValueError(f"baseline/source-model physical configuration differs: {section}")
    return {"run": str(root), "checkpoint_sha256": sha,
            "training_sha256": digest(training), "evaluations_sha256": digest(evaluations),
            "pipeline_sha256": digest(root / "pipeline.json"),
            "summary_sha256": digest(root / "summary.json"),
            "successes": sum(r["task_success"] for r in rows), "episodes": len(rows),
            "validation_seeds": json.loads(training.read_text())["settings"]["validation_seeds"],
            "test_seeds": [r["seed"] for r in rows],
            "config": source["config"], "layout": source["layout"]}


def reuse_baseline(root, evidence):
    source = pathlib.Path(evidence["run"])
    targets = ((source / "models/sac_best.zip", root / "models/sac_best.zip", evidence["checkpoint_sha256"]),
               (source / "models/rl_training.json", root / "models/rl_training.json", evidence["training_sha256"]))
    for src, dst, sha in targets:
        if digest(src) != sha:
            raise ValueError("frozen baseline changed before copying")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and digest(dst) != sha:
            raise ValueError("output contains a different RL checkpoint/report")
        if not dst.exists():
            shutil.copyfile(src, dst)
        if digest(dst) != sha:
            raise ValueError("copied baseline differs from its verified source")
    write_json(root / "frozen_baseline.json", evidence)
    print(f"REUSED FROZEN SAC: {evidence['successes']}/{evidence['episodes']}; "
          f"sha256={evidence['checkpoint_sha256']}; no training", flush=True)


def inputs(args):
    manifest = load_run(args.models_run)
    features = json.loads((args.models_run / "features/meta.json").read_text())
    if not manifest.get("complete") or features["manifest_sha256"] != digest(args.models_run / "manifest.json"):
        raise ValueError("source dataset/encoder cache is incomplete or mismatched")
    if features.get("camera") != "static" or not features.get("pooling"):
        raise ValueError("this live experiment requires the pinned static-camera encoder cache")
    cfg = Config.nested(copy.deepcopy(manifest["config"]))
    cfg.episode.terminate_on_success = False
    layout = json.loads(args.layout.read_text())
    if layout["object"] != "cube" or layout["scale"] != 1 or layout["obstacles"]:
        raise ValueError("first live attempt supports the fixed unit cube with no obstacles")
    if not cfg.control.yaw.enabled or cfg.cameras.record_hz != cfg.control.hz:
        raise ValueError("need the source 5-action interface and one static frame per control step")
    training_rest = [s["object_xyz"][2] for s in manifest["states"] if s["split"] == "train"
                     and not s["held"] and abs(s["object_xyz"][2]-cfg.table.height) < .06]
    if not training_rest:
        raise ValueError("source training data has no resting-cube height calibration")
    options = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()
               if k not in ("resume", "self_check", "stage", "method", "episode_seed")}
    options["methods"] = " ".join(options["methods"]) if isinstance(options.get("methods"), list) else options.get("methods")
    code = list((SIMULATION / "environment").glob("*.py"))
    code += [SIMULATION / "world_model/vision" / f"{name}.py" for name in
             ("task_control", "rl_control", "control_pipeline", "encode", "train", "data", "pipeline")]
    code += [SIMULATION / "world_model/train_dynamics.py", SIMULATION / "data_collection/scripted_policy.py"]
    baseline = frozen_baseline(args.baseline_run, cfg, layout, args.position_jitter) if args.baseline_run else None
    if baseline:
        seeds = set(range(args.seed+20000, args.seed+20000+args.test_episodes))
        used = set(baseline["test_seeds"] + baseline["validation_seeds"])
        source_options = json.loads((args.baseline_run / "pipeline.json").read_text())["inputs"]["options"]
        baseline_seed = source_options["seed"]
        span = max(3*source_options["demos"], 1000+source_options["rl_steps"]+5)
        if seeds & used or any(baseline_seed <= s < baseline_seed+span for s in seeds):
            raise ValueError("comparison seeds overlap baseline training/validation/evaluation; use a fresh --seed")
    return {"options": options, "config": cfg, "layout": layout, "frozen_baseline": baseline,
            "known_rest_z": float(np.median(training_rest)),
            "encoder": {k: features.get(k) for k in ("model", "model_revision", "camera", "clip_frames",
                                                      "frame_stride", "pooling", "pca_sha256", "dtype", "spatial_pool")},
            "source_manifest_sha256": digest(args.models_run / "manifest.json"),
            "source_features_sha256": digest(args.models_run / "features/meta.json"),
            "source_dynamics_sha256": digest(args.models_run / "attempts" / args.tag / "models/dynamics.pt"),
            "source_readout_sha256": digest(args.models_run / "attempts" / args.tag / "models/readout.pt"),
            "code_sha256": {str(p.relative_to(SIMULATION)): digest(p) for p in sorted(code)},
            "contracts": {"reward": "GoalReward potential progress + strict placement bonus15 + penalties + time cost0.01; original rewards logged separately",
                "reward_gamma": GoalReward.gamma,
                "omitted_imagined_penalties": ["collision", "proximity", "table_hit"],
                "held_probability_threshold": .5, "success": "ever grasped AND held lift>=4cm AND released at B within source success_radius and rest-height tolerance2cm for source settle_steps",
                "terminal_score": "discounted supported reward + terminal_weight * gamma^horizon * minimum SAC critic estimate",
                "coverage": "existing D/Q fitted on local grasp/lift data; full-task extrapolation is experimental"}}


def export(root, target, state):
    target.mkdir(parents=True, exist_ok=True)
    previous = target / "run_info.json"
    if previous.exists() and json.loads(previous.read_text())["inputs"] != state["inputs"]:
        raise ValueError("export belongs to another run")
    write_json(previous, state)
    results = []
    for path in sorted((root / "episodes").glob("*/*/result.json")):
        result = json.loads(path.read_text()); results.append(result)
        for source in path.parent.glob("*"):
            if source.suffix in (".csv", ".json"):
                destination = target / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    for folder in (root / "logs", root / "rl_logs"):
        for source in folder.glob("*"):
            if source.suffix in (".txt", ".csv"):
                destination = target / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    for source in (root / "models/rl_training.json", root / "demos.json", root / "rl_monitor.csv", root / "frozen_baseline.json"):
        if source.exists():
            shutil.copyfile(source, target / source.name)
    summary = []
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        if rows:
            summary.append({"method": method, "episodes": len(rows), "successes": sum(r["task_success"] for r in rows),
                "success_rate": float(np.mean([r["task_success"] for r in rows])),
                "mean_original_return": float(np.mean([r["true_original_return"] for r in rows])),
                "mean_control_return": float(np.mean([r["control_return"] for r in rows])),
                "mean_final_goal_distance_cm": float(np.mean([r["final_goal_distance_cm"] for r in rows])),
                "mean_maximum_lift_cm": float(np.mean([r["maximum_lift_cm"] for r in rows])),
                "table_contact_steps": sum(r["table_contact_steps"] for r in rows),
                "model_fallback_steps": sum(r["model_fallback_steps"] for r in rows)})
    frozen = state["inputs"].get("frozen_baseline")
    write_json(target / "summary.json", {"complete": state["complete"], "rows": summary,
        "frozen_baseline": frozen,
        "note": "A working-library implementation is not a guaranteed trained-policy success. Read actual placement counts; BC uses privileged demonstrations. Q/D full-task forecasts and omitted contact penalties are limitations."})
    if summary:
        save_csv(target / "summary.csv", summary)
    text = ["Live A-to-B control comparison", f"complete={state['complete']}"]
    if frozen:
        text.append(f"Frozen SAC: {frozen['successes']}/{frozen['episodes']} source placements; "
                    f"sha256={frozen['checkpoint_sha256']}; no retraining")
    for row in summary:
        text.append(f"{row['method']}: full placements {row['successes']}/{row['episodes']}; "
                    f"final goal distance={row['mean_final_goal_distance_cm']:.2f}cm; "
                    f"model fallbacks={row['model_fallback_steps']}")
    baseline = next((r for r in summary if r["method"] == "rl_true"), None)
    if baseline:
        demonstrated = baseline["episodes"] >= 5 and baseline["success_rate"] >= .8
        text.append(f"Repeatable SAC baseline on this test={demonstrated} (predeclared criterion: >=80% over >=5 near-fixed-scene episodes).")
    text.append("Actual GIFs/JPEGs and candidate NPZ forecasts remain in the notebook data run. No model success is implied by pipeline completion.")
    (target / "summary.txt").write_text("\n".join(text)+"\n")
    write_json(target / "files.json", {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob("*"))
                                      if p.is_file() and p.name != "files.json"})
    print("\n".join(text), flush=True)


def check():
    cfg = load_config("configs/grade_e.yml")
    cfg.episode.terminate_on_success = False
    goal = np.array([.1, .27, .77])
    p = np.zeros(20); p[14:17] = [.18, -.23, .77]
    p[18:20] = [.06, -1]
    memory = GoalReward(cfg)
    left, right = copy.deepcopy(memory), copy.deepcopy(memory)
    action = np.array([0, 0, .5, 0, -1])
    expected = {"object_pos": np.array([.18, -.23, .88]), "ee_pos": p[14:17], "place_pos": goal,
                "object_rest_z": .77, "grasped": True, "failed": False,
                "obstacle_contacts": 0, "table_contacts": 0, "obstacle_distance": memory.safe_distance}
    actual = GoalReward(cfg).compute(expected, action)[0]
    predicted, _ = predicted_reward(left, p, expected["object_pos"], .9, action, goal, .77)
    assert np.isclose(actual, predicted), "supported true/predicted reward equations differ"
    assert left.was_grasped and not right.was_grasped and not memory.was_grasped
    for _ in range(cfg.episode.settle_steps):
        _, components = predicted_reward(left, p, goal, .1, [0, 0, 0, 0, 1], goal, .77)
    assert components["goal"] == 15 and left.task_succeeded
    _, again = predicted_reward(left, p, goal, .1, [0, 0, 0, 0, 1], goal, .77)
    assert again["goal"] == 0, "bonuses repeated"
    assert policy_observation(p, goal, .1, left, goal, .77, .5).shape == (41,)
    holding = GoalReward(cfg)
    holding.compute(expected, action)
    repeated, _ = predicted_reward(holding, p, expected["object_pos"], .9, action, goal, .77)
    assert repeated < 0, "holding forever must not earn a positive per-step reward"
    empty = GoalReward(cfg)
    for _ in range(cfg.episode.settle_steps):
        _, c = predicted_reward(empty, p, goal, .1, [0, 0, 0, 0, 1], goal, .77)
    assert c["goal"] == 0 and not empty.task_succeeded, "unlifted object counted as placement"
    layout = json.loads((SIMULATION / "configs/grade_e_layout.json").read_text())
    session = TaskSession(cfg, layout)
    try:
        session.reset(34)
        policy = ScriptedPickPlace(session.sim, np.random.default_rng(34))
        for _ in range(cfg.episode.max_steps):
            _, _, done, timeout, _ = session.step(policy.act())
            if done or timeout:
                break
        result = session.result()
        assert result["task_success"], f"real scripted A-to-B reference failed: {result}"
        print(f"PASS: reward parity/history/one-time goal/no holding bonus/41D observation; real scripted placement in {result['steps']} steps")
        print("This checks software and physics, not trained SAC or JEPA control quality.")
    finally:
        session.close()
    deadline = copy.deepcopy(cfg)
    deadline.episode.max_steps = 1
    session = TaskSession(deadline, layout)
    try:
        session.reset(34)
        _, _, done, truncated, info = session.step([0, 0, 0, 0, 1])
        assert done and not truncated and info["task_failed"], "deadline must be a terminal task failure"
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("data/combined_test1"),
                        help="run folder holding features/ (PCA basis) and attempts/<tag>/models/ (Q, R, D)")
    parser.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"),
                        help="completed rl_baseline run with the frozen SAC (models/sac_best.zip); no RL training")
    parser.add_argument("--tag", default="combined")
    parser.add_argument("--layout", type=pathlib.Path, default=pathlib.Path("configs/grade_e_layout.json"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/control"),
                        help="run folder (episodes, frames, gifs); overwritten unless --resume")
    parser.add_argument("--export-dir", type=pathlib.Path, help="default: results/<out name>")
    parser.add_argument("--seed", type=int, default=20474010)
    parser.add_argument("--rl-steps", type=int, default=10000)
    parser.add_argument("--demos", type=int, default=20)
    parser.add_argument("--bc-epochs", type=int, default=120)
    parser.add_argument("--bc-weight", type=float, default=100.0)
    parser.add_argument("--critic-warmup", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--position-jitter", type=float, default=.01, help="metres per xy axis; same distribution for all methods")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--terminal-weight", type=float, default=0.0,
                        help="weight of the SAC critic as terminal value; 0 because D errors exploit it (team value 1)")
    parser.add_argument("--cem-std", type=float, default=0.3, help="CEM sampling std around the SAC guide (team value was 0.6)")
    parser.add_argument("--no-penalties", action="store_true", help="ignore a trained penalty head when scoring candidates")
    parser.add_argument("--penalty-scale", type=float, default=2.0, help="multiply the imagined contact penalties; 2 leaves the guide earlier near obstacles (1 = recorded weights)")
    parser.add_argument("--guide-margin", type=float, default=0.25,
                        help="a CEM candidate replaces the SAC guide only if its imagined score beats the guide's by this much")
    parser.add_argument("--free-gripper", action="store_true", help="let CEM also sample the gripper (team behaviour)")
    parser.add_argument("--blind", action="store_true", help="control: use the proprio-only readout_p instead of Q(z, p)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS, help="which controllers to run")
    parser.add_argument("--no-width-gate", action="store_true", help="do not zero Q's held reading when the fingers are closed on nothing")
    parser.add_argument("--cem-smooth", type=float, default=0.5,
                        help="share of each candidate's noise variance that is constant over the horizon (0 = independent per step)")
    parser.add_argument("--headless", action="store_true", help="no MuJoCo window (the default when there is no display)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed of the window, times real time")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run instead of overwriting it")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--stage", choices=("rl", "episode"), help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--episode-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_check:
        check(); return
    if (min(args.rl_steps, args.demos, args.bc_epochs, args.eval_every, args.validation_episodes,
            args.test_episodes, args.horizon, args.iterations, args.threads) < 1
            or not 2 <= args.elites < args.population or not 0 <= args.position_jitter <= .03
            or args.seed < 0 or args.critic_warmup < 0 or not 0 <= args.bc_weight <= 1000
            or not 0 <= args.terminal_weight <= 2 or not 0 <= args.cem_smooth <= 1):
        parser.error("invalid training/planning/scene settings")
    args.models_run, args.out, args.layout = [p.resolve() for p in (args.models_run, args.out, args.layout)]
    if args.baseline_run:
        args.baseline_run = args.baseline_run.resolve()
        if (args.out == args.baseline_run or args.out in args.baseline_run.parents
                or args.baseline_run in args.out.parents):
            parser.error("output must be outside the frozen baseline run")
    args.export_dir = (args.export_dir or SIMULATION.parent / "results" / args.out.name).resolve()
    if (SIMULATION / "data" not in args.out.parents or args.out == args.models_run
            or args.models_run in args.out.parents or args.out in args.models_run.parents):
        parser.error("use a fresh output under simulation/data, outside source model data")
    if SIMULATION.parent / "results" not in args.export_dir.parents:
        parser.error("keep shareable exports in a dedicated folder under repository results/")
    signature = inputs(args)
    path = args.out / "pipeline.json"
    if args.stage:
        state = json.loads(path.read_text())
        if state["inputs"] != signature:
            raise ValueError("code or settings changed while this run was active; start the run again")
        if args.stage == "rl":
            if signature["frozen_baseline"]:
                reuse_baseline(args.out, signature["frozen_baseline"])
            else:
                from .rl_control import train
                train(args.out, signature["config"], signature["layout"], args)
        else:
            if args.method is None or args.episode_seed is None:
                parser.error("episode stage requires method and seed")
            episode(args.out, signature, args)
        return
    if path.exists() and args.resume:
        state = json.loads(path.read_text())
        if state["inputs"] != signature:
            parser.error("inputs/code/settings changed since this run; drop --resume to start over")
    else:
        # a fresh run replaces whatever is in the folder; --resume continues one instead
        for folder in (args.out, args.export_dir):
            if folder.exists():
                print(f"overwriting {folder}", flush=True)
                shutil.rmtree(folder)
        state = {"inputs": signature, "provenance": provenance(), "complete": False, "stages": {},
                 "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(path, state)
    command = [sys.executable, "-u", "-m", "world_model.vision.control_pipeline"]
    for name, value in signature["options"].items():
        if value is True:
            command += ["--"+name.replace("_", "-")]
        elif name == "methods":
            command += ["--methods", *str(value).split()]
        elif value is not None and value is not False:
            command += ["--"+name.replace("_", "-"), str(value)]
    rl_stage = "reuse_rl" if signature["frozen_baseline"] else "train_rl"
    jobs = [(rl_stage, command+["--stage", "rl"], args.out / "models/rl_training.json")]
    for method in args.methods:
        for i in range(args.test_episodes):
            seed = args.seed+20000+i
            jobs.append((f"{method}_{seed}", command+["--stage", "episode", "--method", method,
                "--episode-seed", str(seed)], args.out / "episodes" / method / f"seed_{seed}" / "result.json"))
    try:
        run_command([sys.executable, "-c", "import torch, stable_baselines3, transformers, mujoco; "
            "assert torch.cuda.is_available(), 'CUDA required'; "
            "print('GPU:', torch.cuda.get_device_name(0), 'SB3:', stable_baselines3.__version__)"], args.out / "logs/preflight.txt")
        for i, (name, cmd, artifact) in enumerate(jobs, 1):
            old = state["stages"].get(name, {})
            if old.get("complete"):
                if any(not pathlib.Path(p).is_file() or digest(p) != h for p, h in old["artifacts"].items()):
                    raise ValueError(f"completed stage changed: {name}")
                print(f"[{i}/{len(jobs)}] SKIP {name}", flush=True); continue
            print(f"[{i}/{len(jobs)}] START {name}", flush=True)
            start = time.monotonic()
            run_command(cmd, args.out / "logs" / f"{name}.txt")
            files = [artifact]
            if name in ("train_rl", "reuse_rl"):
                files.append(args.out / "models/sac_best.zip")
                if name == "reuse_rl":
                    files.append(args.out / "frozen_baseline.json")
            else:
                files += list(artifact.parent.glob("*.csv"))
            state["stages"][name] = {"complete": True, "seconds": time.monotonic()-start,
                                     "artifacts": {str(p): digest(p) for p in files}}
            write_json(path, state)
            export(args.out, args.export_dir, state)
        state["complete"] = True
        write_json(path, state)
    except BaseException as error:
        state["last_error"] = str(error); write_json(path, state)
        raise
    finally:
        export(args.out, args.export_dir, state)
    print(f"COMPLETE: shareable reports {args.export_dir}; actual videos/frames/forecasts {args.out / 'episodes'}", flush=True)


if __name__ == "__main__":
    main()
