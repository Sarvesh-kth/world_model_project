# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import os
import copy
import json
import pathlib
import shutil
import time

import cv2
import numpy as np

from environment.config import Config
from data_collection.scripted_policy import ScriptedPickPlace
from world_model.common import SIMULATION, load_run, save_csv, write_json, outcome_metrics
from rl.task_control import GoalReward, TaskSession, policy_observation, predicted_reward
from . import mpc

# Run the four controllers on fresh empty-table scenes and compare them:
#   scripted  the scripted policy on the exact simulator state, the upper bound
#   rl_true   the frozen SAC on the exact state
#   rl_q      the same SAC, but cube position and held come from Q on the live camera
#   jepa_mpc  the planner: CEM around the SAC's own action sequence, candidates imagined by D for --horizon
#             steps, scored by Q (progress reward) and the penalty head R; the SAC guide is candidate 0 and a
#             candidate only replaces it when it beats the guide's forecast by --guide-margin
#   mpc       the standalone planner (controller/mpc.py): the same imagined scoring as jepa_mpc but no SAC at
#             all, the CEM starts from its own previous plan; tuning knobs in controller/mpc_config.py
# Everything is loaded from two run folders: --models-run (Q, R, D, the PCA basis and encoder settings) and
# --baseline-run (the SAC). Frames, gifs and per step csv land in data/<out>/episodes/<method>/seed_*/,
# the summary in results/<out>/.
#   python -m controller.control_pipeline --methods jepa_mpc
#   python -m controller.control_pipeline --test-episodes 20 --headless

METHODS = ("scripted", "rl_true", "rl_q", "jepa_mpc", "mpc")
# the controllers that see the cube only through the camera (they need WorldModels and CUDA)
CAMERA_METHODS = ("rl_q", "jepa_mpc", "mpc")


# The world model at run time: the online V-JEPA encoder (same weights, pooling and PCA basis as the cached
# features), Q, D and the optional penalty head R, all from one run folder and tag
class WorldModels:

  def __init__(self, root, tag, args):
    import torch
    from world_model.train import load_model
    from world_model.encode import Encoder, PINNED_REVISION
    if not torch.cuda.is_available():
      raise RuntimeError("live JEPA encoding needs CUDA")
    self.torch = torch
    root = pathlib.Path(root)
    self.d, self.dc = load_model(root, "dynamics", tag)
    # blind = readout_p, the proprio only twin of Q, a control for what the image contributes
    self.q, self.qc = load_model(root, "readout_p" if args.blind else "readout", tag)
    self.r = self.rc = None
    if (root / "attempts" / tag / "models/reward.pt").exists() and not args.no_penalties:
      self.r, self.rc = load_model(root, "reward", tag)
    self.penalty_scale = args.penalty_scale
    self.width_gate = not args.no_width_gate

    meta = json.loads((root / "features/meta.json").read_text())
    pooling = meta["pooling"].split("_pca")[0]
    pca = np.load(root / "features/pca.npz") if pooling != "mean_all_encoder_tokens" else None
    self.encoder = Encoder(meta["model"], meta["model_revision"] or PINNED_REVISION, "mean_all" if pca is None else pooling,
                           pca=pca, clip_frames=meta["clip_frames"], stride=meta.get("frame_stride", 1),
                           dtype=meta.get("dtype", "bf16"), spatial_pool=meta.get("spatial_pool", 1))

  # z from the frame history (newest last)
  def encode(self, frames):
    z = self.encoder.encode_frames(frames)
    if z.shape != (self.dc["z_dim"],) or not np.isfinite(z).all():
      raise ValueError("invalid latent from the online encoder")
    return z

  # D one step for a batch of (z, p, action), raw units in and out
  def predict(self, z, p, actions):
    from world_model.train import normalized
    with self.torch.inference_mode():
      zz, pp = self.d(normalized(z, self.dc, "z"), normalized(p, self.dc, "p"),
                      self.torch.as_tensor(actions, dtype=self.torch.float32))
    return (zz.numpy() * self.dc["z_std"].numpy() + self.dc["z_mean"].numpy(),
            pp.numpy() * self.dc["p_std"].numpy() + self.dc["p_mean"].numpy())

  # Q: cube xyz and held probability
  def read(self, z, p):
    from world_model.train import readout
    xyz, held = readout(self.q, self.qc, z, p)
    # fingers closed to under 2 cm hold nothing (the cube is 4.5 cm); without this Q kept saying held after a
    # missed grasp and the policy carried air to B in every failed episode
    if self.width_gate:
      held = held * (np.asarray(p)[:, 18] > .02)
    return xyz, held

  # R: weighted imagined penalty per candidate, <= 0, zero without a penalty head
  def penalty(self, z, p, weights):
    from world_model.train import normalized
    if self.r is None:
      return np.zeros(len(z), np.float32)
    with self.torch.inference_mode():
      y = self.r(self.torch.cat((normalized(z, self.rc, "z"), normalized(p, self.rc, "p")), -1)).numpy()
    proximity = np.clip(y[:, 0], -1, 0)
    collision, table = 1 / (1 + np.exp(-y[:, 1])), 1 / (1 + np.exp(-y[:, 2]))
    # scaled above 1 the planner leaves the guide earlier, before the forecast reaches the wall
    return (self.penalty_scale * (weights["proximity"] * proximity - weights["collision"] * collision
                                  - weights["table_hit"] * table)).astype(np.float32)


# The SAC critic's value of a batch of observations, the optional terminal value of a plan
def terminal_values(policy, observations):
  import torch
  with torch.inference_mode():
    obs, _ = policy.policy.obs_to_tensor(np.asarray(observations, np.float32))
    actions = policy.actor(obs, deterministic=True)
    values = torch.cat(policy.critic(obs, actions), dim=1).min(dim=1).values
  return values.cpu().numpy()


# Imagine every candidate action sequence with D, read the imagined states with Q and score them:
# discounted predicted reward plus imagined penalties, plus terminal_weight * critic at the end.
# Candidates whose imagination leaves the physical range are marked invalid.
def forecasts(models, policy, z, p, actions, memory, goal, rest_z, step, cfg, terminal_weight):
  n, horizon, _ = actions.shape
  zs, ps = [np.repeat(z[None], n, 0)], [np.repeat(p[None], n, 0)]
  xyz, held = models.read(zs[0], ps[0])
  xyzs, helds = [xyz], [held]
  memories = [copy.deepcopy(memory) for _ in range(n)]
  rewards = np.zeros((n, horizon), np.float32)
  penalties = np.zeros((n, horizon), np.float32)
  # the task potential of every imagined state (used by mpc's "progress" score)
  potentials = np.zeros((n, horizon), np.float32)
  valid = np.ones(n, bool)
  finished = np.zeros(n, bool)

  for t in range(horizon):
    zz, pp = models.predict(zs[-1], ps[-1], actions[:, t])
    finite = np.isfinite(zz).all(1) & np.isfinite(pp).all(1)
    zz[~finite], pp[~finite] = zs[-1][~finite], ps[-1][~finite]
    xyz, held = models.read(zz, pp)
    # the open gripper measures up to .083, D overshoots by a few mm during the carry which is fine
    valid &= (finite & np.isfinite(xyz).all(1) & np.isfinite(held)
              & (pp[:, 18] >= -.002) & (pp[:, 18] <= .09) & (np.abs(pp[:, 14:17]) < 5).all(1)
              & (np.abs(xyz) < 5).all(1) & (np.abs(pp) < 100).all(1) & (np.abs(zz) < 1e4).all(1))
    zs.append(zz)
    ps.append(pp)
    xyzs.append(xyz)
    helds.append(held)
    penalties[:, t] = models.penalty(zz, pp, cfg.rewards.weights)
    for i in range(n):
      if finished[i]:
        continue
      failed = xyz[i, 2] < cfg.table.height - cfg.episode.fall_margin or step + t + 1 >= cfg.episode.max_steps
      rewards[i, t], _ = predicted_reward(memories[i], pp[i], xyz[i], held[i], actions[i, t], goal, rest_z, failed)
      rewards[i, t] += penalties[i, t]
      potentials[i, t] = memories[i].potential
      finished[i] = memories[i].task_succeeded or failed

  continuation = np.zeros(n)
  if terminal_weight:
    remaining = max(0, 1 - (step + horizon) / cfg.episode.max_steps)
    observations = [policy_observation(ps[-1][i], xyzs[-1][i], helds[-1][i], memories[i], goal, rest_z, remaining)
                    if valid[i] else policy_observation(p, xyzs[0][i], helds[0][i], memory, goal, rest_z, 0)
                    for i in range(n)]
    continuation = terminal_values(policy, observations)
  continuation[finished | ~valid] = 0.0
  discounted = rewards @ np.power(policy.gamma, np.arange(horizon))
  scores = discounted + terminal_weight * policy.gamma ** horizon * continuation
  valid &= np.isfinite(scores)
  scores[~valid] = -1e9
  return {"actions": actions, "z": np.stack(zs, 1), "p": np.stack(ps, 1), "xyz": np.stack(xyzs, 1),
          "held": np.stack(helds, 1), "rewards": rewards, "penalties": penalties, "scores": scores,
          "valid": valid, "terminal_critic": continuation, "discounted_reward": discounted,
          "potentials": potentials}


# The guide: what the SAC would do over the horizon if Q's readings of D's imagination were the truth
def actor_sequence(models, policy, z, p, memory, goal, rest_z, step, cfg, horizon):
  memory = copy.deepcopy(memory)
  z, p = z.copy()[None], p.copy()[None]
  actions = []
  for t in range(horizon):
    xyz, held = models.read(z, p)
    remaining = max(0, 1 - (step + t) / cfg.episode.max_steps)
    action, _ = policy.predict(policy_observation(p[0], xyz[0], held[0], memory, goal, rest_z, remaining), deterministic=True)
    actions.append(action)
    z, p = models.predict(z, p, action[None])
    xyz, held = models.read(z, p)
    if not (np.isfinite(z).all() and np.isfinite(p).all() and np.isfinite(xyz).all() and np.isfinite(held).all()):
      actions.extend([action.copy()] * (horizon - t - 1))
      break
    predicted_reward(memory, p[0], xyz[0], held[0], action, goal, rest_z)
  return np.asarray(actions, np.float32)


# CEM around the guide. Returns the chosen first action, the chosen sequence and the bookkeeping for the csv.
def plan(models, policy, z, p, memory, goal, rest_z, step, cfg, args, rng, previous=None):
  horizon = min(args.horizon, cfg.episode.max_steps - step)
  guide = actor_sequence(models, policy, z, p, memory, goal, rest_z, step, cfg, horizon)

  # the gripper follows the guide unless --free-gripper: D and Q never saw fingers closed on nothing, so a
  # candidate that closes early looks like a grasp in imagination and gets rewarded for it
  gripper = np.where(guide[:, -1] > 0, 1, -1).astype(np.float32)
  mean, std = guide.copy(), np.full_like(guide, args.cem_std)
  std_floor = np.full(5, min(.1, args.cem_std), np.float32)
  if not args.free_gripper:
    std[:, -1] = std_floor[-1] = 0

  best = None
  for iteration in range(args.iterations):
    # part of each candidate's noise is shared over the horizon (--cem-smooth): independent per step noise
    # averages out to under 1 cm of lateral spread over 8 steps, so no candidate could ever go around anything
    noise = rng.normal(size=(args.population, horizon, 5))
    if args.cem_smooth > 0:
      noise = np.sqrt(args.cem_smooth) * rng.normal(size=(args.population, 1, 5)) + np.sqrt(1 - args.cem_smooth) * noise
    actions = np.clip(mean + std * noise, -1, 1).astype(np.float32)
    actions[..., -1] = gripper if not args.free_gripper else np.where(actions[..., -1] > 0, 1, -1)
    # candidate 0 is always the guide, candidate 1 the previous plan shifted by one step
    actions[0] = guide
    if previous is not None:
      actions[1] = np.concatenate((previous[1:], previous[-1:]))[:horizon]
      if not args.free_gripper:
        actions[1, :, -1] = gripper
    pool = forecasts(models, policy, z, p, actions, memory, goal, rest_z, step, cfg, args.terminal_weight)
    selected = int(np.argmax(pool["scores"]))
    if best is None or pool["scores"][selected] > best["pool"]["scores"][best["selected"]]:
      best = {"pool": pool, "selected": selected}
    elites = actions[np.argsort(pool["scores"])[-args.elites:]]
    mean, std = elites.mean(0), np.maximum(elites.std(0), std_floor)

  # the guide's forecast is scored with the same D / Q / R as everything else, so a candidate only replaces
  # it when the gap is bigger than the model noise
  pool, selected = best["pool"], best["selected"]
  best["fallback"] = not bool(pool["valid"][selected])
  best["guide_score"] = float(pool["scores"][0])
  best["best_score"] = float(pool["scores"][selected])
  best["kept_guide"] = best["fallback"] or (bool(pool["valid"][0]) and best["best_score"] - best["guide_score"] <= args.guide_margin)
  if best["kept_guide"]:
    selected = 0
  best["selected"] = selected
  best["action"] = guide[0] if best["fallback"] else pool["actions"][selected, 0]
  best["sequence"] = guide if best["fallback"] else pool["actions"][selected]
  return best


# Save the static camera frame as jpeg and keep the decoded jpeg in the history (what the encoder sees)
def save_frame(session, folder, serial, history):
  path = folder / "frames" / f"static_{serial:04d}.jpg"
  path.parent.mkdir(parents=True, exist_ok=True)
  frame = session.sim.render("static")
  if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, session.cfg.data.jpeg_quality]):
    raise OSError(f"failed to save {path}")
  history.append(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
  del history[:-64]


# Separate process that owns the MuJoCo window. The episode process renders its camera through EGL and a
# window cannot share that context, so this one builds the same scene and mirrors the joint positions it is sent.
def viewer_process(cfg, layout, queue):
  import mujoco
  import mujoco.viewer
  from environment import scene, EpisodeLayout
  model = scene.build_scene(Config.nested(cfg), EpisodeLayout.from_dict(layout))
  data = mujoco.MjData(model)
  previous = target = None
  arrived = time.monotonic()
  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      try:
        state = queue.get(timeout=1 / 60)
        if state is None:
          break
        previous, target, arrived = (target if target is not None else state[0]), state[0], time.monotonic()
        gap = state[1]
      except Exception:
        pass
      if target is None:
        continue
      # the planner delivers a few states a second, glide between the last two so the window moves smoothly
      alpha = min(1.0, (time.monotonic() - arrived) / max(gap, 1e-3))
      data.qpos[:] = previous + alpha * (target - previous)
      mujoco.mj_forward(model, data)
      viewer.sync()
      time.sleep(1 / 60)
  # the viewer thread can hang on a normal exit, leave the hard way
  os._exit(0)


# Mirror of the simulation in a MuJoCo window, unless --headless or there is no display
class LiveViewer:

  # the environment the window process gets is probed once (window.py), the same for every episode
  window_env = None

  def __init__(self, session, args, cfg):
    self.queue = self.process = None
    self.last = None
    if args.headless or not os.environ.get("DISPLAY"):
      return
    if LiveViewer.window_env is None:
      from window import glx_environment
      LiveViewer.window_env = glx_environment() or {}
      if not LiveViewer.window_env:
        print("no OpenGL window can be created on this display, running without the live window "
              "(see the README on the rendering backend)", flush=True)
    if not LiveViewer.window_env:
      return
    import multiprocessing
    context = multiprocessing.get_context("spawn")
    self.queue = context.Queue(maxsize=4)
    self.process = context.Process(target=viewer_process, daemon=True, args=(cfg, session.sim.layout.to_dict(), self.queue))
    # a spawned child inherits os.environ, so swap in the window environment just for the start
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(LiveViewer.window_env)
    try:
      self.process.start()
    finally:
      os.environ.clear()
      os.environ.update(saved)

  def sync(self, session):
    if self.process is None or not self.process.is_alive():
      return
    now = time.monotonic()
    gap, self.last = (now - self.last if self.last else .1), now
    try:
      self.queue.put_nowait((session.sim.data.qpos.copy(), gap))
    except Exception:
      pass

  def close(self):
    if self.process is not None and self.process.is_alive():
      try:
        self.queue.put(None, timeout=1)
      except Exception:
        pass
      self.process.join(timeout=3)
      if self.process.is_alive():
        self.process.terminate()


# One episode of one controller. Writes frames/, steps.csv, candidates.csv (planner), trajectory.npz,
# actual.gif and result.json into <root>/episodes/<method>/seed_<seed>/ and returns the result.
def episode(root, method, seed, cfg, layout, rest_z, args, policy=None, models=None):
  cfg = Config.nested(cfg)
  folder = root / "episodes" / method / f"seed_{seed}"
  if folder.exists():
    shutil.rmtree(folder)
  folder.mkdir(parents=True)
  visual = method in CAMERA_METHODS
  session = TaskSession(cfg, layout, args.position_jitter)
  goal = np.array([*layout["place"], rest_z], np.float32)
  memory = GoalReward(cfg)
  history, rows, candidates = [], [], []
  states_p, states_xyz, states_held, states_z, executed = [], [], [], [], []
  # the full simulator state per step, so controller/replay.py can redraw the episode exactly
  states_qpos, plans = [], []
  settings = mpc.load_settings(args.mpc_config) if method == "mpc" else None
  if settings is not None:
    mpc.save_settings(folder, settings)
  previous = None
  started = time.monotonic()
  viewer = None
  try:
    obs = session.reset(seed)
    viewer = LiveViewer(session, args, cfg)
    step_time = 1 / (cfg.control.hz * max(args.speed, 1e-3))
    scripted = ScriptedPickPlace(session.sim, np.random.default_rng(seed)) if method == "scripted" else None
    save_frame(session, folder, 0, history)
    states_p.append(session.obs["proprio"].copy())
    states_xyz.append(session.obs["state"][:3].copy())
    states_held.append(bool(session.sim._check_contacts()[0]))
    states_qpos.append(session.sim.data.qpos.copy())
    if visual:
      z = models.encode(history)
      states_z.append(z.copy())
      xyz, held = models.read(z[None], session.obs["proprio"][None])
      xyz, held = xyz[0], float(held[0])

    for step in range(cfg.episode.max_steps):
      p = session.obs["proprio"].copy()
      decision = None
      decision_start = time.monotonic()
      if method == "scripted":
        action = scripted.act()
      elif method == "rl_true":
        action, _ = policy.predict(obs, deterministic=True)
      elif method == "rl_q":
        estimate = policy_observation(p, xyz, held, memory, goal, rest_z, 1 - step / cfg.episode.max_steps)
        action, _ = policy.predict(estimate, deterministic=True)
      elif method == "mpc":
        decision = mpc.plan_standalone(models, z, p, memory, goal, rest_z, step, cfg, settings, np.random.default_rng(seed * 1000 + step), previous)
        action, previous = decision["action"], decision["sequence"]
        plans.append(decision["overlay"])
      else:
        decision = plan(models, policy, z, p, memory, goal, rest_z, step, cfg, args, np.random.default_rng(seed * 1000 + step), previous)
        action, previous = decision["action"], decision["sequence"]
      decision_seconds = time.monotonic() - decision_start

      obs, reward, done, timeout, info = session.step(action)
      if viewer.process is not None:
        viewer.sync(session)
        time.sleep(max(0, step_time - (time.monotonic() - decision_start)))
      actual_xyz, actual_p = session.obs["state"][:3].copy(), session.obs["proprio"].copy()
      states_p.append(actual_p)
      states_xyz.append(actual_xyz)
      states_held.append(info["held_endpoint"])
      states_qpos.append(session.sim.data.qpos.copy())
      executed.append(np.asarray(action).copy())
      save_frame(session, folder, step + 1, history)

      row = {"step": step + 1, "time_s": float(session.sim.data.time), "true_original_reward": info["original_reward"],
             "control_reward": reward, "actual_held": info["held_endpoint"],
             "actual_goal_distance_cm": float(100 * np.linalg.norm(actual_xyz[:2] - goal[:2])),
             "actual_lift_cm": float(100 * (actual_xyz[2] - session.rest_z)), "task_success": info["task_success"],
             "table_contact": info["table_contact"], "obstacle_contact": info["obstacle_contact"],
             "decision_seconds": decision_seconds}
      row.update({f"action_{k}": float(v) for k, v in zip(("dx", "dy", "dz", "dyaw", "gripper"), action)})
      row.update({f"actual_object_{k}": float(v) for k, v in zip("xyz", actual_xyz)})

      if visual:
        # what Q reads from the new frame, against the truth, and what the planner expected
        next_z = models.encode(history)
        states_z.append(next_z.copy())
        next_xyz, next_held = models.read(next_z[None], actual_p[None])
        next_xyz, next_held = next_xyz[0], float(next_held[0])
        row.update({"Q_held_probability": next_held, "Q_xyz_mae_cm": float(100 * np.abs(next_xyz - actual_xyz).mean())})
        row.update({f"Q_{k}": float(v) for k, v in zip("xyz", next_xyz)})
        # the reward memory (was grasped, ever lifted, settle count, potential) is part of the policy's
        # observation; for the visual controllers it is advanced from Q's estimates, never from the true state
        failed = next_xyz[2] < cfg.table.height - cfg.episode.fall_margin or step + 1 >= cfg.episode.max_steps
        row["Q_estimated_reward"], _ = predicted_reward(memory, actual_p, next_xyz, next_held, action, goal, rest_z, failed)
        if decision is not None:
          pool, selected = decision["pool"], decision["selected"]
          row.update({"predicted_score": float(pool["scores"][selected]),
                      "predicted_penalty_sum": float(pool["penalties"][selected].sum()),
                      "planner_fallback": decision["fallback"], "guide_score": decision["guide_score"],
                      "best_candidate_score": decision["best_score"], "kept_guide": decision["kept_guide"],
                      "valid_candidates": int(pool["valid"].sum())})
          if not decision["fallback"]:
            predicted_xyz = pool["xyz"][selected, 1]
            row.update({"D_Q_one_step_xyz_mae_cm": float(100 * np.abs(predicted_xyz - actual_xyz).mean()),
                        "D_one_step_latent_normalized_mse": float(np.square((pool["z"][selected, 1] - next_z) / models.dc["z_std"].numpy()).mean()),
                        "D_one_step_ee_mae_cm": float(100 * np.abs(pool["p"][selected, 1, 14:17] - actual_p[14:17]).mean())})
          for i in range(len(pool["scores"])):
            candidates.append({"step": step + 1, "candidate": i, "selected": i == selected, "valid": bool(pool["valid"][i]),
                               "score": float(pool["scores"][i]), "discounted_reward": float(pool["discounted_reward"][i]),
                               "penalty_sum": float(pool["penalties"][i].sum()), "terminal_critic": float(pool["terminal_critic"][i])})
        z, xyz, held = next_z, next_xyz, next_held

      rows.append(row)
      if (step + 1) % 10 == 0 or done or timeout:
        save_csv(folder / "steps.csv", rows)
        print(f"{method} seed={seed} step={step + 1}: held={info['held_endpoint']} "
              f"goal_distance={row['actual_goal_distance_cm']:.1f}cm lift={row['actual_lift_cm']:.1f}cm "
              f"success={info['task_success']}", flush=True)
      if done or timeout:
        break

    # outputs
    save_csv(folder / "steps.csv", rows)
    if candidates:
      save_csv(folder / "candidates.csv", candidates)
    arrays = {"p": np.asarray(states_p), "object_xyz": np.asarray(states_xyz), "held": np.asarray(states_held),
              "actions": np.asarray(executed), "qpos": np.asarray(states_qpos)}
    if visual:
      arrays["z"] = np.asarray(states_z)
    np.savez_compressed(folder / "trajectory.npz", **arrays)
    if plans:
      mpc.save_plans(folder, plans)
    from PIL import Image
    images = [Image.open(f).convert("RGB").resize((256, 256)) for f in sorted((folder / "frames").glob("*.jpg"))[::2]]
    images[0].save(folder / "actual.gif", save_all=True, append_images=images[1:], duration=200, loop=0)

    result = {"method": method, "seed": seed, **session.result(), "seconds": time.monotonic() - started,
              "planner_fallback_steps": sum(bool(r.get("planner_fallback")) for r in rows),
              "guide_kept_steps": sum(bool(r.get("kept_guide")) for r in rows)}
    if visual:
      result["Q_on_actual_observations"] = outcome_metrics([[r[f"Q_{k}"] for k in "xyz"] for r in rows],
                                                           [r["Q_held_probability"] for r in rows], states_xyz[1:], states_held[1:])
    write_json(folder / "result.json", result)
    print(f"FINISHED {method} seed={seed}: success={result['task_success']} goal={result['final_goal_distance_cm']:.1f}cm", flush=True)
    return result
  finally:
    if viewer is not None:
      viewer.close()
    session.close()


# The environment config the models were trained on and the resting cube height (the policy's and the planner's
# reference height). train.py writes both to attempts/<tag>/info.json, so a clone that has only the checkpoints
# and features/{pca.npz,meta.json} can run; without that file they come from the full manifest.
def run_settings(models_run, tag):
  info = pathlib.Path(models_run) / "attempts" / tag / "info.json"
  if info.exists():
    info = json.loads(info.read_text())
    cfg, rest_z = Config.nested(info["config"]), float(info["rest_z"])
  else:
    manifest = load_run(models_run)
    cfg = Config.nested(copy.deepcopy(manifest["config"]))
    rest = [s["object_xyz"][2] for s in manifest["states"]
            if s["split"] == "train" and not s["held"] and abs(s["object_xyz"][2] - cfg.table.height) < .06]
    rest_z = float(np.median(rest))
  # the session ends episodes itself
  cfg.episode.terminate_on_success = False
  return cfg, rest_z


# Encoder settings and placements per method into results/<name>/: summary.json / .csv / .txt plus a copy
# of every episode's result.json, steps.csv and candidates.csv
def summarize(root, target, methods):
  results = []
  for path in sorted((root / "episodes").glob("*/*/result.json")):
    results.append(json.loads(path.read_text()))
    for source in path.parent.glob("*"):
      if source.suffix in (".csv", ".json"):
        destination = target / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
  summary, lines = [], []
  for method in methods:
    rows = [r for r in results if r["method"] == method]
    if not rows:
      continue
    summary.append({"method": method, "episodes": len(rows), "successes": sum(r["task_success"] for r in rows),
                    "mean_final_goal_distance_cm": float(np.mean([r["final_goal_distance_cm"] for r in rows])),
                    "median_final_goal_distance_cm": float(np.median([r["final_goal_distance_cm"] for r in rows])),
                    "obstacle_contact_steps": sum(r["obstacle_contact_steps"] for r in rows),
                    "planner_fallback_steps": sum(r["planner_fallback_steps"] for r in rows)})
    lines.append(f"{method}: placements {summary[-1]['successes']}/{len(rows)}; "
                 f"final goal distance {summary[-1]['mean_final_goal_distance_cm']:.2f} cm; "
                 f"obstacle contacts {summary[-1]['obstacle_contact_steps']}")
  target.mkdir(parents=True, exist_ok=True)
  write_json(target / "summary.json", summary)
  if summary:
    save_csv(target / "summary.csv", summary)
  (target / "summary.txt").write_text("\n".join(lines) + "\n")
  print("\n".join(lines), flush=True)


# The planner flags, shared with control_clutter
def planner_arguments(p):
  p.add_argument("--horizon", type=int, default=8, help="steps D imagines ahead")
  p.add_argument("--population", type=int, default=64, help="CEM candidates per iteration")
  p.add_argument("--elites", type=int, default=8)
  p.add_argument("--iterations", type=int, default=3)
  p.add_argument("--cem-std", type=float, default=0.3, help="CEM noise around the SAC guide (the original used 0.6)")
  p.add_argument("--cem-smooth", type=float, default=0.5, help="share of the noise variance that is constant over the horizon")
  p.add_argument("--guide-margin", type=float, default=0.25, help="a candidate replaces the guide only if it wins by this much")
  p.add_argument("--terminal-weight", type=float, default=0.0, help="weight of the SAC critic at the end of the horizon (the original used 1)")
  p.add_argument("--penalty-scale", type=float, default=2.0, help="multiplier on R's imagined penalties")
  p.add_argument("--no-penalties", action="store_true", help="ignore the penalty head")
  p.add_argument("--free-gripper", action="store_true", help="let CEM sample the gripper too (the original behaviour)")
  p.add_argument("--no-width-gate", action="store_true", help="do not zero Q's held reading when the fingers are closed on nothing")
  p.add_argument("--blind", action="store_true", help="use the proprio-only readout_p instead of Q(z, p)")
  p.add_argument("--mpc-config", type=pathlib.Path, default=None, help="tuning file of the mpc planner (default controller/mpc_config.py)")
  p.add_argument("--headless", action="store_true", help="no MuJoCo window (also the default without a display)")
  p.add_argument("--speed", type=float, default=1.0, help="playback speed of the window, times real time")
  p.add_argument("--threads", type=int, default=4)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("data/combined_test1"),
                 help="run folder with features/ (PCA basis, encoder settings) and attempts/<tag>/models/ (Q, R, D)")
  p.add_argument("--tag", default="combined")
  p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"),
                 help="rl_baseline run with the frozen SAC in models/sac_best.zip")
  p.add_argument("--layout", type=pathlib.Path, default=pathlib.Path("configs/grade_e_layout.json"))
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/control"),
                 help="run folder for frames, gifs and csv; overwritten unless --resume")
  p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
  p.add_argument("--test-episodes", type=int, default=5, help="fresh scenes per method")
  p.add_argument("--seed", type=int, default=20474010)
  p.add_argument("--position-jitter", type=float, default=.01, help="cube start jitter per xy axis in metres")
  p.add_argument("--resume", action="store_true", help="keep finished episodes in --out and run the missing ones")
  planner_arguments(p)
  args = p.parse_args()
  if not 2 <= args.elites < args.population or not 0 <= args.cem_smooth <= 1 or min(args.horizon, args.iterations, args.test_episodes) < 1:
    p.error("invalid planner settings")
  args.out = args.out.resolve()
  if SIMULATION / "data" not in args.out.parents:
    p.error("keep --out under simulation/data/")
  export_dir = SIMULATION.parent / "results" / args.out.name
  if not args.resume:
    for folder in (args.out, export_dir):
      if folder.exists():
        print(f"overwriting {folder}", flush=True)
        shutil.rmtree(folder)
  args.out.mkdir(parents=True, exist_ok=True)

  # config and reference height from the run the models were trained on, the layout from the file
  import torch
  from stable_baselines3 import SAC
  torch.set_num_threads(args.threads)
  cfg, rest_z = run_settings(args.models_run, args.tag)
  layout = json.loads(args.layout.read_text())
  policy = SAC.load(args.baseline_run / "models/sac_best.zip", device="cpu")
  models = WorldModels(args.models_run, args.tag, args) if any(m in CAMERA_METHODS for m in args.methods) else None
  write_json(args.out / "settings.json", {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()})

  for method in args.methods:
    for i in range(args.test_episodes):
      seed = args.seed + 20000 + i
      if args.resume and (args.out / "episodes" / method / f"seed_{seed}" / "result.json").exists():
        continue
      episode(args.out, method, seed, cfg, layout, rest_z, args, policy, models)
      summarize(args.out, export_dir, args.methods)
  summarize(args.out, export_dir, args.methods)
  print(f"DONE: summary in {export_dir}, frames and gifs in {args.out / 'episodes'}", flush=True)


if __name__ == "__main__":
  main()
