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
from environment.rewards import fill_weights
from data_collection.scripted_policy import ScriptedPickPlace
from world_model.common import SIMULATION, load_run, save_csv, write_json, outcome_metrics
from rl.task_control import GoalReward, TaskSession, policy_observation, predicted_reward
from . import mpc, planner

# Run the controllers on fresh scenes of one layout and compare them:
#   scripted  the scripted policy on the exact simulator state, the upper bound
#   rl_true   the frozen SAC on the exact state (the actor-critic controller, one forward pass per step)
#   rl_q      the same SAC, but cube position and held come from Q on the live cameras
#   jepa_mpc  the latent planner (planner.py): Cartesian waypoint candidates imagined by D, scored by Q's
#             task potential and R's penalties, CEM, no policy anywhere in the loop
#   sac_imagined  a SAC trained only on transitions imagined by D (imagined_sac.py), run like rl_q on Q's estimate
#   mpc       the rule-free latent planner (mpc.py): CEM over raw action sequences imagined by D, scored by the
#             task reward on Q's readings plus R's and O's penalties; its knobs in controller/mpc_config.py
# The world model comes from --models-run (Q, R, D, the PCA bases and encoder settings), the SAC from
# --baseline-run (only loaded for rl_true / rl_q). Frames, gifs and per step csv land in
# data/<out>/episodes/<method>/seed_*/, the summary in results/<out>/.
#   python -m controller.control_pipeline --methods jepa_mpc
#   python -m controller.control_pipeline --test-episodes 20 --headless

METHODS = ("scripted", "rl_true", "rl_q", "jepa_mpc", "sac_imagined", "mpc")
CAMERAS = ("static", "wrist")


# The world model at run time: the online V-JEPA encoder (same weights, pooling, cameras and PCA bases as
# the cached features, plus the live height map), Q, D and the optional penalty head R, all from one run
# folder and tag
class WorldModels:

  def __init__(self, root, tag, args):
    import torch
    from world_model.train import load_model, model_folder
    from world_model.encode import Encoder, PINNED_REVISION
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if device is None:
      raise RuntimeError("live JEPA encoding needs CUDA (or Apple MPS)")
    self.torch, self.device = torch, device
    root = pathlib.Path(root)
    self.d, self.dc = load_model(root, "dynamics", tag)
    # blind = readout_p, the proprio only twin of Q, a control for what the image contributes
    self.q, self.qc = load_model(root, "readout_p" if args.blind else "readout", tag)
    self.r = self.rc = None
    if (model_folder(root, tag) / "models/reward.pt").exists() and not args.no_penalties:
      self.r, self.rc = load_model(root, "reward", tag)
      if "reach" not in self.rc.get("output_names", ()) and args.reward_source != "readout":
        raise ValueError("this reward.pt predicts penalties only; train the reward stage again or use --reward-source readout")
    # the small networks run on the GPU next to the encoder: batches of 64 candidates per imagined step
    for model in (self.d, self.q, self.r):
      if model is not None:
        model.to(device)
    self.penalty_scale = args.penalty_scale
    self.reward_source = "head" if args.no_goal else args.reward_source
    self.no_goal = args.no_goal
    if args.no_goal and self.r is None:
      raise ValueError("--no-goal needs the reward model R (its terms carry the task)")
    self.obstacle = None
    if args.obstacle_source == "model":
      from world_model.train_obstacle import load_obstacle_model
      self.obstacle, self.oc = load_obstacle_model(root, tag)
      self.obstacle.to(device)
    self.cost_to_go = CostToGoFields(root, tag) if getattr(args, "cost_to_go", False) else None
    self.width_gate = not args.no_width_gate
    self.grasp_prior = args.grasp_prior

    meta = json.loads((root / "features/meta.json").read_text())
    pooling = meta["pooling"].split("_pca")[0]
    self.cameras = tuple(meta.get("cameras", ["static"]))
    self.heightmap = bool(meta.get("heightmap", False))
    clip_frames = meta["clip_frames"] if isinstance(meta["clip_frames"], dict) else {"static": meta["clip_frames"]}
    self.encoder = Encoder(meta["model"], meta["model_revision"] or PINNED_REVISION, "mean_all" if pooling == "mean_all_encoder_tokens" else pooling,
                           cameras=self.cameras, clip_frames=clip_frames, stride=meta.get("frame_stride", 1),
                           dtype=meta.get("dtype", "fp16"), spatial_pool=meta.get("spatial_pool"),
                           heightmap=self.heightmap, heightmap_scale=meta.get("heightmap_scale", .25), device=device)
    if pooling != "mean_all_encoder_tokens":
      self.encoder.set_pca(np.load(root / "features/pca.npz"))

  # z from the frame histories (camera -> frames, newest last) and the live height map
  def encode(self, histories, heightmap):
    z = self.encoder.encode_frames(histories, heightmap if self.heightmap else None)
    if z.shape != (self.dc["z_dim"],) or not np.isfinite(z).all():
      raise ValueError("invalid latent from the online encoder")
    return z

  # D one step for a batch of (z, p, action), raw units in and out
  def predict(self, z, p, actions):
    from world_model.train import normalized
    with self.torch.inference_mode():
      zz, pp = self.d(normalized(z, self.dc, "z").to(self.device), normalized(p, self.dc, "p").to(self.device),
                      self.torch.as_tensor(actions, dtype=self.torch.float32, device=self.device))
    return (zz.cpu().numpy() * self.dc["z_std"].numpy() + self.dc["z_mean"].numpy(),
            pp.cpu().numpy() * self.dc["p_std"].numpy() + self.dc["p_mean"].numpy())

  # Q: cube xyz and held probability
  def read(self, z, p):
    from world_model.train import readout
    xyz, held = readout(self.q, self.qc, z, p)
    # fingers closed to under 2 cm hold nothing (the cube is 4.5 cm); without this Q kept saying held after a
    # missed grasp and the controller carried air to B in every failed episode
    if self.width_gate:
      p = np.asarray(p)
      held = held * (p[:, 18] > .02)
      # and fingers commanded shut that stopped at a cube's width (3.5-6.5 cm; the cubes are 4-5 cm, the grip
      # reads 4.2-5.4) hold one: Q's held reading lags a real grasp by ~12 steps because the latent is a
      # 16-frame time mean, and the planner stood still that long after every grasp
      held = np.where((p[:, 19] < 0) & (p[:, 18] > .035) & (p[:, 18] < .065), np.maximum(held, .99), held)
    # the old stop-gap on top: nothing is held unless the cube estimate sits between the fingers
    if self.grasp_prior:
      p = np.asarray(p)
      held = held * (np.linalg.norm(p[:, 14:16] - xyz[:, :2], axis=1) < .05) * (np.abs(p[:, 16] - xyz[:, 2]) < .06)
    return xyz, held

  # R: its predicted terms for a batch (name -> array), None without a reward model
  def reward(self, z, p):
    from world_model.train import reward_terms
    if self.r is None:
      return None
    return reward_terms(self.r, self.rc, z, p)

  # The weighted imagined penalty per candidate from R's terms, <= 0, zero without a reward model
  # The learned obstacle model's penalty, scaled like R's: proximity and contact from a window of the
  # obstacle-only depth map around the imagined hand (--obstacle-source model)
  def obstacle_penalty(self, obstacle_map, p, held, weights):
    from world_model.train_obstacle import obstacle_terms
    proximity, contact = obstacle_terms(self.obstacle, self.oc, obstacle_map, p, held)
    return (self.penalty_scale * (weights["proximity"] * proximity - weights["collision"] * contact)).astype(np.float32)

  def penalty(self, terms, weights):
    if terms is None:
      return 0.0
    # scaled above 1 the planner turns away earlier, before the forecast reaches the wall; a reward model
    # from before the empty-grasp penalty has no such term
    return (self.penalty_scale * (weights["proximity"] * terms["proximity"] - weights["collision"] * terms["collision"]
                                  - weights["table_hit"] * terms["table_hit"]
                                  - weights.get("empty_grasp", 0) * terms.get("empty_grasp", 0))).astype(np.float32)


# The learned cost-to-go (--cost-to-go): a field over the table per target, looked up at imagined positions
class CostToGoFields:

  def __init__(self, root, tag):
    from world_model.train_cost_to_go import load_cost_to_go
    # stays on the CPU: 6 ms per field there, while a tiny convolution queued behind D on a busy GPU took 500 ms
    self.model, self.ck = load_cost_to_go(root, tag)

  def field(self, obstacle_map, target):
    from world_model.train_cost_to_go import cost_field
    return cost_field(self.model, self.ck, obstacle_map, target)

  def lookup(self, field, points):
    from world_model.train_cost_to_go import lookup
    return lookup(field, points, self.ck["centre"], self.ck["half"])


# The frozen SAC and what it expects to see: a policy trained with --obstacles reads the height map too
class Policy:

  def __init__(self, baseline_run):
    from stable_baselines3 import SAC
    baseline_run = pathlib.Path(baseline_run)
    self.sac = SAC.load(baseline_run / "models/sac_best.zip", device="cpu")
    options = json.loads((baseline_run / "pipeline.json").read_text())["inputs"]["options"]
    self.heightmap = options.get("obstacles") is not None

  def act(self, observation):
    return self.sac.predict(observation, deterministic=True)[0]


# Save both camera frames as jpeg and keep the decoded jpegs in the histories (what the encoder sees)
def save_frames(session, folder, serial, histories):
  for camera in CAMERAS:
    path = folder / "frames" / f"{camera}_{serial:04d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = session.sim.render(camera)
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, session.cfg.data.jpeg_quality]):
      raise OSError(f"failed to save {path}")
    histories[camera].append(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    del histories[camera][:-64]


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
  visual = method in ("rl_q", "jepa_mpc", "sac_imagined", "mpc")
  session = TaskSession(cfg, layout, args.position_jitter, heightmap=policy is not None and policy.heightmap)
  goal = np.array([*layout["place"], rest_z], np.float32)
  memory = GoalReward(cfg)
  task_memory = planner.TaskMemory(args.cube_anchor)
  histories = {camera: [] for camera in CAMERAS}
  rows, candidates = [], []
  states_p, states_xyz, states_held, states_z, executed = [], [], [], [], []
  # the full simulator state per step (controller/replay.py redraws the episode from it) and mpc's imagined plans
  states_qpos, plans = [], []
  settings = mpc.load_settings(args.mpc_config) if method == "mpc" else None
  if settings is not None:
    mpc.save_settings(folder, settings)
  previous = None
  started = time.monotonic()
  viewer = None

  # z, Q's reading and the live height map of the current frame
  def perceive(p):
    heightmap = session.sim.height_map()
    z = models.encode(histories, heightmap)
    xyz, held = models.read(z[None], p[None])
    return z, xyz[0], float(held[0]), heightmap

  try:
    obs = session.reset(seed)
    viewer = LiveViewer(session, args, cfg)
    step_time = 1 / (cfg.control.hz * max(args.speed, 1e-3))
    scripted = ScriptedPickPlace(session.sim, np.random.default_rng(seed)) if method == "scripted" else None
    save_frames(session, folder, 0, histories)
    states_p.append(session.obs["proprio"].copy())
    states_xyz.append(session.obs["state"][:3].copy())
    states_held.append(bool(session.sim._check_contacts()[0]))
    states_qpos.append(session.sim.data.qpos.copy())
    if visual:
      z, xyz, held, heightmap = perceive(session.obs["proprio"])
      states_z.append(z.copy())
      task_memory.update(xyz, held, rest_z, session.obs["proprio"][14:17])
      # the obstacles as the depth camera measures them, for the planner's geometric clearance check
      task_memory.obstacles = planner.ObstacleMap(heightmap, cfg, session.obs["proprio"][14:17], xyz[:2])
      if models.obstacle is not None:
        # the obstacle-only map for the learned obstacle model; obstacles do not move, so the running maximum
        # keeps every cell the arm hides later
        task_memory.obstacle_map = session.sim.obstacle_map()

    for step in range(cfg.episode.max_steps):
      p = session.obs["proprio"].copy()
      decision = None
      decision_start = time.monotonic()
      if method == "scripted":
        action = scripted.act()
      elif method == "rl_true":
        action = policy.act(obs)
      elif method in ("rl_q", "sac_imagined"):
        estimate = policy_observation(p, xyz, held, memory, goal, rest_z, 1 - step / cfg.episode.max_steps,
                                      heightmap if policy.heightmap else None)
        action = policy.act(estimate)
      elif method == "mpc":
        decision = mpc.plan_standalone(models, z, p, memory, goal, rest_z, step, cfg, settings,
                                       np.random.default_rng(seed * 1000 + step), previous, task_memory.obstacle_map)
        action, previous = decision["action"], decision["sequence"]
        plans.append(decision["overlay"])
      else:
        decision = planner.plan(models, z, p, xyz, held, task_memory, goal, rest_z, cfg, args,
                                np.random.default_rng(seed * 1000 + step), previous)
        action, previous = decision["action"], decision["plan"]
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
      save_frames(session, folder, step + 1, histories)

      row = {"step": step + 1, "time_s": float(session.sim.data.time), "true_original_reward": info["original_reward"],
             "control_reward": reward, "actual_held": info["held_endpoint"],
             "actual_goal_distance_cm": float(100 * np.linalg.norm(actual_xyz[:2] - goal[:2])),
             "actual_lift_cm": float(100 * (actual_xyz[2] - session.rest_z)), "task_success": info["task_success"],
             "table_contact": info["table_contact"], "obstacle_contact": info["obstacle_contact"],
             "decision_seconds": decision_seconds}
      row.update({f"action_{k}": float(v) for k, v in zip(("dx", "dy", "dz", "dyaw", "gripper"), action)})
      row.update({f"actual_object_{k}": float(v) for k, v in zip("xyz", actual_xyz)})

      if visual:
        # what Q reads from the new frames, against the truth, and what the planner expected
        next_z, next_xyz, next_held, heightmap = perceive(actual_p)
        task_memory.obstacles.update(heightmap, actual_p[14:17], next_xyz[:2])
        if models.obstacle is not None:
          task_memory.obstacle_map = np.maximum(task_memory.obstacle_map, session.sim.obstacle_map())
        states_z.append(next_z.copy())
        row.update({"Q_held_probability": next_held, "Q_xyz_mae_cm": float(100 * np.abs(next_xyz - actual_xyz).mean())})
        row.update({f"Q_{k}": float(v) for k, v in zip("xyz", next_xyz)})
        # the task memory (was held, ever lifted, the SAC's reward memory) is advanced from Q's estimates,
        # never from the true state
        task_memory.update(next_xyz, next_held, rest_z, actual_p[14:17])
        failed = next_xyz[2] < cfg.table.height - cfg.episode.fall_margin or step + 1 >= cfg.episode.max_steps
        row["Q_estimated_reward"], _ = predicted_reward(memory, actual_p, next_xyz, next_held, action, goal, rest_z, failed)
        if decision is not None and method == "mpc":
          fields, rows_of_candidates = mpc.step_record(decision, step, actual_xyz, actual_p)
          row.update(fields)
          candidates.extend(rows_of_candidates)
        elif decision is not None:
          pool, selected = decision["pool"], decision["selected"]
          row.update({"predicted_score": decision["score"], "predicted_penalty_sum": float(pool["penalties"][selected].sum()),
                      "predicted_potential": float(pool["potentials"][selected, -1]), "planner_fallback": decision["fallback"],
                      "chosen": decision["proposal"], "valid_candidates": int(pool["valid"].sum()),
                      "clearance_max_cm": float(pool["clearance"][selected].max() * 100),
                      "grasp_clearance_max_cm": float(pool["clearance"][planner.PROPOSALS.index("grasp")].max() * 100),
                      "obstacle_cells": int((task_memory.obstacle_map > .01).sum()) if task_memory.obstacle_map is not None
                      else int(len(task_memory.obstacles.occupied)) if task_memory.obstacles is not None else 0})
          row.update({f"score_{name}": float(v) for name, v in zip(planner.PROPOSALS, decision["proposal_scores"])})
          if not decision["fallback"]:
            row.update({"D_Q_one_step_xyz_mae_cm": float(100 * np.abs(pool["xyz"][selected, 0] - actual_xyz).mean()),
                        "waypoint_x": float(decision["waypoints"][selected, 0]), "waypoint_y": float(decision["waypoints"][selected, 1]),
                        "waypoint_z": float(decision["waypoints"][selected, 2]), "switch_step": int(decision["switch"][selected]),
                        "grip_after": float(decision["grip_after"][selected])})
          for i in range(len(pool["scores"])):
            candidates.append({"step": step + 1, "candidate": i, "proposal": planner.PROPOSALS[i] if i < len(planner.PROPOSALS) else "sampled",
                               "selected": i == selected, "valid": bool(pool["valid"][i]), "score": float(pool["scores"][i]),
                               "terminal_potential": float(pool["potentials"][i, -1]), "penalty_sum": float(pool["penalties"][i].sum()),
                               "waypoint_x": float(decision["waypoints"][i, 0]), "waypoint_y": float(decision["waypoints"][i, 1]),
                               "waypoint_z": float(decision["waypoints"][i, 2]), "switch": int(decision["switch"][i]),
                               "grip_after": float(decision["grip_after"][i])})
        z, xyz, held = next_z, next_xyz, next_held

      rows.append(row)
      if (step + 1) % 10 == 0 or done or timeout:
        save_csv(folder / "steps.csv", rows)
        print(f"{method} seed={seed} step={step + 1}: held={info['held_endpoint']} "
              f"goal_distance={row['actual_goal_distance_cm']:.1f}cm lift={row['actual_lift_cm']:.1f}cm "
              f"success={info['task_success']}" + (f" chose={row['chosen']}" if "chosen" in row else ""), flush=True)
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
    images = [Image.open(f).convert("RGB").resize((256, 256)) for f in sorted((folder / "frames").glob("static_*.jpg"))[::2]]
    images[0].save(folder / "actual.gif", save_all=True, append_images=images[1:], duration=200, loop=0)

    obstacle_cells = ([[round(float(v), 3) for v in task_memory.obstacles.centres[c]] + [round(float(task_memory.obstacles.heights[c]), 3)]
                       for c in task_memory.obstacles.occupied] if task_memory.obstacles is not None else [])
    result = {"method": method, "seed": seed, **session.result(), "seconds": time.monotonic() - started, "layout": layout,
              "obstacle_cells": obstacle_cells,
              "planner_fallback_steps": sum(bool(r.get("planner_fallback")) for r in rows),
              "mean_decision_seconds": float(np.mean([r["decision_seconds"] for r in rows]))}
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
# reference height). train.py writes both to attempts/<tag>/info.json, so a packaged model folder (models/v3c:
# models/, info.json, features/) can run without the manifest; without that file they come from the full manifest.
def run_settings(models_run, tag):
  from world_model.train import model_folder
  info = model_folder(models_run, tag) / "info.json"
  if info.exists():
    info = json.loads(info.read_text())
    cfg, rest_z = Config.nested(info["config"]), float(info["rest_z"])
  else:
    manifest = load_run(models_run)
    cfg = Config.nested(copy.deepcopy(manifest["config"]))
    rest = [s["object_xyz"][2] for s in manifest["states"]
            if s["split"] == "train" and not s["held"] and abs(s["object_xyz"][2] - cfg.table.height) < .06]
    rest_z = float(np.median(rest))
  # the session ends episodes itself; reward weights added after the models' data was recorded come from default.yml
  cfg.episode.terminate_on_success = False
  fill_weights(cfg.rewards.weights)
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
                    "planner_fallback_steps": sum(r["planner_fallback_steps"] for r in rows),
                    "mean_decision_seconds": float(np.mean([r["mean_decision_seconds"] for r in rows]))})
    lines.append(f"{method}: placements {summary[-1]['successes']}/{len(rows)}; "
                 f"final goal distance {summary[-1]['mean_final_goal_distance_cm']:.2f} cm; "
                 f"obstacle contacts {summary[-1]['obstacle_contact_steps']}; "
                 f"decision {1000 * summary[-1]['mean_decision_seconds']:.0f} ms")
  target.mkdir(parents=True, exist_ok=True)
  write_json(target / "summary.json", summary)
  if summary:
    save_csv(target / "summary.csv", summary)
  (target / "summary.txt").write_text("\n".join(lines) + "\n")
  print("\n".join(lines), flush=True)


# The run flags shared with control_clutter, the planner's own are in planner.arguments
def run_arguments(p):
  planner.arguments(p)
  p.add_argument("--headless", action="store_true", help="no MuJoCo window (also the default without a display)")
  p.add_argument("--speed", type=float, default=1.0, help="playback speed of the window, times real time")
  p.add_argument("--threads", type=int, default=4)
  p.add_argument("--mpc-config", type=pathlib.Path, default=None, help="tuning file of the mpc planner (default controller/mpc_config.py)")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("models/v3c"),
                 help="a packaged model folder (models/, info.json, features/) or a run folder with attempts/<tag>/ (then give --tag)")
  p.add_argument("--tag", default="", help="attempts/<tag> inside a run folder; empty for a packaged model folder")
  p.add_argument("--baseline-run", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"),
                 help="rl_baseline run with the frozen SAC in models/sac_best.zip, for rl_true / rl_q")
  p.add_argument("--imagined-run", type=pathlib.Path, default=pathlib.Path("data/sac_imagined_v1"),
                 help="imagined_sac run with models/sac_best.zip, for sac_imagined")
  p.add_argument("--layout", type=pathlib.Path, default=pathlib.Path("configs/grade_e_layout.json"))
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/control"),
                 help="run folder for frames, gifs and csv; overwritten unless --resume")
  p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
  p.add_argument("--test-episodes", type=int, default=5, help="fresh scenes per method")
  p.add_argument("--seed", type=int, default=20474010)
  p.add_argument("--position-jitter", type=float, default=.01, help="cube start jitter per xy axis in metres")
  p.add_argument("--obstacles", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"),
                 help="draw this many fresh corridor obstacles per scene (control_clutter replays recorded ones)")
  p.add_argument("--random-ends", action="store_true",
                 help="cube and goal anywhere on the reachable table per scene (the environment's own sampler) instead of the layout file's, for the planner only")
  p.add_argument("--resume", action="store_true", help="keep finished episodes in --out and run the missing ones")
  run_arguments(p)
  args = p.parse_args()
  if not 2 <= args.elites < args.population or args.population <= len(planner.PROPOSALS) or min(args.horizon, args.iterations, args.test_episodes) < 1:
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
  torch.set_num_threads(args.threads)
  cfg, rest_z = run_settings(args.models_run, args.tag)
  layout = json.loads(args.layout.read_text())
  policy = Policy(args.baseline_run) if any(m in ("rl_true", "rl_q") for m in args.methods) else None
  imagined = Policy(args.imagined_run) if "sac_imagined" in args.methods else None
  models = WorldModels(args.models_run, args.tag, args) if any(m in ("rl_q", "jepa_mpc", "sac_imagined", "mpc") for m in args.methods) else None
  write_json(args.out / "settings.json", {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()})

  for method in args.methods:
    for i in range(args.test_episodes):
      seed = args.seed + 20000 + i
      if args.resume and (args.out / "episodes" / method / f"seed_{seed}" / "result.json").exists():
        continue
      scene = scene_layout(layout, cfg, seed, args.obstacles, args.random_ends)
      episode(args.out, method, seed, cfg, scene, rest_z, args, imagined if method == "sac_imagined" else policy, models)
      summarize(args.out, export_dir, args.methods)
  summarize(args.out, export_dir, args.methods)
  print(f"DONE: summary in {export_dir}, frames and gifs in {args.out / 'episodes'}", flush=True)


# The layout of one scene. With --random-ends the WHOLE scene comes from the environment's own sampler, the
# way the training episodes were made: cube and goal anywhere on the reachable table, the cube's size, mass,
# friction and colour drawn the same way (the layout file's cube is teal and of one size, the data's cubes are
# grey and vary), obstacles per --obstacles (none by default). Without it the file's layout is used, with
# fresh corridor obstacles when --obstacles is given.
def scene_layout(layout, cfg, seed, obstacles, random_ends=False):
  if obstacles is None and not random_ends:
    return layout
  from environment.obstacles import sample_obstacles
  from environment.scene import sample_layout
  rng = np.random.default_rng(seed)
  cfg = copy.deepcopy(cfg)
  if random_ends:
    cfg.task.pick_end = "anywhere"
    cfg.task.obstacles.count = list(obstacles) if obstacles is not None else [0, 0]
    return sample_layout(cfg, rng).to_dict()
  scene = copy.deepcopy(layout)
  cfg.task.obstacles.count = list(obstacles)
  drawn = sample_obstacles(cfg, rng, np.asarray(scene["pick"]), np.asarray(scene["place"]))
  scene["obstacles"] = [o.to_dict() for o in drawn] if drawn else []
  return scene


if __name__ == "__main__":
  main()
