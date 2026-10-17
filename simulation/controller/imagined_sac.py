# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import copy
import json
import pathlib
import shutil
import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from world_model.common import SIMULATION, load_run, write_json
from rl.task_control import GoalReward, TaskSession, policy_observation, predicted_reward
from .control_pipeline import WorldModels, run_settings, scene_layout

# M3's second controller: an actor-critic trained only in imagination (the proposal's "actor-critic trained with
# RL on imagined rollouts"). A standard SAC (stable-baselines3, the implementation the team's exact-state SAC uses)
# learns from transitions the world model makes up. An imagined episode starts at a recorded state (z, p) of the
# training data; every step the actor picks an action, D imagines the next (z, p), Q reads the cube position and
# held from the imagined latent, and the reward is the GoalReward the exact-state SAC was trained with, computed on
# Q's readings, plus R's predicted penalties. The simulator is never stepped to train.
# The actor sees the same 41 numbers as rl_q (policy_observation on Q's estimate), so the trained policy runs on the
# live cameras through the existing rl_q code path; the run folder has the layout of an rl_baseline run:
#   python -m controller.imagined_sac --out data/sac_imagined_v1
#   python -m controller.control_pipeline --methods rl_q --baseline-run data/sac_imagined_v1 --random-ends --test-episodes 20 --headless --out data/eval_sac_imagined_v1

# the task memory the controller keeps between steps (GoalReward's fields), copied into every imagined episode
MEMORY = ("placed", "succeeded", "was_grasped", "prev_grasped", "ever_lifted", "task_succeeded", "settled_steps", "potential")


# Q, R and D of a model folder without the V-JEPA encoder (training reads the recorded latents and encodes
# nothing), on any device. read() and penalty() are WorldModels' own, with the controllers' default gates.
class ImaginationModels(WorldModels):

  def __init__(self, root, tag, device, penalty_scale=1.0):
    import torch
    from world_model.train import load_model
    self.torch, self.device = torch, device
    self.d, self.dc = load_model(root, "dynamics", tag)
    self.q, self.qc = load_model(root, "readout", tag)
    self.r, self.rc = load_model(root, "reward", tag)
    for model in (self.d, self.q, self.r):
      model.to(device)
    self.penalty_scale = penalty_scale
    self.width_gate, self.grasp_prior = True, True

  # D one step for a batch of (z, p, action), raw units in and out (WorldModels.predict without the fixed CUDA)
  def predict(self, z, p, actions):
    from world_model.train import normalized
    torch = self.torch
    with torch.inference_mode():
      zz, pp = self.d(normalized(z, self.dc, "z").to(self.device), normalized(p, self.dc, "p").to(self.device),
                      torch.as_tensor(actions, dtype=torch.float32, device=self.device))
    return (zz.cpu().numpy() * self.dc["z_std"].numpy() + self.dc["z_mean"].numpy(),
            pp.cpu().numpy() * self.dc["p_std"].numpy() + self.dc["p_mean"].numpy())


# The recorded states imagined episodes start from. Every recorded episode is replayed through Q the way the
# controller would have watched it (Q's reading of each latent advances a GoalReward), so each state comes with
# Q's reading, the task memory the controller would have had there and the action the data took next. States after a finished
# placement are left out, and scenes with obstacles unless asked (the 41 inputs do not show obstacles).
def load_starts(run, splits, models, cfg, rest_z, obstacles=False):
  manifest = load_run(run)
  scenes = {s["rollout"]: s for s in json.loads((run / "scene_settings.json").read_text())}
  latents = np.load(run / "features/latents.npy", mmap_mode="r")
  episodes = {}
  for i, s in enumerate(manifest["states"]):
    if s["split"] in splits and (obstacles or not scenes[s["scene"]]["layout"].get("obstacles")):
      episodes.setdefault(s["scene"], []).append(i)
  rows, policies = {k: [] for k in ("index", "p", "xyz", "held", "goal", "step", "action", *MEMORY)}, []
  for scene, ids in episodes.items():
    ids = sorted(ids, key=lambda i: manifest["states"][i]["key"])
    states = [manifest["states"][i] for i in ids]
    p = np.asarray([s["p"] for s in states], np.float32)
    xyz, held = models.read(np.asarray(latents[ids]), p)
    goal = np.array([*scenes[scene]["layout"]["place"], rest_z], np.float32)
    memory = GoalReward(cfg)
    for k, (i, s) in enumerate(zip(ids, states)):
      previous = np.asarray(s.get("action_from_previous") or np.zeros(5), np.float32)
      predicted_reward(memory, p[k], xyz[k], held[k], previous, goal, rest_z)
      if memory.task_succeeded:
        break
      rows["index"].append(i)
      rows["p"].append(p[k])
      rows["xyz"].append(xyz[k])
      rows["held"].append(held[k])
      rows["goal"].append(goal)
      rows["step"].append(round(s["time"] * cfg.control.hz))
      last = k + 1 >= len(ids)
      rows["action"].append(np.full(5, np.nan, np.float32) if last else np.asarray(states[k + 1]["action_from_previous"], np.float32))
      for name in MEMORY:
        rows[name].append(getattr(memory, name))
      policies.append(scenes[scene].get("policy", ""))
  starts = {k: np.asarray(v) for k, v in rows.items()}
  starts["policy"] = np.asarray(policies)
  starts["latents"] = latents
  return starts


# n imagined episodes side by side, as a stable-baselines3 VecEnv: one batched D step per call for all of them.
# An episode starts at a random recorded state and runs `horizon` imagined steps (D was trained on 8-step
# windows and drifts beyond); the critic carries the value past the cut (a truncation, not an ending). It ends
# for real on a placement (GoalReward's 15-point bonus), the cube read below the table, or the 300-step deadline.
# An imagined state outside the physical range (non-finite, fingers wider than they open) is cut off like the
# horizon, without a reward for that step.
# The grasp check (the planner's gripper envelope, our mpc's GRASP_NEEDS_CONTACT): unless the cube is held at the
# start, an imagined hold only counts if the fingers closed with the hand within 1.5 cm sideways and 2 cm in
# height of where Q read the cube at the start. D imagines a hold for fingers closing mid-travel (no such closes
# in the data), and without the check the SAC learned closes that imagination rewarded and the simulator did not.
class ImaginedEnv(VecEnv):

  render_mode = None

  def __init__(self, models, starts, n, horizon, cfg, rest_z, seed, fixed=None, grasp_check=True):
    super().__init__(n, gym.spaces.Box(-np.inf, np.inf, (41,), np.float32), gym.spaces.Box(-1, 1, (5,), np.float32))
    self.models, self.starts, self.horizon, self.cfg, self.rest_z = models, starts, horizon, cfg, rest_z
    self.weights = cfg.rewards.weights
    self.rng = np.random.default_rng(seed)
    # fixed: start indices for a repeatable evaluation (episode j always starts at fixed[j])
    self.fixed = fixed
    self.z = np.zeros((n, starts["latents"].shape[1]), np.float32)
    self.p = np.zeros((n, 20), np.float32)
    self.goal = np.zeros((n, 3), np.float32)
    # the episode's step number (the recorded one plus the imagined steps, for the time left) and the imagined steps
    self.step_no = np.zeros(n, int)
    self.t = np.zeros(n, int)
    self.memory = [GoalReward(cfg) for _ in range(n)]
    self.xyz, self.held = np.zeros((n, 3), np.float32), np.zeros(n, np.float32)
    self.grasp_check = grasp_check
    self.cube0, self.grasp_ok, self.was_closed = np.zeros((n, 3), np.float32), np.zeros(n, bool), np.zeros(n, bool)
    self.actions = None

  def start(self, j, k):
    s = self.starts
    self.z[j] = s["latents"][s["index"][k]]
    self.p[j], self.xyz[j], self.held[j] = s["p"][k], s["xyz"][k], s["held"][k]
    self.goal[j], self.step_no[j], self.t[j] = s["goal"][k], s["step"][k], 0
    self.cube0[j], self.grasp_ok[j], self.was_closed[j] = s["xyz"][k], s["held"][k] >= .5, s["p"][k][19] < 0
    memory = self.memory[j]
    memory.reset()
    for name in MEMORY:
      setattr(memory, name, s[name][k].item())

  def observe(self, j):
    remaining = max(0.0, 1 - self.step_no[j] / self.cfg.episode.max_steps)
    return policy_observation(self.p[j], self.xyz[j], float(self.held[j]), self.memory[j], self.goal[j], self.rest_z, remaining)

  def reset(self):
    for j in range(self.num_envs):
      self.start(j, self.fixed[j] if self.fixed is not None else self.rng.integers(len(self.starts["index"])))
    return np.stack([self.observe(j) for j in range(self.num_envs)])

  def step_async(self, actions):
    self.actions = np.clip(np.asarray(actions, np.float32), -1, 1)

  def step_wait(self):
    n, cfg = self.num_envs, self.cfg
    closing = (self.actions[:, 4] < 0) & ~self.was_closed
    at_cube = ((np.linalg.norm(self.p[:, 14:16] - self.cube0[:, :2], axis=1) <= .015)
               & (np.abs(self.p[:, 16] - self.cube0[:, 2]) <= .02))
    self.grasp_ok |= closing & at_cube
    self.was_closed = self.actions[:, 4] < 0
    z, p = self.models.predict(self.z, self.p, self.actions)
    xyz, held = self.models.read(z, p)
    if self.grasp_check:
      held = held * self.grasp_ok
    penalties = self.models.penalty(self.models.reward(z, p), self.weights)
    valid = (np.isfinite(z).all(1) & np.isfinite(p).all(1) & np.isfinite(xyz).all(1) & (p[:, 18] >= -.02) & (p[:, 18] <= .12)
             & (np.abs(p[:, 14:17]) < 5).all(1) & (np.abs(xyz) < 5).all(1))
    rewards, dones, infos = np.zeros(n, np.float32), np.zeros(n, bool), [{} for _ in range(n)]
    observations = np.zeros((n, 41), np.float32)
    restart = []
    for j in range(n):
      if not valid[j]:
        dones[j] = True
        infos[j].update({"TimeLimit.truncated": True, "terminal_observation": self.observe(j), "invalid": True})
        restart.append(j)
        continue
      self.z[j], self.p[j], self.xyz[j], self.held[j] = z[j], p[j], xyz[j], held[j]
      self.t[j] += 1
      self.step_no[j] += 1
      fell = xyz[j, 2] < cfg.table.height - cfg.episode.fall_margin
      timeout = self.step_no[j] >= cfg.episode.max_steps
      reward, _ = predicted_reward(self.memory[j], p[j], xyz[j], held[j], self.actions[j], self.goal[j], self.rest_z, fell or timeout)
      rewards[j] = reward + penalties[j]
      observations[j] = self.observe(j)
      ended = self.memory[j].task_succeeded or fell or timeout
      if ended or self.t[j] >= self.horizon:
        dones[j] = True
        infos[j].update({"TimeLimit.truncated": not ended, "terminal_observation": observations[j].copy(),
                         "is_success": bool(self.memory[j].task_succeeded)})
        restart.append(j)
    for j in restart:
      self.start(j, self.rng.integers(len(self.starts["index"])) if self.fixed is None else self.fixed[j])
      observations[j] = self.observe(j)
    return observations, rewards, dones, infos

  def close(self):
    pass

  def get_attr(self, name, indices=None):
    return [getattr(self, name, None)] * self.num_envs

  def set_attr(self, name, value, indices=None):
    setattr(self, name, value)

  def env_method(self, name, *args, indices=None, **kwargs):
    return [None] * self.num_envs

  def env_is_wrapped(self, wrapper_class, indices=None):
    return [False] * self.num_envs


# The deterministic actor from fixed held-out starts for the validation horizon: how many episodes ended in an
# imagined placement and the mean return. Picks the checkpoint without touching the simulator.
def imagined_validation(model, env):
  observations, total = env.reset(), np.zeros(env.num_envs)
  alive, placed = np.ones(env.num_envs, bool), np.zeros(env.num_envs, bool)
  for _ in range(env.horizon):
    actions = model.predict(observations, deterministic=True)[0]
    observations, rewards, dones, infos = env.step(actions)
    total += alive * rewards
    placed |= alive & np.array([bool(i.get("is_success")) for i in infos])
    alive &= ~dones
  return {"imagined_placed": int(placed.sum()), "imagined_return": float(total.mean()), "starts": env.num_envs}


# A diagnostic only, never used to pick the checkpoint: the actor on the EXACT simulator state (as rl_true) in
# fresh random-end scenes. It separates "the policy learned the task" from "Q reads the camera well enough",
# which the camera evaluation mixes.
def exact_state_check(model, cfg, seeds):
  successes, distances = 0, []
  for seed in seeds:
    session = TaskSession(cfg, scene_layout(None, cfg, seed, None, random_ends=True))
    try:
      observation = session.reset(seed)
      for _ in range(cfg.episode.max_steps):
        observation, _, done, _, _ = session.step(model.predict(observation, deterministic=True)[0])
        if done:
          break
      result = session.result()
      successes += result["task_success"]
      distances.append(result["final_goal_distance_cm"])
    finally:
      session.close()
  return {"exact_state_placed": successes, "exact_state_episodes": len(seeds), "exact_state_median_cm": float(np.median(distances))}


# The demonstrations: the recorded clean episodes of the training scenes (the scripted successes in the data),
# each state as the controller would have seen it through Q, with the action the data took next
def demonstrations(env, starts):
  clean = np.flatnonzero(np.isin(starts["policy"], ("success", "scripted")) & np.isfinite(starts["action"]).all(1))
  observations = []
  for k in clean:
    env.start(0, k)
    observations.append(env.observe(0))
  return np.stack(observations), starts["action"][clean]


# Behaviour cloning of the actor on the demonstrations, the start for the SAC as in rl_control: the actor mean is
# fitted to the recorded actions, the gripper counting double
def clone(model, x, y, epochs, seed):
  import torch
  optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
  rng = np.random.default_rng(seed)
  for epoch in range(epochs):
    loss_sum = 0.0
    for ids in np.array_split(rng.permutation(len(x)), max(1, len(x) // 256)):
      error = (model.actor(x[ids], deterministic=True) - y[ids]).square()
      loss = error[:, :4].mean() + 2 * error[:, 4].mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_sum += float(loss.detach()) * len(ids)
    if (epoch + 1) % 40 == 0 or epoch + 1 == epochs:
      print(f"BC epoch {epoch + 1}: action MSE={loss_sum / len(x):.5f} on {len(x)} recorded states", flush=True)
  # explore with a small spread around the cloned actions
  with torch.no_grad():
    model.actor.log_std.weight.zero_()
    model.actor.log_std.bias.fill_(-2)


# Imagined transitions of the cloned actor (with its small exploration noise) into the replay buffer, so the
# critic warmup has the actor's own states to learn from; the stored next observation of an ended episode is its
# last one, as stable-baselines3 stores it
def prefill(model, env, steps):
  observations = env.reset()
  for _ in range(steps):
    actions = model.predict(observations, deterministic=False)[0]
    following, rewards, dones, infos = env.step(actions)
    stored = following.copy()
    for j in np.flatnonzero(dones):
      stored[j] = infos[j]["terminal_observation"]
    model.replay_buffer.add(observations, stored, actions, rewards, dones, infos)
    observations = following


def arguments():
  p = argparse.ArgumentParser(description="SAC trained only on transitions imagined by the world model")
  p.add_argument("--run", type=pathlib.Path, default=pathlib.Path("data/episodes_v3"),
                 help="the run folder with manifest.json, scene_settings.json and features/latents.npy (the starts)")
  p.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("models/v3c"))
  p.add_argument("--tag", default="")
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/sac_imagined_v1"))
  p.add_argument("--initialize-run", type=pathlib.Path, default=None,
                 help="continue the saved imagined SAC weights/optimizers, with fresh replay; skip BC reinitialization")
  p.add_argument("--steps", type=int, default=200000, help="imagined transitions for the SAC")
  p.add_argument("--envs", type=int, default=64, help="imagined episodes side by side (one batched D step)")
  p.add_argument("--horizon", type=int, default=10, help="imagined steps per training episode before the cut")
  p.add_argument("--updates-per-step", type=float, default=0.25, help="SAC gradient updates per imagined transition")
  # the exact-state SAC's recipe (rl/rl_baseline.py), so the two differ only in where the transitions come from
  p.add_argument("--bc-epochs", type=int, default=240, help="behaviour cloning on the recorded clean episodes first; 0 = none")
  p.add_argument("--bc-weight", type=float, default=100, help="weight of the cloning term in every actor update; 0 = pure SAC")
  p.add_argument("--td3bc-alpha", type=float, default=0.0,
                 help="TD3+BC: the cloning weight follows the critic's scale, mean|Q| / alpha (2.5 in the paper); 0 = --bc-weight fixed")
  p.add_argument("--critic-warmup", type=int, default=2000, help="critic-only updates on the cloned actor's imagined transitions")
  p.add_argument("--prefill", type=int, default=100, help="imagined steps (of all envs) of the cloned actor before the warmup")
  p.add_argument("--learning-rate", type=float, default=3e-5)
  p.add_argument("--ent-coef", default="auto_0.005")
  p.add_argument("--penalty-scale", type=float, default=1.0, help="multiplier on R's penalties (the exact-state SAC's weights at 1)")
  p.add_argument("--obstacles", action="store_true", help="also start in scenes with obstacles (the 41 inputs do not show them)")
  p.add_argument("--no-grasp-check", dest="grasp_check", action="store_false", help="let any imagined hold count (see ImaginedEnv)")
  p.add_argument("--eval-every", type=int, default=20000, help="imagined transitions between checkpoints")
  p.add_argument("--validation-starts", type=int, default=256)
  p.add_argument("--validation-horizon", type=int, default=30, help="imagined steps per validation episode")
  p.add_argument("--exact-checks", type=int, default=20, help="exact-state simulator episodes per checkpoint (diagnostic); 0 = off")
  p.add_argument("--seed", type=int, default=20474010)
  p.add_argument("--device", default=None, help="cuda / mps / cpu; default cuda if available, else cpu")
  p.add_argument("--threads", type=int, default=4)
  return p


def main():
  p = arguments()
  args = p.parse_args()

  import torch
  from stable_baselines3 import SAC
  from stable_baselines3.common.callbacks import BaseCallback
  from stable_baselines3.common.logger import configure
  torch.set_num_threads(args.threads)
  torch.manual_seed(args.seed)
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  args.out = args.out.resolve()
  if SIMULATION / "data" not in args.out.parents:
    p.error("keep --out under simulation/data/")
  if args.out.exists():
    raise FileExistsError(f"refusing to overwrite {args.out}; use a new --out")
  (args.out / "models").mkdir(parents=True)
  started = time.monotonic()

  cfg, rest_z = run_settings(args.models_run, args.tag)
  models = ImaginationModels(args.models_run, args.tag, device, args.penalty_scale)
  train = load_starts(args.run, ("train",), models, cfg, rest_z, args.obstacles)
  val = load_starts(args.run, ("val",), models, cfg, rest_z, args.obstacles)
  if not len(train["index"]) or not len(val["index"]):
    raise ValueError("training and validation both need recorded starts")
  print(f"starts: {len(train['index'])} training, {len(val['index'])} validation states "
        f"({'with' if args.obstacles else 'without'} obstacle scenes)", flush=True)
  env = ImaginedEnv(models, train, args.envs, args.horizon, cfg, rest_z, args.seed, grasp_check=args.grasp_check)
  fixed = np.random.default_rng(args.seed + 1).choice(len(val["index"]), min(args.validation_starts, len(val["index"])), replace=False)
  val_env = ImaginedEnv(models, val, len(fixed), args.validation_horizon, cfg, rest_z, args.seed + 2, fixed=fixed, grasp_check=args.grasp_check)

  options = dict(learning_rate=args.learning_rate, buffer_size=max(args.steps + args.prefill * args.envs, 1000),
                 learning_starts=0, batch_size=256, gamma=GoalReward.gamma, train_freq=1,
                 gradient_steps=max(1, round(args.updates_per_step * args.envs)))
  initial_sha256 = None
  if args.initialize_run is None:
    model = SAC("MlpPolicy", env, device=device, seed=args.seed, verbose=0,
                policy_kwargs={"net_arch": [128, 128]}, ent_coef=args.ent_coef, **options)
  else:
    import hashlib
    source = args.initialize_run / "models/sac_best.zip"
    record = json.loads((args.initialize_run / "pipeline.json").read_text())
    if record.get("kind") != "imagined_sac":
      raise ValueError("--initialize-run must be Calle's imagined SAC, not the ordinary simulator SAC")
    initial_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    model = SAC.load(source, env=env, device=device, **options)
    model.set_random_seed(args.seed)
    model.num_timesteps = 0  # report additional transitions, including the initial checkpoint at zero
  model.set_logger(configure(str(args.out / "sac_logs"), ["csv"]))
  demo_x, demo_y = demonstrations(env, train)
  demo_x = torch.as_tensor(demo_x, device=model.device)
  demo_y = torch.as_tensor(demo_y, dtype=torch.float32, device=model.device)
  if args.bc_epochs and args.initialize_run is None:
    clone(model, demo_x, demo_y, args.bc_epochs, args.seed)

  # a cloning term added to every SAC actor update (rl_control's), so RL does not wander off the demonstrations;
  # with --td3bc-alpha its weight is mean|Q| / alpha on the demonstration batch (TD3+BC's normalisation)
  def demonstration_gradient(optimizer, positional, keyword):
    if not model.actor.mu.weight.requires_grad or not (args.bc_weight or args.td3bc_alpha):
      return
    ids = torch.randint(len(demo_x), (model.batch_size,), device=model.device)
    actions = model.actor(demo_x[ids], deterministic=True)
    error = (actions - demo_y[ids]).square()
    weight = args.bc_weight
    if args.td3bc_alpha:
      with torch.no_grad():
        q = torch.cat(model.critic(demo_x[ids], actions.detach()), dim=1).min(1).values
      weight = float(q.abs().mean()) / args.td3bc_alpha
    (weight * (error[:, :4].mean() + 2 * error[:, 4].mean())).backward()

  model.actor.optimizer.register_step_pre_hook(demonstration_gradient)
  exact_seeds = [args.seed + 10000 + i for i in range(args.exact_checks)]
  history = {"validation": [], "best": None}

  # the best checkpoint by imagined placements, then imagined return (the simulator is not consulted)
  def checkpoint(name):
    evaluation = {"transitions": int(model.num_timesteps), "seconds": time.monotonic() - started, **imagined_validation(model, val_env)}
    if args.exact_checks:
      evaluation.update(exact_state_check(model, cfg, exact_seeds))
    model.save(args.out / "models" / f"{name}.zip")
    key = lambda e: (e["imagined_placed"], e["imagined_return"])
    if history["best"] is None or key(evaluation) > key(history["best"]):
      history["best"] = {**evaluation, "file": f"{name}.zip"}
      shutil.copyfile(args.out / "models" / f"{name}.zip", args.out / "models/sac_best.zip")
    history["validation"].append(evaluation)
    print(json.dumps(evaluation), flush=True)

  class Checkpoints(BaseCallback):
    def _on_step(self):
      if self.num_timesteps // args.eval_every != (self.num_timesteps - self.training_env.num_envs) // args.eval_every:
        checkpoint(f"sac_{self.num_timesteps:07d}")
      return True

  checkpoint("sac_cloned" if args.initialize_run is None else "sac_initial")
  prefill(model, env, args.prefill)
  if args.critic_warmup:
    # the critic first, with the actor frozen: a random critic would pull the cloned actor apart
    for parameter in model.actor.parameters():
      parameter.requires_grad_(False)
    model.train(gradient_steps=args.critic_warmup, batch_size=256)
    for parameter in model.actor.parameters():
      parameter.requires_grad_(True)
    print(f"critic warmup: {args.critic_warmup} updates on {args.prefill * args.envs} imagined transitions", flush=True)
  model.learn(args.steps, callback=Checkpoints())
  if history["validation"][-1]["transitions"] != model.num_timesteps:
    checkpoint(f"sac_{model.num_timesteps:07d}")
  # what the controllers read: control_pipeline.Policy takes inputs.options.obstacles to decide on the height map
  write_json(args.out / "pipeline.json", {"inputs": {"options": {"obstacles": None}}, "kind": "imagined_sac"})
  write_json(args.out / "imagined_training.json", {
    **history, "seconds": time.monotonic() - started, "demonstrations": int(len(demo_x)),
    "starts": {"train": int(len(train["index"])), "val": int(len(val["index"]))},
    "settings": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()},
    "exact_state_seeds": exact_seeds, "device": device,
    "initial_checkpoint_sha256": initial_sha256,
    "bc_reinitialized": args.initialize_run is None and bool(args.bc_epochs),
    "terminal_observation_copy_fix": True})
  print(f"DONE: best checkpoint {history['best']['file']} (by imagined placements) copied to {args.out / 'models/sac_best.zip'}", flush=True)


if __name__ == "__main__":
  main()
