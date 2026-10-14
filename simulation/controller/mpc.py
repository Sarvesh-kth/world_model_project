import copy
import json
import pathlib
import runpy
import time

import numpy as np

from rl.task_control import predicted_reward

# The standalone MPC planner, method "mpc" of controller/control_pipeline.py: the proposal's CEM/MPC
# controller, searching action sequences directly in the world model's imagination, no policy and no
# hand-coded motion anywhere. Every candidate is a sequence of raw actions; D imagines it, Q reads every
# imagined state, and it is scored with the team's task reward (GoalReward) on those readings plus R's
# penalties and the obstacle model O's at the imagined hand (imagine()), plus the hand-made parts of the
# config. jepa_mpc (planner.py) is the rule-based counterpart: it plans over waypoints that a go-to rule turns
# into actions. All tuning knobs live in controller/mpc_config.py.
#   python -m controller.control_pipeline --methods mpc --random-ends --test-episodes 1 --headless --out data/mpc_smoke
#   python -m controller.control_pipeline --methods mpc --mpc-config controller/my_copy.py
#   python -m controller.mpc --self-check          (on CPU, no camera: checks the planner logic)

DEFAULT_CONFIG = pathlib.Path(__file__).with_name("mpc_config.py")
REQUIRED = ("HORIZON", "POPULATION", "ELITES", "ITERATIONS", "NOISE_STD", "NOISE_MIN", "NOISE_SMOOTH", "GRIPPER",
            "RULE_CLOSE_DISTANCE", "RULE_RELEASE_RADIUS", "RULE_RELEASE_HEIGHT", "SCORE", "PROGRESS_WEIGHT",
            "DISCOUNT", "PENALTY_SCALE", "SAVE_RUNNER_UPS", "REACH_PULL", "HELD_NEEDS_CLOSED", "PLACE_SLOPE",
            "PLACED_AFTER", "RETREAT_AFTER_PLACE", "CUBE_STAYS_PUT", "CARRY_HEIGHT", "CARRY_WEIGHT",
            "GRASP_NEEDS_CONTACT", "CONTACT_RADIUS", "PLACED_CLEARANCE", "EMPTY_GRIP_COST", "PLACED_RADIUS",
            "PULL_WHOLE_WAY", "HELD_Z_OFFSET", "CARRY_STEADY", "PLACED_KEEPS_POTENTIAL", "RELEASED_STAYS_PUT",
            "APPROACH_SLOPE")


# The knobs of a config file (every UPPERCASE name), checked once
def load_settings(path=None):
  path = pathlib.Path(path) if path else DEFAULT_CONFIG
  settings = {k: v for k, v in runpy.run_path(str(path)).items() if k.isupper()}
  missing = [k for k in REQUIRED if k not in settings]
  if missing:
    raise ValueError(f"{path} is missing {missing}")
  s = settings
  if not (s["HORIZON"] >= 1 and s["ITERATIONS"] >= 1 and 2 <= s["ELITES"] < s["POPULATION"]
          and s["NOISE_STD"] > 0 and s["NOISE_MIN"] >= 0 and 0 <= s["NOISE_SMOOTH"] <= 1 and 0 < s["DISCOUNT"] <= 1):
    raise ValueError(f"{path}: invalid numbers (HORIZON, ITERATIONS >= 1; 2 <= ELITES < POPULATION; NOISE_STD > 0; "
                     f"NOISE_MIN >= 0; 0 <= NOISE_SMOOTH <= 1; 0 < DISCOUNT <= 1)")
  if not (0 < s["PLACE_SLOPE"] < 1 and int(s["PLACED_AFTER"]) == s["PLACED_AFTER"] >= 1 and s["CARRY_HEIGHT"] >= 0
          and s["HELD_Z_OFFSET"] >= 0):
    raise ValueError(f"{path}: 0 < PLACE_SLOPE < 1, PLACED_AFTER a whole number >= 1, CARRY_HEIGHT >= 0, "
                     f"HELD_Z_OFFSET >= 0")
  if not (s["CARRY_WEIGHT"] >= 0 and s["CONTACT_RADIUS"] > 0 and s["PLACED_CLEARANCE"] >= 0 and s["EMPTY_GRIP_COST"] >= 0
          and s["PLACED_RADIUS"] > 0 and s["CARRY_STEADY"] >= 0 and s["APPROACH_SLOPE"] >= 0):
    raise ValueError(f"{path}: CARRY_WEIGHT >= 0, CONTACT_RADIUS > 0, PLACED_CLEARANCE >= 0, EMPTY_GRIP_COST >= 0, "
                     f"PLACED_RADIUS > 0, CARRY_STEADY >= 0, APPROACH_SLOPE >= 0")
  if s["GRIPPER"] not in ("sampled", "switch", "rule") or s["SCORE"] not in ("reward", "progress"):
    raise ValueError(f"{path}: GRIPPER must be 'sampled', 'switch' or 'rule', SCORE 'reward' or 'progress'")
  settings["CONFIG_FILE"] = str(path.resolve())
  return settings


# The "rule" gripper from Q's reading of the current state: +1 open, -1 closed
def gripper_rule(xyz, held, p, goal, s):
  if held > .5:
    over_target = np.linalg.norm(xyz[:2] - goal[:2]) < s["RULE_RELEASE_RADIUS"] and xyz[2] - goal[2] < s["RULE_RELEASE_HEIGHT"]
    return 1.0 if over_target else -1.0
  return -1.0 if np.linalg.norm(p[14:17] - xyz) < s["RULE_CLOSE_DISTANCE"] else 1.0


# Whether a lifted cube has just been let go at the target, from Q's reading and the reward memory: the
# gripper is commanded open, the cube was held and lifted before, and Q puts it within the success radius of B
def released_at_target(memory, xyz, p, goal, radius):
  return bool(p[19] > 0 and memory.was_grasped and memory.ever_lifted and np.linalg.norm(xyz[:2] - goal[:2]) < radius)


# Q's reading inside the imagination, with the physical checks of the config: the cube only counts as held
# while the fingers are closed around something (HELD_NEEDS_CLOSED) and, unless it is really held now, only
# after the imagined gripper has come within CONTACT_RADIUS of where Q sees it now with the fingers still open
# (GRASP_NEEDS_CONTACT): a grasp needs open fingers arriving at the cube and closing there. D and Q never saw
# failed grasps, so without this a plan that closes the fingers anywhere imagines a grasp. With
# RELEASED_STAYS_PUT a cube let go in imagination stays (in x and y) where the fingers last held it, Q's held
# reading, instead of where Q reads it on the imagined latent. One wrapper per imagine() call, which reads the
# candidates in the same order at every step, so `touched` and `let_go` follow each candidate through its plan.
def imagined_read(read, s, xyz_now, really_held):
  touched = let_go = None

  def gated(zz, pp):
    nonlocal touched, let_go
    xyz, held = read(zz, pp)
    pp = np.asarray(pp)
    if s["HELD_NEEDS_CLOSED"]:
      held = held * (pp[:, 18] < .06)
    if s["GRASP_NEEDS_CONTACT"]:
      if touched is None or len(touched) != len(pp):
        touched = np.full(len(pp), really_held)
      touched |= (np.linalg.norm(pp[:, 14:17] - xyz_now, axis=1) < s["CONTACT_RADIUS"]) & (pp[:, 18] > .06)
      held = held * touched
    if s["RELEASED_STAYS_PUT"]:
      if let_go is None or len(let_go) != len(pp):
        let_go = np.full((len(pp), 3), np.nan, np.float32)
      xyz = np.array(xyz, np.float32)
      released = (held < .5) & np.isfinite(let_go[:, 0])
      xyz[released, :2] = let_go[released, :2]
      let_go = np.where((held >= .5)[:, None], xyz, let_go)
    return xyz, held
  return gated


# The gripper column of plans from one switch time each: the current command until step `switch`, then the
# opposite (at or past the horizon: no switch)
def gripper_from_switch(switch, command, horizon):
  return np.where(np.arange(horizon)[None] < np.asarray(switch)[:, None], command, -command).astype(np.float32)


# Imagine candidate plans (n, horizon, 5) from the current (z, p) and score them with the task reward: D
# imagines every step, `read` (Q with the config's checks) reads every imagined state, and each step pays the
# team's GoalReward on those readings plus R's penalties and, given the obstacle-only map, the obstacle model
# O's at the imagined hand. The score is the discounted sum, with no terminal value. The potentials (for the
# "progress" score) are the reward's own, except that after an imagined placement they stay at placed_potential
# (the reward sets them to 0 once the episode is over). (On world_model_testing this was
# control_pipeline.forecasts, shared with jepa_mpc; the v3c jepa_mpc scores waypoints its own way.)
def imagine(models, read, z, p, actions, memory, goal, rest_z, step, cfg, discount, obstacle_map=None, placed_potential=0.0):
  n, horizon, _ = actions.shape
  weights = cfg.rewards.weights
  zz, pp = np.repeat(z[None], n, 0), np.repeat(p[None], n, 0)
  xyz, held = read(zz, pp)
  ps, xyzs, helds = [pp], [xyz], [held]
  memories = [copy.deepcopy(memory) for _ in range(n)]
  rewards = np.zeros((n, horizon), np.float32)
  penalties = np.zeros((n, horizon), np.float32)
  # the task potential of every imagined state (the "progress" score)
  potentials = np.zeros((n, horizon), np.float32)
  valid = np.ones(n, bool)
  finished = np.zeros(n, bool)

  for t in range(horizon):
    z_next, p_next = models.predict(zz, pp, actions[:, t])
    finite = np.isfinite(z_next).all(1) & np.isfinite(p_next).all(1)
    z_next[~finite], p_next[~finite] = zz[~finite], pp[~finite]
    xyz, held = read(z_next, p_next)
    # the bounds of the v3c planner: the open gripper measures up to .083 and the v3 D overshoots it by up to a
    # centimetre, so only clearly impossible widths are rejected
    valid &= (finite & np.isfinite(xyz).all(1) & np.isfinite(held) & (p_next[:, 18] >= -.02) & (p_next[:, 18] <= .12)
              & (np.abs(p_next[:, 14:17]) < 5).all(1) & (np.abs(xyz) < 5).all(1) & (np.abs(p_next) < 100).all(1)
              & (np.abs(z_next) < 1e4).all(1))
    penalties[:, t] = models.penalty(models.reward(z_next, p_next), weights)
    if obstacle_map is not None and models.obstacle is not None:
      penalties[:, t] += models.obstacle_penalty(obstacle_map, p_next, held, weights)
    for i in range(n):
      if finished[i]:
        potentials[i, t] = potentials[i, t - 1]
        continue
      failed = xyz[i, 2] < cfg.table.height - cfg.episode.fall_margin or step + t + 1 >= cfg.episode.max_steps
      rewards[i, t], _ = predicted_reward(memories[i], p_next[i], xyz[i], held[i], actions[i, t], goal, rest_z, failed)
      rewards[i, t] += penalties[i, t]
      potentials[i, t] = placed_potential if memories[i].task_succeeded else memories[i].potential
      finished[i] = memories[i].task_succeeded or failed
    zz, pp = z_next, p_next
    ps.append(pp)
    xyzs.append(xyz)
    helds.append(held)

  discounted = rewards @ np.power(discount, np.arange(horizon))
  valid &= np.isfinite(discounted)
  scores = np.where(valid, discounted, -1e9).astype(np.float32)
  return {"actions": actions, "p": np.stack(ps, 1), "xyz": np.stack(xyzs, 1), "held": np.stack(helds, 1),
          "rewards": rewards, "penalties": penalties, "potentials": potentials, "scores": scores, "valid": valid,
          "discounted_reward": discounted}


# Score candidate plans (n, horizon, 5) as the planner does: imagine() with Q's reading under the physical
# checks, plus the hand-made parts of the config. Returns imagine()'s pool with the final "scores" and their
# "parts" (task reward incl. R's and O's penalties, progress, pull, empty grip, steadiness).
def score_plans(models, z, p, actions, memory, goal, rest_z, step, cfg, s, obstacle_map=None):
  xyz, held = models.read(z[None], p[None])
  xyz, held = xyz[0], float(held[0])
  # really held now: Q says so and the robot's own fingers are closed around something cube-sized
  really_held = bool(held > .5 and p[19] <= 0 and .02 < p[18] < .06)
  placed = released_at_target(memory, xyz, p, goal, cfg.episode.success_radius)

  # in imagination the cube counts as placed after PLACED_AFTER settled steps instead of the task's 15, so
  # letting go at B can pay off inside the horizon; the real success test is unchanged
  imagined = copy.deepcopy(memory)
  imagined.settle_required = min(memory.settle_required, s["PLACED_AFTER"])
  # and only within PLACED_RADIUS of B instead of the task's 7 cm, a margin for Q's few cm of error (letting go
  # beyond twice the radius then also counts as a drop in the task reward)
  imagined.success_radius = min(memory.success_radius, s["PLACED_RADIUS"])

  # R's and O's penalties are scaled by models.penalty_scale (shared with jepa_mpc's --penalty-scale): set to
  # the config's for the call and restored after
  saved_scale, models.penalty_scale = models.penalty_scale, s["PENALTY_SCALE"]
  # a placed cube keeps the full potential (every shaped term fulfilled) with PLACED_KEEPS_POTENTIAL
  full = sum(cfg.rewards.weights[k] for k in ("reach", "grasp", "lift", "transport")) if s["PLACED_KEEPS_POTENTIAL"] else 0.0
  try:
    pool = imagine(models, imagined_read(models.read, s, xyz, really_held), z, p, actions, imagined, goal, rest_z, step,
                   cfg, s["DISCOUNT"], obstacle_map, full)
  finally:
    models.penalty_scale = saved_scale

  parts = {"task": pool["discounted_reward"].copy()}
  if s["SCORE"] == "progress":
    # also pay the task potential of every imagined state: getting there early is worth more
    parts["progress"] = s["PROGRESS_WEIGHT"] * pool["potentials"].mean(1)
  if s["REACH_PULL"]:
    gripper, cube, carried = pool["p"][:, 1:, 14:17], pool["xyz"][:, 1:], pool["held"][:, 1:] > .5
    across = np.linalg.norm(cube[..., :2] - goal[:2], axis=-1)
    # Q reads a carried cube HELD_Z_OFFSET too high (measured: about 3 cm), the heights below correct for it
    height = cube[..., 2] - s["HELD_Z_OFFSET"]
    # a carried cube may be at most PLACE_SLOPE x its distance to B (+1 cm) above its resting height, so it
    # comes down as it arrives instead of hovering over B; with a slope under 1 getting closer always pays
    above = np.maximum(0, height - (goal[2] + .01 + s["PLACE_SLOPE"] * across))
    # and while the glide slope allows it, at least CARRY_HEIGHT up: a placement only counts if the cube was
    # lifted 4 cm, and a strong pull alone drags it along low
    lifted_zone = across > (s["CARRY_HEIGHT"] - .01) / s["PLACE_SLOPE"]
    below = np.where(lifted_zone, np.maximum(0, goal[2] + s["CARRY_HEIGHT"] - height), 0)
    # an unheld cube does not move: with CUBE_STAYS_PUT the reach part measures to Q's reading of the current
    # frame instead of the imagined cube, which off the expert route drifts along with the arm
    target = np.broadcast_to(xyz, cube.shape) if s["CUBE_STAYS_PUT"] else cube
    reach = np.linalg.norm(gripper - target, axis=-1)
    if s["APPROACH_SLOPE"]:
      # the hand comes down onto the cube from above: under a cone over the cube (APPROACH_SLOPE x the
      # horizontal distance beyond 2 cm) every centimetre costs CARRY_WEIGHT centimetres of distance
      lateral = np.linalg.norm(gripper[..., :2] - target[..., :2], axis=-1)
      cone = target[..., 2] + s["APPROACH_SLOPE"] * np.maximum(0, lateral - .02)
      reach = reach + s["CARRY_WEIGHT"] * np.maximum(0, cone - gripper[..., 2])
    if s["PULL_WHOLE_WAY"] and not placed:
      # the whole way still to go, so finishing a stage never costs: before the grasp also the cube's way from
      # where it rests now to B and the lift it still needs (otherwise grasping made the pull jump from the cm
      # to the cube to the half metre to B, and the planner never closed the fingers)
      rest_across = float(np.linalg.norm(xyz[:2] - goal[:2]))
      rest_below = (max(0.0, float(goal[2] + s["CARRY_HEIGHT"] - xyz[2]))
                    if rest_across > (s["CARRY_HEIGHT"] - .01) / s["PLACE_SLOPE"] else 0.0)
      reach = reach + rest_across + s["CARRY_WEIGHT"] * rest_below
    if placed and s["PLACED_CLEARANCE"]:
      # the cube was let go at B: now the gripper should keep PLACED_CLEARANCE away from it, not reach for it
      reach = np.maximum(0, s["PLACED_CLEARANCE"] - np.linalg.norm(gripper - xyz, axis=-1))
    far = np.where(carried, across + above + s["CARRY_WEIGHT"] * below, reach)
    parts["pull"] = -s["REACH_PULL"] * far.mean(1)
  if s["EMPTY_GRIP_COST"]:
    # fingers commanded closed while nothing is held cost a little: the hand stays open on the way to the
    # cube, and after a missed grasp opening again pays, so the planner can retry on its own
    empty = (actions[:, :, -1] < 0) & (pool["held"][:, 1:] <= .5)
    parts["empty_grip"] = -s["EMPTY_GRIP_COST"] * empty.mean(1)
  if s["CARRY_STEADY"] and actions.shape[1] > 1:
    # (Sarvesh's idea) a carried cube should not keep changing height: the change of the imagined gripper's
    # vertical speed costs while carrying, so steady lifting and lowering are free but up-down wobble is not
    z = pool["p"][:, :, 16]
    wobble = np.abs(z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]) * (pool["held"][:, 1:-1] > .5)
    parts["steady"] = -s["CARRY_STEADY"] * wobble.mean(1)
  extra = sum(v for k, v in parts.items() if k != "task") if len(parts) > 1 else 0
  pool["scores"] = np.where(pool["valid"], pool["scores"] + extra, -1e9).astype(np.float32)
  pool["parts"] = parts
  return pool


# One planning step. Returns the action, the whole chosen plan (the next step's warm start) and the
# bookkeeping step_record() logs. obstacle_map = the obstacle-only depth map the controller keeps for O
# (None: R's penalties only).
def plan_standalone(models, z, p, memory, goal, rest_z, step, cfg, settings, rng, previous=None, obstacle_map=None):
  s = settings
  horizon = min(s["HORIZON"], cfg.episode.max_steps - step)
  command = 1.0 if p[19] > 0 else -1.0
  xyz, held = models.read(z[None], p[None])
  xyz, held = xyz[0], float(held[0])
  score = lambda actions: score_plans(models, z, p, actions, memory, goal, rest_z, step, cfg, s, obstacle_map)

  if s["RETREAT_AFTER_PLACE"] and released_at_target(memory, xyz, p, goal, cfg.episode.success_radius):
    # the cube was let go at B: no search, the fingers stay open and the arm rises once they are clear of
    # the cube, so nothing knocks it while it settles (scored as a one-candidate plan for the logs)
    retreat = np.array([0, 0, .5 if p[18] > .06 else 0, 0, 1], np.float32)
    best = {"pool": score(np.tile(retreat, (1, horizon, 1))), "selected": 0}
  else:
    # warm start: the previous plan shifted by one step with its last action repeated; the first plan holds still
    if previous is None:
      mean = np.zeros((horizon, 5), np.float32)
      mean[:, -1] = command
    else:
      shifted = np.concatenate((previous[1:], previous[-1:]))
      mean = np.concatenate((shifted, np.repeat(shifted[-1:], max(0, horizon - len(shifted)), 0)))[:horizon].astype(np.float32)
    std = np.full((horizon, 5), s["NOISE_STD"], np.float32)
    floor = np.full(5, s["NOISE_MIN"], np.float32)

    # the "rule" gripper is decided once from the current reading and fixed for every candidate
    rule = None
    if s["GRIPPER"] == "rule":
      rule = gripper_rule(xyz, held, p, goal, s)
      mean[:, -1], std[:, -1], floor[-1] = rule, 0, 0

    # the "switch" gripper: one switch time per plan, sampled and refitted like the motions; the warm start
    # keeps the previous plan's switch (one step earlier now), the first plan does not switch
    if s["GRIPPER"] == "switch":
      flips = np.nonzero(mean[:, -1] != command)[0]
      switch_mean, switch_std = (float(flips[0]) if len(flips) else horizon + 2.0), horizon / 2

    best = None
    for _ in range(s["ITERATIONS"]):
      noise = rng.normal(size=(s["POPULATION"], horizon, 5))
      if s["NOISE_SMOOTH"] > 0:
        noise = np.sqrt(s["NOISE_SMOOTH"]) * rng.normal(size=(s["POPULATION"], 1, 5)) + np.sqrt(1 - s["NOISE_SMOOTH"]) * noise
      actions = np.clip(mean + std * noise, -1, 1).astype(np.float32)
      # candidate 0 is the current mean itself, so a good plan can simply be continued
      actions[0] = np.clip(mean, -1, 1)
      if s["GRIPPER"] == "switch":
        switch = switch_mean + switch_std * rng.normal(size=s["POPULATION"])
        switch[0] = switch_mean
        actions[..., -1] = gripper_from_switch(switch, command, horizon)
      else:
        actions[..., -1] = rule if rule is not None else np.where(actions[..., -1] > 0, 1, -1)
      pool = score(actions)
      selected = int(np.argmax(pool["scores"]))
      if best is None or pool["scores"][selected] > best["pool"]["scores"][best["selected"]]:
        best = {"pool": pool, "selected": selected}
      order = np.argsort(pool["scores"])[-s["ELITES"]:]
      elites = actions[order]
      mean, std = elites.mean(0), np.maximum(elites.std(0), floor)
      if s["GRIPPER"] == "switch":
        switch_mean, switch_std = float(switch[order].mean()), max(float(switch[order].std()), .5)

  pool, selected = best["pool"], best["selected"]
  fallback = not bool(pool["valid"][selected])
  if fallback:
    # every candidate left the physical range in imagination: hold still and keep the gripper as it is
    action = np.array([0, 0, 0, 0, command], np.float32)
    sequence = np.tile(action, (horizon, 1))
  else:
    action, sequence = pool["actions"][selected, 0], pool["actions"][selected]
  return {"pool": pool, "selected": selected, "fallback": fallback, "best_score": float(pool["scores"][selected]),
          "action": action, "sequence": sequence, "overlay": overlay_record(pool, selected, s["SAVE_RUNNER_UPS"], s["HORIZON"])}


# What episode() logs for one planning step, once the action has been executed: the steps.csv fields (the
# chosen plan's score and its parts, and how far its first imagined step was from what happened) and one
# candidates.csv row per candidate
def step_record(decision, step, actual_xyz, actual_p):
  pool, selected = decision["pool"], decision["selected"]
  row = {"predicted_score": decision["best_score"], "predicted_penalty_sum": float(pool["penalties"][selected].sum()),
         "planner_fallback": decision["fallback"], "valid_candidates": int(pool["valid"].sum())}
  row.update({f"score_{k}": float(v[selected]) for k, v in pool["parts"].items()})
  if not decision["fallback"]:
    row.update({"D_Q_one_step_xyz_mae_cm": float(100 * np.abs(pool["xyz"][selected, 1] - actual_xyz).mean()),
                "D_one_step_ee_mae_cm": float(100 * np.abs(pool["p"][selected, 1, 14:17] - actual_p[14:17]).mean())})
  candidates = [{"step": step + 1, "candidate": i, "selected": i == selected, "valid": bool(pool["valid"][i]),
                 "score": float(pool["scores"][i]), "discounted_reward": float(pool["discounted_reward"][i]),
                 "penalty_sum": float(pool["penalties"][i].sum())} for i in range(len(pool["scores"]))]
  return row, candidates


# What the replay viewer draws for one step: the chosen plan's imagined gripper and cube paths and the
# runner-ups' gripper paths, padded with NaN to a fixed horizon (the horizon shrinks at the end of an episode)
def overlay_record(pool, selected, runner_ups, horizon):
  def padded(path):
    out = np.full((horizon + 1, 3), np.nan, np.float32)
    out[:len(path)] = path
    return out

  gripper = pool["p"][:, :, 14:17]
  order = [i for i in np.argsort(pool["scores"])[::-1] if pool["valid"][i] and i != selected][:runner_ups]
  runners = np.full((runner_ups, horizon + 1, 3), np.nan, np.float32)
  for k, i in enumerate(order):
    runners[k] = padded(gripper[i])
  held = np.full(horizon + 1, np.nan, np.float32)
  held[:pool["held"].shape[1]] = pool["held"][selected]
  return {"chosen_gripper": padded(gripper[selected]), "chosen_cube": padded(pool["xyz"][selected]), "chosen_held": held,
          "runner_gripper": runners, "score": float(pool["scores"][selected]), "valid": int(pool["valid"].sum())}


def save_plans(folder, records):
  keys = records[0].keys()
  np.savez_compressed(pathlib.Path(folder) / "plans.npz", **{k: np.asarray([r[k] for r in records]) for k in keys})


def save_settings(folder, settings):
  (pathlib.Path(folder) / "mpc_settings.json").write_text(json.dumps(settings, indent=2) + "\n")


# The planner's world model on the CPU without the live encoder: D, Q, R and O, with WorldModels' defaults
# (width gate, gripper envelope). For offline checks; the controllers use WorldModels with the encoder.
def offline_models(models_run="models/v3c", tag="", penalty_scale=2.0):
  import torch
  from world_model.train import load_model
  from world_model.train_obstacle import load_obstacle_model
  from .control_pipeline import WorldModels
  models = WorldModels.__new__(WorldModels)
  models.torch, models.device = torch, "cpu"
  root = pathlib.Path(models_run)
  models.d, models.dc = load_model(root, "dynamics", tag)
  models.q, models.qc = load_model(root, "readout", tag)
  models.r, models.rc = load_model(root, "reward", tag)
  models.obstacle, models.oc = load_obstacle_model(root, tag)
  models.penalty_scale, models.width_gate, models.grasp_prior = penalty_scale, True, True
  models.reward_source, models.no_goal, models.cost_to_go = "readout", False, None
  return models


# Planner logic check on the CPU: real D / Q / R and a real robot state from the simulator, but a stand-in
# latent (D's mean latent) because the live encoder needs CUDA. Checks shapes, the gripper modes, the warm
# start, the scoring options, the fallback, that the penalty scale is restored, and the time per step.
def self_check():
  from rl.task_control import GoalReward, TaskSession
  from .control_pipeline import run_settings
  cfg, rest_z = run_settings("models/v3c", "")
  layout = json.loads(pathlib.Path("configs/grade_e_layout.json").read_text())
  session = TaskSession(cfg, layout, .01)
  session.reset(20494010)
  p = session.obs["proprio"].copy()
  session.close()
  models = offline_models()
  z = models.dc["z_mean"].numpy().astype(np.float32)
  goal = np.array([*layout["place"], rest_z], np.float32)
  base = load_settings()

  plain = {"GRIPPER": "sampled", "SCORE": "reward", "REACH_PULL": 0.0, "CUBE_STAYS_PUT": False, "HELD_NEEDS_CLOSED": False,
           "GRASP_NEEDS_CONTACT": False, "PLACED_AFTER": 15, "PLACED_CLEARANCE": 0.0, "RETREAT_AFTER_PLACE": False,
           "EMPTY_GRIP_COST": 0.0, "PLACED_RADIUS": 1.0, "PULL_WHOLE_WAY": False, "CARRY_STEADY": 0.0,
           "RELEASED_STAYS_PUT": False}
  rule = {"GRIPPER": "rule", "SCORE": "reward", "RETREAT_AFTER_PLACE": True, "GRASP_NEEDS_CONTACT": False, "PLACED_CLEARANCE": 0.0,
          "CARRY_WEIGHT": 1.0, "EMPTY_GRIP_COST": 0.0, "PLACED_RADIUS": 1.0, "PULL_WHOLE_WAY": False,
          "HELD_Z_OFFSET": 0.0, "CARRY_HEIGHT": 0.05, "CARRY_STEADY": 0.0, "RELEASED_STAYS_PUT": False, "APPROACH_SLOPE": 0.0}
  for variant in ({}, plain, rule, {"GRIPPER": "sampled"}, {"SCORE": "reward"}, {"PULL_WHOLE_WAY": False},
                  {"APPROACH_SLOPE": 0.0, "RELEASED_STAYS_PUT": False, "PLACED_KEEPS_POTENTIAL": False}):
    s = {**base, **variant}
    memory, previous, rng = GoalReward(cfg), None, np.random.default_rng(0)
    started = time.monotonic()
    for step in range(3):
      d = plan_standalone(models, z, p, memory, goal, rest_z, step, cfg, s, rng, previous)
      previous = d["sequence"]
      assert d["action"].shape == (5,) and d["sequence"].shape == (s["HORIZON"], 5)
      assert np.all(np.abs(d["sequence"]) <= 1) and set(np.unique(d["sequence"][:, -1])) <= {-1.0, 1.0}
      assert d["overlay"]["chosen_gripper"].shape == (s["HORIZON"] + 1, 3)
      assert models.penalty_scale == 2.0, "penalty scale was not restored"
      if s["GRIPPER"] == "switch":
        flips = np.abs(np.diff(d["pool"]["actions"][:, :, -1], axis=1)).sum(1) / 2
        assert flips.max() <= 1, "a switch-gripper plan switched more than once"
    seconds = (time.monotonic() - started) / 3
    print(f"variant {variant or 'default'}: ok, {seconds:.2f} s per planning step on the CPU, "
          f"valid candidates {d['overlay']['valid']}/{s['POPULATION']}, chosen score {d['best_score']:.3f}, "
          f"first action {np.round(d['action'], 2)}")

  # the same seed gives the same plan
  a = plan_standalone(models, z, p, GoalReward(cfg), goal, rest_z, 0, cfg, base, np.random.default_rng(1))
  b = plan_standalone(models, z, p, GoalReward(cfg), goal, rest_z, 0, cfg, base, np.random.default_rng(1))
  assert np.array_equal(a["sequence"], b["sequence"])

  # a horizon cut short at the end of the episode still gives full-size overlay records
  late = plan_standalone(models, z, p, GoalReward(cfg), goal, rest_z, cfg.episode.max_steps - 3, cfg, base, np.random.default_rng(2))
  assert late["sequence"].shape == (3, 5) and np.isnan(late["overlay"]["chosen_gripper"][-1]).all()

  # every candidate invalid: hold still and keep the gripper (score_plans finds imagine() in this module's names)
  real = globals()["imagine"]
  def all_invalid(*args):
    pool = real(*args)
    pool["valid"][:] = False
    pool["scores"][:] = -1e9
    return pool
  globals()["imagine"] = all_invalid
  try:
    f = plan_standalone(models, z, p, GoalReward(cfg), goal, rest_z, 0, cfg, base, np.random.default_rng(3))
  finally:
    globals()["imagine"] = real
  assert f["fallback"] and np.array_equal(f["action"][:4], np.zeros(4)) and f["action"][4] == (1.0 if p[19] > 0 else -1.0)
  print("same seed same plan, short horizon at the episode end, fallback: ok")

  # the log record, and O: the same plans scored on a table full of 15 cm obstacles cost more than on an
  # empty one, and with no obstacle map O is not asked at all
  row, rows = step_record(a, 0, np.zeros(3), p)
  assert len(rows) == base["POPULATION"] and sum(r["selected"] for r in rows) == 1 and "score_pull" in row
  actions = a["pool"]["actions"]
  cells = 16
  empty, walls = (score_plans(models, z, p, actions, GoalReward(cfg), goal, rest_z, 0, cfg, base, np.full((cells, cells), h, np.float32))
                  for h in (0.0, .15))
  none = score_plans(models, z, p, actions, GoalReward(cfg), goal, rest_z, 0, cfg, base, None)
  ok = empty["valid"] & walls["valid"]
  assert ok.any() and np.all(walls["penalties"].sum(1)[ok] < empty["penalties"].sum(1)[ok]), "O did not penalise the obstacles"
  assert np.all(none["penalties"].sum(1)[ok] >= walls["penalties"].sum(1)[ok])
  print(f"log record ok; O's penalty per plan: empty table {empty['penalties'].sum(1)[ok].mean():.3f}, "
        f"15 cm obstacles everywhere {walls['penalties'].sum(1)[ok].mean():.3f}, no map {none['penalties'].sum(1)[ok].mean():.3f}")

  # the switch gripper and the imagined read's checks on their own
  assert np.array_equal(gripper_from_switch([0, 3, 10], 1.0, 5), [[-1] * 5, [1, 1, 1, -1, -1], [1] * 5])
  always_held = lambda zz, pp: (np.zeros((len(pp), 3)), np.ones(len(pp), np.float32))
  gated = imagined_read(always_held, base, np.zeros(3), False)
  pp = np.zeros((2, 20), np.float32)
  pp[:, 14:17] = [[.2, 0, 0], [.01, 0, 0]]
  steps = []
  # candidate 0: arrives open at step 2 and closes at step 3; candidate 1: sits at the cube with closed fingers
  for arrive, width in ((False, .08), (False, .08), (True, .08), (True, .045), (False, .045)):
    pp[0, 14:17], pp[0, 18], pp[1, 18] = ([.01, 0, 0] if arrive else [.2, 0, 0]), width, .045
    steps.append(gated(None, pp)[1].copy())
  assert np.array_equal(np.array(steps).T, [[0, 0, 0, 1, 1], [0, 0, 0, 0, 0]]), "a grasp needs open fingers arriving, then closing"
  pp[:, 18] = .08
  assert imagined_read(always_held, base, np.zeros(3), True)(None, pp)[1].sum() == 0, "open fingers hold nothing"
  # a cube let go in imagination stays where the fingers held it, whatever Q reads afterwards
  readings = iter([([[.1, .2, .9]], [1.0]), ([[.1, .2, .9]], [1.0]), ([[.3, .4, .8]], [0.0]), ([[.5, .6, .8]], [0.0])])
  fake = lambda zz, pp: tuple(np.asarray(v, np.float32) for v in next(readings))
  gated = imagined_read(fake, {**base, "HELD_NEEDS_CLOSED": False, "GRASP_NEEDS_CONTACT": False}, np.zeros(3), True)
  one = np.zeros((1, 20), np.float32)
  xy = [gated(None, one)[0][0].copy() for _ in range(4)]
  assert np.allclose(xy[2], [.1, .2, .8]) and np.allclose(xy[3], [.1, .2, .8]), xy
  print("switch gripper, held needs contact and closed fingers, a released cube stays put: ok")

  # after letting go at B: one candidate, fingers open, the arm rises only once the fingers are clear; the
  # real reward memory is not changed by the shorter imagined settle count
  released = GoalReward(cfg)
  released.was_grasped = released.ever_lifted = True
  open_p = p.copy()
  open_p[19] = 1.0
  models.read = lambda zz, pp: (np.repeat(goal[None], len(zz), 0), np.zeros(len(zz), np.float32))
  try:
    for width, rise in ((.08, .5), (.045, 0.0)):
      open_p[18] = width
      r = plan_standalone(models, z, open_p, released, goal, rest_z, 50, cfg, {**base, **rule}, np.random.default_rng(4))
      assert len(r["pool"]["scores"]) == 1 and np.allclose(r["action"], [0, 0, rise, 0, 1]), r["action"]
    # the pure version keeps searching after placing (the pull turns into a clearance)
    r = plan_standalone(models, z, open_p, released, goal, rest_z, 50, cfg, base, np.random.default_rng(4))
    assert len(r["pool"]["scores"]) == base["POPULATION"]
    assert released.settle_required == cfg.episode.settle_steps and released.success_radius == cfg.episode.success_radius
  finally:
    del models.read
  assert not released_at_target(GoalReward(cfg), goal, open_p, goal, cfg.episode.success_radius)
  print("after letting go at B: retreat rule (rule row), search with clearance (pure), real memory untouched: ok")
  print("self-check passed (planner logic only: the latent is a stand-in, run a real episode for the rest)")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--self-check", action="store_true")
  if parser.parse_args().self_check:
    self_check()
  else:
    parser.print_help()
