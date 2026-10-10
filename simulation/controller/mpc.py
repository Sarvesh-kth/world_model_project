import copy
import json
import pathlib
import runpy
import time
import types

import numpy as np

# The standalone MPC planner, method "mpc" of controller/control_pipeline.py: the proposal's CEM/MPC
# controller, searching action sequences directly in the world model's imagination with no SAC anywhere.
# Same scoring as jepa_mpc (forecasts(): D imagines, Q reads, GoalReward + R's penalties), so the only
# difference between the two is the SAC prior. All tuning knobs live in controller/mpc_config.py.
#   python -m controller.control_pipeline --methods mpc --test-episodes 1 --headless --out data/mpc_smoke
#   python -m controller.control_pipeline --methods mpc --mpc-config controller/my_copy.py
#   python -m controller.mpc --self-check          (on CPU, no camera: checks the planner logic)

DEFAULT_CONFIG = pathlib.Path(__file__).with_name("mpc_config.py")
REQUIRED = ("HORIZON", "POPULATION", "ELITES", "ITERATIONS", "NOISE_STD", "NOISE_MIN", "NOISE_SMOOTH", "GRIPPER",
            "RULE_CLOSE_DISTANCE", "RULE_RELEASE_RADIUS", "RULE_RELEASE_HEIGHT", "SCORE", "PROGRESS_WEIGHT",
            "DISCOUNT", "PENALTY_SCALE", "SAVE_RUNNER_UPS", "REACH_PULL", "HELD_NEEDS_CLOSED", "PLACE_SLOPE",
            "PLACED_AFTER", "RETREAT_AFTER_PLACE", "CUBE_STAYS_PUT", "CARRY_HEIGHT")


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
  if not (0 <= s["PLACE_SLOPE"] < 1 and int(s["PLACED_AFTER"]) == s["PLACED_AFTER"] >= 1
          and 0 <= s["CARRY_HEIGHT"] <= .01 + .1 * s["PLACE_SLOPE"]):
    raise ValueError(f"{path}: 0 <= PLACE_SLOPE < 1, PLACED_AFTER a whole number >= 1, "
                     f"0 <= CARRY_HEIGHT <= 0.01 + 0.1 * PLACE_SLOPE (the glide slope's height 10 cm from B)")
  if s["GRIPPER"] not in ("sampled", "rule") or s["SCORE"] not in ("reward", "progress"):
    raise ValueError(f"{path}: GRIPPER must be 'sampled' or 'rule', SCORE 'reward' or 'progress'")
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


# One planning step. Returns the action, the whole chosen plan (the next step's warm start) and the same
# bookkeeping fields plan() returns, so episode() logs both planners the same way.
def plan_standalone(models, z, p, memory, goal, rest_z, step, cfg, settings, rng, previous=None):
  from .control_pipeline import forecasts
  s = settings
  horizon = min(s["HORIZON"], cfg.episode.max_steps - step)
  command = 1.0 if p[19] > 0 else -1.0
  xyz, held = models.read(z[None], p[None])
  xyz, held = xyz[0], float(held[0])

  # in imagination the cube counts as placed after PLACED_AFTER settled steps instead of the task's 15, so
  # letting go at B can pay off inside the horizon; the real success test is unchanged
  imagined = copy.deepcopy(memory)
  imagined.settle_required = min(memory.settle_required, s["PLACED_AFTER"])

  # forecasts() only needs a discount from the policy; there is no critic here (terminal weight 0)
  scorer = types.SimpleNamespace(gamma=s["DISCOUNT"])

  def score(actions):
    pool = forecasts(models, scorer, z, p, actions, imagined, goal, rest_z, step, cfg, 0.0)
    extra = np.zeros(len(actions), np.float32)
    if s["SCORE"] == "progress":
      # also pay the task potential of every imagined state: getting there early is worth more
      extra += s["PROGRESS_WEIGHT"] * pool["potentials"].mean(1)
    if s["REACH_PULL"]:
      gripper, cube, carried = pool["p"][:, 1:, 14:17], pool["xyz"][:, 1:], pool["held"][:, 1:] > .5
      across = np.linalg.norm(cube[..., :2] - goal[:2], axis=-1)
      # a carried cube may be at most PLACE_SLOPE x its distance to B (+1 cm) above its resting height, so it
      # comes down as it arrives instead of hovering over B; with a slope under 1 getting closer always pays
      above = np.maximum(0, cube[..., 2] - (goal[2] + .01 + s["PLACE_SLOPE"] * across))
      # and while still more than 10 cm from B it should be at least CARRY_HEIGHT up: a placement only counts if
      # the cube was lifted 4 cm, and a strong pull alone drags it along low
      below = np.where(across > .1, np.maximum(0, goal[2] + s["CARRY_HEIGHT"] - cube[..., 2]), 0)
      # an unheld cube does not move: with CUBE_STAYS_PUT the reach part measures to Q's reading of the current
      # frame instead of the imagined cube, which off the expert route drifts along with the arm
      target = np.broadcast_to(xyz, cube.shape) if s["CUBE_STAYS_PUT"] else cube
      far = np.where(carried, across + above + below, np.linalg.norm(gripper - target, axis=-1))
      extra -= s["REACH_PULL"] * far.mean(1)
    pool["scores"] = np.where(pool["valid"], pool["scores"] + extra, -1e9)
    return pool

  saved_scale, models.penalty_scale = models.penalty_scale, s["PENALTY_SCALE"]
  if s["HELD_NEEDS_CLOSED"]:
    # an instance attribute shadows WorldModels.read for this planning step only (undone below)
    shadowed, plain_read = "read" in vars(models), models.read
    def read(zz, pp):
      xyz, held = plain_read(zz, pp)
      return xyz, held * (np.asarray(pp)[:, 18] < .06)
    models.read = read
  try:
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

      best = None
      for _ in range(s["ITERATIONS"]):
        noise = rng.normal(size=(s["POPULATION"], horizon, 5))
        if s["NOISE_SMOOTH"] > 0:
          noise = np.sqrt(s["NOISE_SMOOTH"]) * rng.normal(size=(s["POPULATION"], 1, 5)) + np.sqrt(1 - s["NOISE_SMOOTH"]) * noise
        actions = np.clip(mean + std * noise, -1, 1).astype(np.float32)
        # candidate 0 is the current mean itself, so a good plan can simply be continued
        actions[0] = np.clip(mean, -1, 1)
        actions[..., -1] = rule if rule is not None else np.where(actions[..., -1] > 0, 1, -1)
        pool = score(actions)
        selected = int(np.argmax(pool["scores"]))
        if best is None or pool["scores"][selected] > best["pool"]["scores"][best["selected"]]:
          best = {"pool": pool, "selected": selected}
        elites = actions[np.argsort(pool["scores"])[-s["ELITES"]:]]
        mean, std = elites.mean(0), np.maximum(elites.std(0), floor)
  finally:
    models.penalty_scale = saved_scale
    if s["HELD_NEEDS_CLOSED"]:
      if shadowed:
        models.read = plain_read
      else:
        del models.read

  pool, selected = best["pool"], best["selected"]
  fallback = not bool(pool["valid"][selected])
  if fallback:
    # every candidate left the physical range in imagination: hold still and keep the gripper as it is
    action = np.array([0, 0, 0, 0, command], np.float32)
    sequence = np.tile(action, (horizon, 1))
  else:
    action, sequence = pool["actions"][selected, 0], pool["actions"][selected]
  return {"pool": pool, "selected": selected, "fallback": fallback, "guide_score": None,
          "best_score": float(pool["scores"][selected]), "kept_guide": False, "action": action, "sequence": sequence,
          "overlay": overlay_record(pool, selected, s["SAVE_RUNNER_UPS"], s["HORIZON"])}


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


# The planner's world model on the CPU without the live encoder (which needs CUDA): D, Q and R only.
# For offline checks; the controllers use WorldModels with the encoder.
def offline_models(models_run="data/combined_test1", tag="combined", penalty_scale=2.0):
  import torch
  from world_model.train import load_model
  from .control_pipeline import WorldModels
  models = WorldModels.__new__(WorldModels)
  models.torch = torch
  root = pathlib.Path(models_run)
  models.d, models.dc = load_model(root, "dynamics", tag)
  models.q, models.qc = load_model(root, "readout", tag)
  models.r = models.rc = None
  if (root / "attempts" / tag / "models/reward.pt").exists():
    models.r, models.rc = load_model(root, "reward", tag)
  models.penalty_scale, models.width_gate = penalty_scale, True
  return models


# Planner logic check on the CPU: real D / Q / R and a real robot state from the simulator, but a stand-in
# latent (D's mean latent) because the live encoder needs CUDA. Checks shapes, the gripper modes, the warm
# start, the scoring options, the fallback, that the penalty scale is restored, and the time per step.
def self_check():
  from rl.task_control import GoalReward, TaskSession
  from .control_pipeline import run_settings
  cfg, rest_z = run_settings("data/combined_test1", "combined")
  layout = json.loads(pathlib.Path("configs/grade_e_layout.json").read_text())
  session = TaskSession(cfg, layout, .01)
  session.reset(20494010)
  p = session.obs["proprio"].copy()
  session.close()
  models = offline_models()
  z = models.dc["z_mean"].numpy().astype(np.float32)
  goal = np.array([*layout["place"], rest_z], np.float32)
  base = load_settings()

  plain = {"GRIPPER": "sampled", "REACH_PULL": 0.0, "CUBE_STAYS_PUT": False, "HELD_NEEDS_CLOSED": False, "PLACED_AFTER": 15, "RETREAT_AFTER_PLACE": False}
  for variant in ({}, plain, {"SCORE": "progress"}, {"GRIPPER": "sampled"}, {"CUBE_STAYS_PUT": not base["CUBE_STAYS_PUT"]}):
    s = {**base, **variant}
    memory, previous, rng = GoalReward(cfg), None, np.random.default_rng(0)
    started = time.monotonic()
    for step in range(3):
      d = plan_standalone(models, z, p, memory, goal, rest_z, step, cfg, s, rng, previous)
      previous = d["sequence"]
      assert d["action"].shape == (5,) and d["sequence"].shape == (s["HORIZON"], 5)
      assert np.all(np.abs(d["sequence"]) <= 1) and set(np.unique(d["sequence"][:, -1])) <= {-1.0, 1.0}
      assert d["overlay"]["chosen_gripper"].shape == (s["HORIZON"] + 1, 3) and not d["kept_guide"]
      assert models.penalty_scale == 2.0, "penalty scale was not restored"
      assert "read" not in vars(models), "the held gate was not removed"
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

  # every candidate invalid: hold still and keep the gripper
  import controller.control_pipeline as cp
  real = cp.forecasts
  def all_invalid(*args):
    pool = real(*args)
    pool["valid"][:] = False
    pool["scores"][:] = -1e9
    return pool
  cp.forecasts = all_invalid
  try:
    f = plan_standalone(models, z, p, GoalReward(cfg), goal, rest_z, 0, cfg, base, np.random.default_rng(3))
  finally:
    cp.forecasts = real
  assert f["fallback"] and np.array_equal(f["action"][:4], np.zeros(4)) and f["action"][4] == (1.0 if p[19] > 0 else -1.0)
  print("same seed same plan, short horizon at the episode end, fallback: ok")

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
      r = plan_standalone(models, z, open_p, released, goal, rest_z, 50, cfg, base, np.random.default_rng(4))
      assert len(r["pool"]["scores"]) == 1 and np.allclose(r["action"], [0, 0, rise, 0, 1]), r["action"]
    assert released.settle_required == cfg.episode.settle_steps and models.read.__name__ == "<lambda>"
  finally:
    del models.read
  assert not released_at_target(GoalReward(cfg), goal, open_p, goal, cfg.episode.success_radius)
  print("retreat after letting go at B, real memory untouched: ok")
  print("self-check passed (planner logic only: the latent is a stand-in, run a real episode for the rest)")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--self-check", action="store_true")
  if parser.parse_args().self_check:
    self_check()
  else:
    parser.print_help()
