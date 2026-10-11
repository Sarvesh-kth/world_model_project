import copy

import numpy as np

# The latent MPC, nothing here touches the simulator or a learned policy. Every control step it
#   1. proposes candidates in Cartesian space: a waypoint for the grasp site, when the gripper command
#      switches (at a step, or on arrival at the waypoint) and the command after the switch (open or closed);
#      the actions along the horizon come from a go-to rule: travel at a safe height, descend only once
#      aligned within 2 cm, rise first when holding something, full speed and slowing inside one step, yaw untouched
#   2. imagines every candidate with D from the current (z, p), reading each imagined state with Q (cube
#      position, held) and the reward model R (the shaped task terms reach / grasp / lift / transport,
#      placed, and the proximity / collision / table-hit / empty-grasp penalties)
#   3. scores them: the task potential at the end of the horizon (from R's terms, from Q's readings through
#      the same formula, or both, --reward-source) plus the imagined penalties along the way, minus a small
#      action cost, and runs CEM on the waypoints for a few iterations
#   4. executes the first action of the best candidate and keeps it as the warm start for the next step
# A fixed set of proposals (go to the cube, grasp, carry, lower, release, hold, the previous plan) is always
# in the population; they are scored like every sampled candidate, nothing is executed unscored.

PROPOSALS = ("hold", "previous", "to_cube", "grasp", "above_cube", "carry", "lower", "release")


# What the planner remembers about the task between steps, from Q's readings only (never the true state):
# whether the cube has been held and lifted, so "released at the goal" can be told from "never picked up".
# With --cube-anchor it also keeps the last clear sighting of the cube and aims at that while the hand is
# within 8 cm and nothing is held (a stop-gap from when Q drifted 1-3 cm under the hand; off by default,
# the readout trained on hover data is meant to handle it).
class TaskMemory:
  NEAR = .08

  def __init__(self, use_anchor=False):
    self.was_held = False
    self.ever_lifted = False
    self.anchor = None
    self.use_anchor = use_anchor

  def update(self, xyz, held, rest_z, hand=None):
    if held >= .5:
      self.was_held = True
      self.ever_lifted |= bool(xyz[2] - rest_z >= .04)
    if hand is None:
      return
    if held >= .5 or self.anchor is None or np.linalg.norm(hand - xyz) > self.NEAR:
      self.anchor = np.asarray(xyz, np.float32).copy()

  # Where the planner takes the cube to be right now
  def cube(self, xyz, held, hand):
    if not self.use_anchor or held >= .5 or self.anchor is None or np.linalg.norm(hand - xyz) > self.NEAR:
      return np.asarray(xyz, np.float32)
    return self.anchor


# Stage potential of a batch of readings, monotone over the stages so the planner is never indifferent:
#   reaching   0.5 * (1 - tanh(5 |hand - cube|))                           0 .. 0.5
#   held       1 + 0.5 * lift/4cm + (1 - tanh(3 |cube - goal|))            1 .. 2.5
#   placed     3.5   lifted before, now released within the success radius at rest height
def potential(p, xyz, held, lifted, goal, rest_z, radius):
  reach = .5 * (1 - np.tanh(5 * np.linalg.norm(p[:, 14:17] - xyz, axis=1)))
  lift = np.clip((xyz[:, 2] - rest_z) / .04, 0, 1)
  to_goal = np.linalg.norm(xyz[:, :2] - goal[:2], axis=1)
  carry = 1 + .5 * lift + (1 - np.tanh(3 * to_goal))
  # imagined "placed" needs the cube well inside the success radius: Q reads it 1-2 cm off and D drifts, so at
  # the full radius a release 9 cm out looked placed and the planner opened early
  placed = lifted & (held < .5) & (to_goal < .6 * radius) & (np.abs(xyz[:, 2] - rest_z) < .02)
  return np.where(placed, 3.5, np.where(held >= .5, carry, reach)).astype(np.float32)


# The same potential from the reward model's predicted terms (the environment's own reach / grasp / lift /
# transport in [0, 1] and the placed probability), the stages blended by the predicted grasp and placed
def learned_potential(terms):
  reaching = .5 * terms["reach"]
  carry = 1 + .5 * terms["lift"] + terms["transport"]
  phi = (1 - terms["grasp"]) * reaching + terms["grasp"] * carry
  return ((1 - terms["placed"]) * phi + 3.5 * terms["placed"]).astype(np.float32)


# The potential the planner scores with: Q's readings (default), R's terms, or the mean of both
# (--reward-source). The task's reach and transport terms saturate beyond 30 cm, so a linear pull is added:
# 0.1 over 60 cm toward the cube while it is not held, 0.3 over 80 cm toward the goal while it is; without
# it, 8 steps of carry 50 cm from the goal gain 0.05 of potential and one imagined step near an obstacle
# costs 0.6, so the planner retreats instead of going around. R's terms are off by default because its
# "placed" flag fires on imagined states with the hand high and the gripper opening (the post-placement
# retreat in the data), which pulled the planner toward the ceiling.
def task_potential(models, terms, p, xyz, held, lifted, goal, rest_z, radius):
  from_q = potential(p, xyz, held, lifted, goal, rest_z, radius)
  if models.reward_source == "readout" or terms is None:
    phi = from_q
  else:
    from_r = learned_potential(terms)
    phi = from_r if models.reward_source == "head" else .5 * (from_q + from_r)
  distance = np.linalg.norm(p[:, 14:17] - xyz, axis=1)
  to_goal = np.linalg.norm(xyz[:, :2] - goal[:2], axis=1)
  return (phi + .1 * (held < .5) * (1 - np.clip(distance, 0, .6) / .6)
          + .3 * (held >= .5) * (1 - np.clip(to_goal, 0, .8) / .8))


# Aligned enough to start descending, and close enough to the waypoint to count as arrived and switch the
# gripper. Measured on the planner's own closes: within 1.5 cm of the cube 92-100 % hold, 1.5-2.5 cm about
# half, beyond that none; at 1.2 cm the imagined hand (Q 1 cm off, D drifting) often never "arrived" and
# the grasp proposal never got its close
ARRIVE = .02
CLOSE = .02
# A switch value of -1 means "switch on arrival"
ON_ARRIVAL = -1


# The actions of one imagined step for a batch of candidates: toward the waypoint, gripper per the switch.
# Lateral travel happens at travel_z or higher and the hand descends only when aligned within ARRIVE, so a low
# waypoint is approached from above and the hand never ploughs into the cube or the table; a hand that is
# below the travel height rises before it moves sideways (lifting a held cube clear of the table first).
def goto_actions(waypoints, switch, grip_after, grip_now, ee, t, max_delta, travel_z):
  n = len(waypoints)
  lateral = np.linalg.norm(waypoints[:, :2] - ee[:, :2], axis=1)
  # once the hand is on its way down a small drift of the waypoint must not send it back up
  descending = ee[:, 2] < travel_z - .03
  aligned = (lateral <= ARRIVE) | (descending & (lateral <= 2 * ARRIVE))
  target_z = np.where(aligned, waypoints[:, 2], np.maximum(waypoints[:, 2], travel_z))
  target = np.column_stack((waypoints[:, :2], target_z))
  actions = np.zeros((n, 5), np.float32)
  actions[:, :3] = np.clip((target - ee) / max_delta, -1, 1)
  low = descending & ~aligned
  actions[low, :2] = 0
  arrived = (lateral <= CLOSE) & (np.abs(waypoints[:, 2] - ee[:, 2]) <= CLOSE)
  switched = np.where(switch == ON_ARRIVAL, arrived, t >= switch)
  actions[:, 4] = np.where(switched, grip_after, grip_now)
  return actions


# Imagine the candidates with D, read them with Q and R, and score them. Returns everything the csv needs.
def forecasts(models, z, p, waypoints, switch, grip_after, memory, goal, rest_z, cfg, args):
  n, horizon = len(waypoints), args.horizon
  weights = cfg.rewards.weights
  zs, ps = np.repeat(z[None], n, 0), np.repeat(p[None], n, 0)
  grip_now = np.full(n, 1.0 if p[19] > 0 else -1.0, np.float32)
  lifted = np.full(n, memory.ever_lifted)
  # --grasp-prior (off by default): credit a hold only if the hand was at the cube when the fingers closed,
  # the stop-gap from when D imagined a hold after a close a few centimetres away; D and R trained on
  # near-miss closes and the empty-grasp penalty are meant to make it unnecessary
  xyz_now, held_read = models.read(z[None], p[None])
  xyz0 = memory.cube(xyz_now[0], float(held_read[0]), p[14:17])[None]
  grasp_ok = np.full(n, float(held_read[0]) >= .5)
  was_closed = grip_now < 0
  actions = np.zeros((n, horizon, 5), np.float32)
  penalties = np.zeros((n, horizon), np.float32)
  potentials = np.zeros((n, horizon), np.float32)
  valid = np.ones(n, bool)
  xyz_path, held_path = [], []

  travel_z = rest_z + args.carry_height
  for t in range(horizon):
    actions[:, t] = goto_actions(waypoints, switch, grip_after, grip_now, ps[:, 14:17], t, cfg.control.max_delta, travel_z)
    # once a candidate has switched its gripper the command stays, so a later step cannot flip it back
    grip_now = actions[:, t, 4]
    closing = (actions[:, t, 4] < 0) & ~was_closed
    at_cube = (np.linalg.norm(ps[:, 14:16] - xyz0[0, :2], axis=1) <= .015) & (np.abs(ps[:, 16] - xyz0[0, 2]) <= .02)
    grasp_ok = grasp_ok | (closing & at_cube)
    was_closed = actions[:, t, 4] < 0
    zz, pp = models.predict(zs, ps, actions[:, t])
    finite = np.isfinite(zz).all(1) & np.isfinite(pp).all(1)
    zz[~finite], pp[~finite] = zs[~finite], ps[~finite]
    xyz, held = models.read(zz, pp)
    if models.grasp_prior:
      held = held * grasp_ok
    # the open gripper measures up to .083 and the closed one reads 0, D overshoots both by a few mm which is fine
    valid &= (finite & np.isfinite(xyz).all(1) & np.isfinite(held) & (pp[:, 18] >= -.01) & (pp[:, 18] <= .09)
              & (np.abs(pp[:, 14:17]) < 5).all(1) & (np.abs(xyz) < 5).all(1) & (np.abs(pp) < 100).all(1) & (np.abs(zz) < 1e4).all(1))
    lifted = lifted | ((held >= .5) & (xyz[:, 2] - rest_z >= .04))
    terms = models.reward(zz, pp)
    penalties[:, t] = models.penalty(terms, weights)
    potentials[:, t] = task_potential(models, terms, pp, xyz, held, lifted, goal, rest_z, cfg.episode.success_radius)
    zs, ps = zz, pp
    xyz_path.append(xyz)
    held_path.append(held)

  # terminal potential, the way there (lightly), the penalties (R's, including fingers imagined shut on
  # nothing) and a little cost on moving
  scores = (potentials[:, -1] + .1 * potentials.mean(1) + penalties.sum(1)
            - args.action_cost * np.square(actions[:, :, :3]).sum((1, 2)))
  valid &= np.isfinite(scores)
  scores[~valid] = -1e9
  return {"actions": actions, "scores": scores.astype(np.float32), "valid": valid, "penalties": penalties,
          "potentials": potentials, "xyz": np.stack(xyz_path, 1), "held": np.stack(held_path, 1), "final_p": ps}


# The fixed proposals for the current reading, each a (waypoint, switch step, gripper after the switch)
def proposals(p, xyz, held, goal, rest_z, cfg, args, previous):
  ee = p[14:17]
  grip = 1.0 if p[19] > 0 else -1.0
  horizon = args.horizon
  grasp_point = np.array([xyz[0], xyz[1], xyz[2] - .015])
  carry_z = rest_z + args.carry_height
  out = {"hold": (ee, horizon, grip),
         "previous": previous if previous is not None else (ee, horizon, grip),
         "to_cube": (grasp_point, horizon, 1.0),
         "grasp": (grasp_point, ON_ARRIVAL, -1.0),
         "above_cube": (np.array([xyz[0], xyz[1], xyz[2] + .10]), horizon, 1.0),
         "carry": (np.array([goal[0], goal[1], carry_z]), horizon, -1.0),
         "lower": (np.array([goal[0], goal[1], rest_z - .015]), horizon, -1.0),
         "release": (np.array([goal[0], goal[1], rest_z - .015]), 0, 1.0)}
  return [out[name] for name in PROPOSALS]


# CEM over waypoints around the proposals. Returns the chosen first action, the plan for the warm start
# and the bookkeeping for the csv.
def plan(models, z, p, xyz, held, memory, goal, rest_z, cfg, args, rng, previous=None):
  horizon, n = args.horizon, args.population
  xyz = memory.cube(xyz, held, p[14:17])
  low = np.asarray(cfg.control.workspace.low) + np.array([0, 0, .005])
  high = np.asarray(cfg.control.workspace.high)
  fixed = proposals(p, xyz, held, goal, rest_z, cfg, args, previous)
  k = len(fixed)
  mean = np.asarray(previous[0] if previous is not None else p[14:17], np.float32)
  # the spread scales with how far the hand is from what it has to reach (the cube, or the goal once it is
  # held): wide for the search, about a centimetre for the final approach, where 3 cm of scatter is the
  # difference between a grasp and a miss
  focus = goal if held >= .5 else xyz
  reach = float(np.clip(.5 * np.linalg.norm(p[14:17] - focus), args.waypoint_std_min, 1))
  std = np.minimum(np.array([args.waypoint_std, args.waypoint_std, args.waypoint_std_z], np.float32), reach)
  std_floor = std / 5
  grip_now = 1.0 if p[19] > 0 else -1.0
  p_switch, p_close = .3, .3
  best = None

  for iteration in range(args.iterations):
    waypoints = np.clip(mean + std * rng.normal(size=(n, 3)), low, high).astype(np.float32)
    # a candidate switches the gripper with probability p_switch, half of those on arrival, else never
    switch = np.where(rng.random(n) < p_switch, np.where(rng.random(n) < .5, ON_ARRIVAL, rng.integers(0, horizon, n)), horizon)
    grip_after = np.where(rng.random(n) < p_close, -1.0, 1.0).astype(np.float32)
    for i, (w, s, g) in enumerate(fixed):
      waypoints[i], switch[i], grip_after[i] = np.clip(w, low, high), s, g
    pool = forecasts(models, z, p, waypoints, switch, grip_after, memory, goal, rest_z, cfg, args)
    # commitment: the previous plan (proposal 1) keeps being followed unless a candidate beats it by a
    # margin, otherwise near ties between fresh samples make the hand flip direction every step
    scores = pool["scores"].copy()
    if previous is not None and pool["valid"][1]:
      scores[np.arange(n) != 1] -= args.plan_margin
    selected = int(np.argmax(scores))
    if best is None or scores[selected] > best["score"]:
      best = {"pool": pool, "selected": selected, "score": float(scores[selected]),
              "waypoints": waypoints, "switch": switch, "grip_after": grip_after, "iteration": iteration,
              "proposal_scores": pool["scores"][:k].copy()}
    elites = np.argsort(pool["scores"])[-args.elites:]
    mean = waypoints[elites].mean(0)
    std = np.maximum(waypoints[elites].std(0), std_floor)
    p_switch = float(np.clip((switch[elites] < horizon).mean(), .1, .9))
    p_close = float(np.clip((grip_after[elites] < 0).mean(), .1, .9))

  i = best["selected"]
  pool = best["pool"]
  best["fallback"] = not bool(pool["valid"][i])
  if best["fallback"]:
    # nothing imaginable, stay put with the current gripper
    best["action"] = np.array([0, 0, 0, 0, grip_now], np.float32)
    best["plan"] = None
  else:
    best["action"] = pool["actions"][i, 0]
    s = int(best["switch"][i])
    best["plan"] = (best["waypoints"][i], s if s == ON_ARRIVAL else max(s - 1, 0), float(best["grip_after"][i]))
  best["proposal"] = PROPOSALS[i] if i < k else "sampled"
  return best


# The planner flags, shared by control_pipeline and control_clutter
def arguments(p):
  p.add_argument("--horizon", type=int, default=8, help="steps D imagines ahead; D trains on 8-step windows and beyond them it imagined a carried cube slipping away")
  p.add_argument("--population", type=int, default=64, help="candidates per CEM iteration, the first 8 are the fixed proposals")
  p.add_argument("--elites", type=int, default=8)
  p.add_argument("--iterations", type=int, default=3)
  p.add_argument("--waypoint-std", type=float, default=.15, help="initial CEM spread of the waypoint in x and y, metres")
  p.add_argument("--waypoint-std-z", type=float, default=.06, help="initial CEM spread of the waypoint height, metres")
  p.add_argument("--plan-margin", type=float, default=.05, help="a new candidate replaces the previous plan only if it scores this much higher (commitment, keeps the motion smooth)")
  p.add_argument("--waypoint-std-min", type=float, default=.03, help="the spread shrinks with the distance to the cube (or the goal when holding it) down to this, metres (1 cm traps the hand at a wrong cube estimate while Q still produces those)")
  p.add_argument("--carry-height", type=float, default=.08, help="carry proposal height above the cube's rest height")
  p.add_argument("--action-cost", type=float, default=.0002, help="cost per unit squared action over the horizon (a full-speed horizon costs 0.005)")
  p.add_argument("--penalty-scale", type=float, default=2.0, help="multiplier on R's imagined penalties")
  p.add_argument("--reward-source", choices=("both", "head", "readout"), default="readout",
                 help="task potential from R's predicted terms, from Q's readings through the formula, or their mean")
  p.add_argument("--no-penalties", action="store_true", help="ignore the reward model R entirely (potential from Q only)")
  p.add_argument("--no-width-gate", action="store_true", help="do not zero Q's held reading when the fingers are closed on nothing")
  p.add_argument("--grasp-prior", action="store_true", help="old stop-gap: credit an imagined hold only if the hand was at Q's cube estimate when it closed, and read held only with the cube between the fingers")
  p.add_argument("--cube-anchor", action="store_true", help="old stop-gap: aim at the last clear sighting of the cube while the hand is within 8 cm of it")
  p.add_argument("--blind", action="store_true", help="use the proprio-only readout_p instead of Q(z, p)")
