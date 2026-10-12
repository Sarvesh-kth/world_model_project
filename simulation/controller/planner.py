import copy

import numpy as np

from data_collection.route import plan_route, segment_ok

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
    # the ObstacleMap, set by the controller (None = no geometric check), and the route point being followed
    # (kept until reached, so A* flipping between two ways round does not flip the hand with it)
    self.obstacles = None
    self.route = {}
    # the obstacle-only depth map (env.obstacle_map, robot and cube pixels dropped) kept as a running maximum
    # over the episode by the controller, for the learned obstacle model (--obstacle-source model)
    self.obstacle_map = None
    # the learned cost-to-go fields of the current decision (--cost-to-go), None without
    self.fields = None

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


# The obstacles as the depth camera measures them: the table height map (cells x cells, highest point per cell
# above the table top), updated every step in the cells the hand and the cube are not in: the hand = a disc
# of HAND_MASK around the grasp site while the hand is low (its body and the wrist camera mount reach 13 cm
# from the site and 10-20 cm up), a small disc when it is high; the cube = within CUBE_RADIUS of Q's reading.
# The rest of the arm is above every obstacle (the tallest is 19 cm) and is discarded by the CLIP rule: a
# reading at or above CLIP is never an obstacle. Masking the whole arm segment instead hid the obstacles
# under it for the whole episode. Obstacles do not move and the arm does, so a cell counts as an obstacle
# only after STABLE consecutive readings that agree within 2 cm; masked cells keep their committed value.
# The learned penalty model R stays
# for what it predicts well (contacts, fingers closed
# on nothing); this adds a geometric check that does not depend on D imagining the obstacle geometry 8 steps
# ahead. A real robot with a depth camera does the same. clearance() gives, for a batch of imagined grasp-site
# positions, how far the hand's lowest point (the fingertips, or the bottom of a carried cube: HANG below the
# site) would sink below the top of any cell within HAND_RADIUS plus CLEARANCE_MARGIN.
class ObstacleMap:
  # distances are from a point to the cell's rectangle, not to its centre (a cell is 8 x 11.5 cm)
  HAND_RADIUS = .06
  HANG = .045
  CLEARANCE_MARGIN = .03
  HAND_MASK = .17
  HAND_MASK_HIGH = .08
  HAND_LOW = .20
  CUBE_RADIUS = .05
  STABLE = 3
  CLIP = .20

  def __init__(self, heightmap, cfg, hand_xyz, cube_xy):
    cells = int(round(len(np.asarray(heightmap).reshape(-1)) ** .5))
    centre, half = np.asarray(cfg.table.center, np.float32), np.asarray(cfg.table.half_size, np.float32)
    xs = centre[0] - half[0] + (np.arange(cells) + .5) * 2 * half[0] / cells
    ys = centre[1] - half[1] + (np.arange(cells) + .5) * 2 * half[1] / cells
    self.centres = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    self.half_cell = half / cells
    self.heights = np.zeros(cells * cells, np.float32)
    self.recent = np.full((self.STABLE, cells * cells), np.nan, np.float32)
    self.table = float(cfg.table.height)
    self.base = np.asarray(cfg.robot.base_pos[:2], np.float32)
    self.occupied = np.zeros(0, int)
    self.update(heightmap, hand_xyz, cube_xy)

  # Distance from points (n, 2) to every cell's rectangle, (n, cells)
  def box_distance(self, points):
    gap = np.abs(np.asarray(points, np.float32)[:, None, :] - self.centres[None]) - self.half_cell
    return np.linalg.norm(np.clip(gap, 0, None), axis=2)

  # One more frame: the live map in every cell the hand and the cube are not in; a cell's committed height
  # changes only when the last STABLE readings there agree
  def update(self, heightmap, hand_xyz, cube_xy):
    hand_xyz = np.asarray(hand_xyz, np.float32)
    radius = self.HAND_MASK if hand_xyz[2] - self.table < self.HAND_LOW else self.HAND_MASK_HIGH
    hand = np.linalg.norm(self.centres - hand_xyz[:2], axis=1) < radius
    cube = self.box_distance(np.asarray(cube_xy, np.float32)[None])[0] < self.CUBE_RADIUS
    live = np.asarray(heightmap, np.float32).reshape(-1)
    reading = np.where(hand | cube | (live >= self.CLIP), np.nan, live)
    self.recent = np.roll(self.recent, 1, axis=0)
    self.recent[0] = reading
    stable = np.isfinite(self.recent).all(0) & (np.ptp(self.recent, axis=0) < .02)
    self.heights[stable] = self.recent[:, stable].mean(0)
    self.occupied = np.flatnonzero(self.heights > .01)

  # The next point of a route around the occupied cells from start to target (both xy), or the target itself
  # when the straight line is clear: A* on the table (data_collection/route.py, the same planner the scripted
  # policy uses on the true obstacles), each occupied cell a circle of ROUTE_RADIUS; the first route point
  # more than 6 cm away is returned, so the MPC gets a travel proposal that leads around the obstacle while
  # its 8-step horizon cannot see past it
  ROUTE_RADIUS = .12

  def route_point(self, start_xy, target_xy, cfg, memory=None, key=None):
    return route_point(self.centres[self.occupied], self.ROUTE_RADIUS, start_xy, target_xy, cfg, memory, key)

  # How far below the obstacle tops the hand would be, metres >= 0, for a batch of grasp-site positions (n, 3)
  def clearance(self, ee):
    ee = np.asarray(ee, np.float32)
    if not len(self.occupied):
      return np.zeros(len(ee), np.float32)
    near = self.box_distance(ee[:, :2])[:, self.occupied] < self.HAND_RADIUS
    tops = self.table + self.heights[self.occupied] + self.CLEARANCE_MARGIN
    intrusion = np.where(near, tops[None] - (ee[:, 2:3] - self.HANG), 0)
    return np.clip(intrusion.max(1), 0, None).astype(np.float32)


# The next point of an A* route from start to target (both xy) around the given obstacle centres (circles of
# radius), or the target itself when the straight line is clear; the first route point more than 6 cm away.
# A point already being followed (memory.route[key]) stays while it is more than 6 cm away, the straight line
# to it is clear and the target has not moved much (the cube's reading jitters, the goal does not)
def route_point(centres, radius, start_xy, target_xy, cfg, memory=None, key=None):
  start_xy, target_xy = np.asarray(start_xy, np.float32), np.asarray(target_xy, np.float32)
  if not len(centres):
    return target_xy
  circles = [(c, radius) for c in centres]
  kept = memory.route.get(key) if memory is not None else None
  if kept is not None and np.linalg.norm(kept[0] - start_xy) > .06 and np.linalg.norm(kept[1] - target_xy) < .04 \
      and segment_ok(start_xy, kept[0], circles):
    return kept[0]
  centre, half = np.asarray(cfg.table.center, np.float32), np.asarray(cfg.table.half_size, np.float32)
  base = np.asarray(cfg.robot.base_pos[:2], np.float32)
  points = plan_route(start_xy, target_xy, circles, centre - half + .04, centre + half - .04, (base, cfg.task.reach_range[1]))
  out = target_xy
  for point in points:
    if np.linalg.norm(point - start_xy) > .06:
      out = np.asarray(point, np.float32)
      break
  if memory is not None:
    memory.route[key] = (out, target_xy)
  return out


# The centres of the occupied cells of an obstacle-only map (cells x cells, metres above the table), (n, 2)
def occupied_centres(obstacle_map, cfg):
  cells = len(obstacle_map)
  centre, half = np.asarray(cfg.table.center, np.float32), np.asarray(cfg.table.half_size, np.float32)
  idx = np.argwhere(np.asarray(obstacle_map) > .01)
  return centre - half + (idx + .5) * 2 * half / cells



# Stage potential of a batch of readings, monotone over the stages so the planner is never indifferent:
#   reaching   0.5 * (1 - tanh(5 |hand - cube|))                           0 .. 0.5
#   held       1 + 0.5 * lift/4cm + (1 - tanh(3 |cube - goal|))            1 .. 2.5
#   placed     3.5   lifted before, now released within the success radius at rest height
def potential(p, xyz, held, lifted, goal, rest_z, radius):
  reach = .5 * (1 - np.tanh(5 * np.linalg.norm(p[:, 14:17] - xyz, axis=1)))
  lift = np.clip((xyz[:, 2] - rest_z) / .04, 0, 1)
  to_goal = np.linalg.norm(xyz[:, :2] - goal[:2], axis=1)
  carry = 1 + .5 * lift + (1 - np.tanh(3 * to_goal))
  # imagined "placed" needs the cube well inside the success radius: Q reads a just-released cube under the
  # hand 2-4 cm off (toward the hand) and D drifts, so at 0.6 of the radius releases 8-9 cm out still looked
  # placed (three of the v3 failures ended 7.8-8.9 cm from the goal, radius 7)
  placed = lifted & (held < .5) & (to_goal < .4 * radius) & (np.abs(xyz[:, 2] - rest_z) < .02)
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
# 0.1 over 60 cm toward the cube while it is not held, 0.8 over 80 cm toward the goal while it is. Without
# it, 8 steps of carry 50 cm from the goal gain 0.05 of potential and the planner retreats from any imagined
# penalty instead of going around; at 0.3 the gain of a carry (0.06 per 8 steps, 16 cm) barely cleared the
# 0.05 commitment margin and one v3 episode held the cube at the pick spot for 260 steps (potential 1.62,
# nothing beat "previous"). R's terms are off by default because its
# "placed" flag fires on imagined states with the hand high and the gripper opening (the post-placement
# retreat in the data), which pulled the planner toward the ceiling.
# With --cost-to-go (fields = the learned cost-to-go fields for the cube and the goal, planner.plan computes
# them once per decision) both pulls use the path length around the obstacles instead of the straight line.
def task_potential(models, terms, p, xyz, held, lifted, goal, rest_z, radius, fields=None):
  if fields is None:
    distance = np.linalg.norm(p[:, 14:17] - xyz, axis=1)
  else:
    distance = np.hypot(models.cost_to_go.lookup(fields["cube"], p[:, 14:16]), p[:, 16] - xyz[:, 2])
  pull = .1 * (held < .5) * (1 - np.clip(distance, 0, .6) / .6)
  if models.no_goal:
    # --no-goal: the task's stages come from R's terms read off the image (the place marker is in the frame);
    # nothing here knows where the goal is, only the pull toward Q's cube (a model output) stays
    return learned_potential(terms) + pull
  from_q = potential(p, xyz, held, lifted, goal, rest_z, radius)
  if models.reward_source == "readout" or terms is None:
    phi = from_q
  else:
    from_r = learned_potential(terms)
    phi = from_r if models.reward_source == "head" else .5 * (from_q + from_r)
  to_goal = np.linalg.norm(xyz[:, :2] - goal[:2], axis=1) if fields is None else models.cost_to_go.lookup(fields["goal"], xyz[:, :2])
  return phi + pull + .8 * (held >= .5) * (1 - np.clip(to_goal, 0, .8) / .8)


# Aligned enough to start descending, and close enough to the waypoint to count as arrived and switch the
# gripper. Measured on the planner's own closes: within 1.5 cm of the cube 92-100 % hold, 1.5-2.5 cm about
# half, beyond that none; at 1.2 cm the imagined hand (Q 1 cm off, D drifting) often never "arrived" and
# the grasp proposal never got its close
# aligned within ARRIVE before the descent starts. 1.5 cm was tried against descents that land a finger on
# the cube (the open fingers are 8 cm apart, the cube 4.5): the imagined hand then rarely "arrived", the planner
# closed on sampled switch steps instead and missed more (2.6 closes per grasp vs 1.5), so 2 cm stays
ARRIVE = .02
CLOSE = .02
# the grasp site never goes below this height above the table top: lower and the fingertips press into the
# table and cannot close (the planner's grasps 1.5 cm below the cube centre sat for 14 steps with the fingers
# pinned open at 8.5 cm, until the hand drifted up). The scripted policy keeps the same floor.
FLOOR = .022
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
  # the hand stands still while the fingers close on something (closing on the move slides the cube)
  actions[switched & (grip_after < 0) & (grip_now > 0)] = 0
  actions[:, 4] = np.where(switched, grip_after, grip_now)
  return actions


# Imagine the candidates with D, read them with Q and R, and score them. Returns everything the csv needs.
def forecasts(models, z, p, waypoints, switch, grip_after, memory, goal, rest_z, cfg, args):
  n, horizon = len(waypoints), args.horizon
  weights = cfg.rewards.weights
  zs, ps = np.repeat(z[None], n, 0), np.repeat(p[None], n, 0)
  grip_now = np.full(n, 1.0 if p[19] > 0 else -1.0, np.float32)
  lifted = np.full(n, memory.ever_lifted)
  # the gripper envelope (on by default, --no-grasp-prior): credit a hold only if the hand was at the cube when
  # the fingers closed. D learned the recorded closes (held 1.0 on grasps, 0.26 on near misses) but it still
  # imagines a hold for a close made mid-travel far from the cube, a case the data does not contain, and such
  # candidates outscored the real grasp (1.9 vs 0.5) so the hand hovered beside the cube for 300 steps
  xyz_now, held_read = models.read(z[None], p[None])
  xyz0 = memory.cube(xyz_now[0], float(held_read[0]), p[14:17])[None]
  grasp_ok = np.full(n, float(held_read[0]) >= .5)
  was_closed = grip_now < 0
  actions = np.zeros((n, horizon, 5), np.float32)
  penalties = np.zeros((n, horizon), np.float32)
  clearance = np.zeros((n, horizon), np.float32)
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
    # the open gripper measures up to .083 and the closed one reads 0; D overshoots both (the v3 D by up to a
    # centimetre on opening, which at a .09 bound made the release proposal invalid and the hand hover at the
    # goal holding the cube), so only clearly impossible widths are rejected
    valid &= (finite & np.isfinite(xyz).all(1) & np.isfinite(held) & (pp[:, 18] >= -.02) & (pp[:, 18] <= .12)
              & (np.abs(pp[:, 14:17]) < 5).all(1) & (np.abs(xyz) < 5).all(1) & (np.abs(pp) < 100).all(1) & (np.abs(zz) < 1e4).all(1))
    lifted = lifted | ((held >= .5) & (xyz[:, 2] - rest_z >= .04))
    terms = models.reward(zz, pp)
    penalties[:, t] = models.penalty(terms, weights)
    # the geometric check: an imagined hand sinking into a measured obstacle costs up to --clearance-weight
    # per step (the full weight at 5 cm of intrusion)
    if args.obstacle_source == "map" and memory.obstacles is not None and args.clearance_weight > 0:
      clearance[:, t] = memory.obstacles.clearance(pp[:, 14:17])
      penalties[:, t] -= args.clearance_weight * np.clip(clearance[:, t] / .05, 0, 1)
    elif args.obstacle_source == "model" and memory.obstacle_map is not None:
      # the all-model alternative: the obstacle model reads a window of the obstacle-only depth map around the
      # IMAGINED hand, with the imagined gripper and held reading (world_model/train_obstacle.py); no
      # clearance formula, no route
      penalties[:, t] += models.obstacle_penalty(memory.obstacle_map, pp, held, weights)
    potentials[:, t] = task_potential(models, terms, pp, xyz, held, lifted, goal, rest_z, cfg.episode.success_radius, memory.fields)
    zs, ps = zz, pp
    xyz_path.append(xyz)
    held_path.append(held)

  # terminal potential, the way there (lightly), the penalties (R's, including fingers imagined shut on
  # nothing) and a little cost on moving
  scores = (potentials[:, -1] + .1 * potentials.mean(1) + penalties.sum(1)
            - args.action_cost * np.square(actions[:, :, :3]).sum((1, 2)))
  valid &= np.isfinite(scores)
  scores[~valid] = -1e9
  return {"actions": actions, "scores": scores.astype(np.float32), "valid": valid, "penalties": penalties, "clearance": clearance,
          "potentials": potentials, "xyz": np.stack(xyz_path, 1), "held": np.stack(held_path, 1), "final_p": ps}


# The fixed proposals for the current reading, each a (waypoint, switch step, gripper after the switch).
# With an obstacle map, the travel proposals (to the cube while it is not held, the carry while it is) aim at
# the next point of a route around the measured obstacles instead of straight at the target
def proposals(p, xyz, held, goal, rest_z, cfg, args, previous, memory=None):
  ee = p[14:17]
  grip = 1.0 if p[19] > 0 else -1.0
  horizon = args.horizon
  floor_z = cfg.table.height + FLOOR
  grasp_point = np.array([xyz[0], xyz[1], max(xyz[2] - .015, floor_z)])
  carry_z = rest_z + args.carry_height
  travel = lambda target, key: np.asarray(target[:2])
  if memory is not None and args.obstacle_source == "map" and memory.obstacles is not None:
    travel = lambda target, key: memory.obstacles.route_point(ee[:2], target, cfg, memory, key)
  elif memory is not None and args.obstacle_source == "model" and args.route_proposals and memory.obstacle_map is not None:
    # --route-proposals: the route only suggests where the travel proposals aim; the scoring stays with the
    # learned obstacle model and R (an 8-step horizon cannot see that a detour pays off)
    centres = occupied_centres(memory.obstacle_map, cfg)
    travel = lambda target, key: route_point(centres, .10, ee[:2], target, cfg, memory, key)
  to_cube = travel(xyz[:2], "cube")
  via_cube = not np.allclose(to_cube, xyz[:2])
  # --no-goal: the goal-based proposals become "lift and hold here" / "lower here" / "open here"; the search
  # for the goal is left to the sampled waypoints and R's transport and placed terms
  goal = np.array([ee[0], ee[1], rest_z], np.float32) if getattr(args, "no_goal", False) else goal
  to_goal = travel(goal[:2], "goal")
  out = {"hold": (ee, horizon, grip),
         "previous": previous if previous is not None else (ee, horizon, grip),
         "to_cube": (np.array([to_cube[0], to_cube[1], carry_z]) if via_cube else grasp_point, horizon, 1.0),
         "grasp": (grasp_point, ON_ARRIVAL, -1.0),
         "above_cube": (np.array([xyz[0], xyz[1], xyz[2] + .10]), horizon, 1.0),
         "carry": (np.array([to_goal[0], to_goal[1], carry_z]), horizon, -1.0),
         "lower": (np.array([goal[0], goal[1], max(rest_z - .015, floor_z)]), horizon, -1.0),
         "release": (np.array([goal[0], goal[1], max(rest_z - .015, floor_z)]), 0, 1.0)}
  return [out[name] for name in PROPOSALS]


# CEM over waypoints around the proposals. Returns the chosen first action, the plan for the warm start
# and the bookkeeping for the csv.
def plan(models, z, p, xyz, held, memory, goal, rest_z, cfg, args, rng, previous=None):
  horizon, n = args.horizon, args.population
  xyz = memory.cube(xyz, held, p[14:17])
  low = np.asarray(cfg.control.workspace.low) + np.array([0, 0, .005])
  high = np.asarray(cfg.control.workspace.high)
  memory.fields = None
  if models.cost_to_go is not None and memory.obstacle_map is not None:
    memory.fields = {"cube": models.cost_to_go.field(memory.obstacle_map, xyz[:2]), "goal": models.cost_to_go.field(memory.obstacle_map, goal[:2])}
  fixed = proposals(p, xyz, held, goal, rest_z, cfg, args, previous, memory)
  k = len(fixed)
  mean = np.asarray(previous[0] if previous is not None else p[14:17], np.float32)
  # the spread scales with how far the hand is from what it has to reach (the cube, or the goal once it is
  # held): wide for the search, about a centimetre for the final approach, where 3 cm of scatter is the
  # difference between a grasp and a miss
  focus = (goal if not args.no_goal else p[14:17]) if held >= .5 else xyz
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
    best["action"] = pool["actions"][i, 0].copy()
    s = int(best["switch"][i])
    # a held cube is released only with the real hand within half the success radius of the goal: a plan whose
    # switch step came up while the hand was still on its way opened 4-5 cm short of the goal (the imagined
    # hand had "arrived"), and candidates 5 cm from the goal all score "placed" so the hand dithered there.
    # Otherwise this step goes to the goal, low, fingers closed, and the plan becomes "open on arrival there".
    # Only while Q reads the cube as held: fingers closed on nothing must open again to retry (one episode
    # kept an empty gripper shut for 220 steps). The height is left to the planner (a cube dropped from the
    # carry height lands fine; requiring it low made the hand hover at the goal while fresh candidates kept
    # winning at carry height).
    if not args.no_goal and not args.no_release_rule and held >= .5 and grip_now < 0 and best["action"][4] > 0 and np.linalg.norm(goal[:2] - p[14:16]) > .5 * cfg.episode.success_radius:
      waypoint = np.array([goal[0], goal[1], max(rest_z - .015, cfg.table.height + FLOOR)], np.float32)
      best["action"] = goto_actions(waypoint[None], np.array([ON_ARRIVAL]), np.array([1.0], np.float32), np.array([-1.0], np.float32),
                                    p[None, 14:17], 0, cfg.control.max_delta, rest_z + args.carry_height)[0]
      best["action"][4] = -1.0
      best["waypoints"][i] = waypoint
      s = ON_ARRIVAL
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
  p.add_argument("--penalty-scale", type=float, default=1.0, help="multiplier on R's imagined penalties (proximity, collision, table hit, empty grasp). 2 when R was the only obstacle signal; with the obstacle map's clearance check doing the geometry, R's proximity (0.32 off when it matters) at 2 pushed carries around for 200 steps near a single box")
  p.add_argument("--no-goal", action="store_true",
                 help="the goal position is NOT given to the planner: the task potential is R's terms read off the image (the green place marker is in the frame, --reward-source head is implied), the goal-based proposals, the goal pull and the release-at-the-goal rule are off")
  p.add_argument("--obstacle-source", choices=("map", "model", "none"), default="model",
                 help="where the obstacle signal comes from: map = the depth height map accumulated over the episode (clearance penalty on the imagined hand + A* route proposals); model = the learned obstacle model (train_obstacle.py) on a window of the obstacle-only depth map (robot and cube pixels dropped) around the IMAGINED hand, no clearance formula and no route; none = R's penalties only")
  p.add_argument("--cost-to-go", action="store_true",
                 help="with --obstacle-source model: the learned cost-to-go of world_model/train_cost_to_go.py (a convolution over the obstacle-only map that outputs the path length to a target from everywhere on the table) replaces the straight-line distance in the reach and goal pulls, so a detour round an obstacle is credited within the 8-step horizon; no route is planned")
  p.add_argument("--route-proposals", action="store_true",
                 help="with --obstacle-source model: aim the to_cube / carry proposals at the next point of an A* route around the occupied cells of the obstacle-only map; the route proposes, the learned models still score (the 8-step MPC alone parks next to an obstacle when no candidate gains goal distance within the horizon)")
  p.add_argument("--clearance-weight", type=float, default=1.0, help="geometric check: cost per imagined step of the hand sinking 5 cm into an obstacle of the depth height map (0 turns it off); R's learned penalties alone let the hand into walls (257 contact steps over 10 v3 scenes)")
  p.add_argument("--reward-source", choices=("both", "head", "readout"), default="readout",
                 help="task potential from R's predicted terms, from Q's readings through the formula, or their mean")
  p.add_argument("--no-penalties", action="store_true", help="ignore the reward model R entirely (potential from Q only)")
  p.add_argument("--no-width-gate", action="store_true", help="do not zero Q's held reading when the fingers are closed on nothing")
  p.add_argument("--no-grasp-prior", dest="grasp_prior", action="store_false",
                 help="drop the gripper envelope: by default an imagined hold is credited only if the hand was within 1.5 cm (xy) / 2 cm (z) of Q's cube when the fingers closed; D imagines a hold for fingers closing mid-travel 14 cm from the cube (no such closes in the data), which otherwise outscores the real grasp")
  p.add_argument("--no-release-rule", action="store_true",
                 help="drop the release rule: by default a held cube is released only with the real hand within half the success radius of the goal, otherwise the step goes to the goal closed and opens on arrival (plans whose switch step came up early opened 4-9 cm short; Q reads a just-released cube toward the hand)")
  p.add_argument("--cube-anchor", action="store_true", help="old stop-gap: aim at the last clear sighting of the cube while the hand is within 8 cm of it")
  p.add_argument("--blind", action="store_true", help="use the proprio-only readout_p instead of Q(z, p)")
