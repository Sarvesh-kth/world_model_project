import numpy as np

from .route import plan_route, segment_depth


def wrap_angle(a):
  return (a + np.pi) % (2 * np.pi) - np.pi


# Waypoint state machine for pick / carry / place demos
# approach -> hover -> descend -> grasp -> lift -> transport -> lower -> release -> retreat -> done
# the carry always goes around the obstacles using the route planner, unless carry_low is set,
# then it goes straight through them at low height (deliberate collision episodes)
# drop_early lets go of the object halfway through the carry
# miss closes the fingers beside or above the object on the first try, then picks it up properly
# wander visits a few random points anywhere on the table (2-15 cm up, fingers open or shut, each leg routed
# around the tall obstacles) before the pick, half the time after first going to the object and leaving it
# again, and half the time makes one detour while carrying; so the hand is seen everywhere while the object
# sits wherever it is, and carries go in every direction
# lateral grasps the way the planner does: arrives at the object low and from the side, closes with up to
# 2.5 cm of offset (where a grasp still holds, or just slips); half of these episodes turn the fingers to the
# object's faces like every other kind, half keep the yaw they arrived with (the planner does not turn yet)
# The object itself spawns at a random yaw every episode (env._settle), so the turn differs every time.
class ScriptedPickPlace:
  # Hover this far above the object before descending
  HOVER_HEIGHT = 0.12
  # Fingertips this far below the object centre, so the pads grip its middle
  GRASP_DEPTH = 0.015
  # Tolerance for grasp
  GRASP_TOL = 0.012
  # Tolerance for XY position
  XY_TOL = 0.02
  # Steps to keep the gripper command before moving on
  HOLD_STEPS = 8
  # Max retries when the grasp did not take
  MAX_RETRIES = 2
  # No route waypoint closer than this to the place spot
  PLACE_KEEP_OUT = 0.12
  # Lowest the grasp site can go, any lower and the fingertips press into the table
  MIN_EE_HEIGHT = 0.022
  # The palm sits about 3.9 cm above the grasp site, keep the site this far below the object top or the palm lands on it
  PALM_CLEARANCE = 0.032
  # Give up when the arm moved less than STUCK_DIST over the last STUCK_STEPS steps in a moving phase
  STUCK_STEPS = 30
  STUCK_DIST = 0.03
  # Lateral reach of the hand, obstacles this close to the path are under the hand
  HAND_RADIUS = 0.08
  # Height kept between the lowest point of hand + object and an obstacle top when passing over it
  OVER_CLEARANCE = 0.04

  def __init__(self, env, rng, noise=0.0, carry_low=False, drop_early=False, miss=False, wander=False, lateral=False):
    self.env = env
    self.rng = rng
    self.noise = noise
    self.carry_low = carry_low
    self.drop_early = drop_early
    self.miss = miss
    self.wander = wander
    self.lateral = lateral
    self.reset()

  # Nothing left to do, the episode can stop here instead of idling till max_steps
  @property
  def done(self):
    return self.phase == "done"

  # Per episode state, only the speed is randomized
  def reset(self):
    cfg = self.env.cfg
    sc = cfg.data.scripted
    table = cfg.table.height
    self.hold = 0
    self.retries = 0
    # the wander points are drawn when the first one is needed; wander_first = go to the object, then roam
    self.wander_points = None
    self.wander_grip = 1.0
    self.wander_first = self.wander and self.rng.uniform() < 0.5
    self.roamed = not self.wander
    self.carry_detour = self.wander and self.rng.uniform() < 0.5
    self.phase = "approach" if (self.wander_first or not self.wander) else "wander"
    self.route = None
    self.speed = self.rng.uniform(*sc.speed_range)
    self.half_h = max(self.env.object_rest_z - table, 0.02)
    # lowest point of hand + held object below the grasp site (fingertips, or the object bottom for tall objects)
    self.hang = max(0.018, self.half_h - 0.015)
    self.ceiling = cfg.control.workspace.high[2]
    # obstacles taller than this cannot be cleared even at the ceiling, the route goes around them
    self.max_over = self.ceiling - table - self.hang - self.OVER_CLEARANCE
    self.margin = sc.route_margin
    self.noise_state = np.zeros(3)
    self.ee_hist = []
    self.carry_z = table + (0.05 if self.carry_low else sc.carry_height + self.half_h)
    # a deliberately bad first grasp: the descent aims 2.5-7 cm beside the object (across the finger axis,
    # along it a finger lands on the object), 2.5-7 cm above it, or both, so the data has fingers closing on
    # nothing or on an edge right next to the object; the retry after it is clean. 2 cm off still catches
    # the cube, 2.5-4 cm is where the dynamics model has to learn the boundary.
    self.miss_beside = self.miss_above = 0.0
    self.miss_offset = np.zeros(3)
    # the lateral grasp: a low hover height, a small offset on the first try, no finger turning
    self.hover_height = self.HOVER_HEIGHT
    self.lateral_offset = np.zeros(3)
    self.lateral_turn = self.rng.uniform() < 0.5
    if self.lateral:
      self.hover_height = self.rng.uniform(0.02, 0.06)
      angle = self.rng.uniform(0, 2 * np.pi)
      radius = self.rng.uniform(0.0, 0.025) if self.rng.uniform() < 0.6 else 0.0
      self.lateral_offset = np.array([radius * np.cos(angle), radius * np.sin(angle), self.rng.uniform(-0.005, 0.02)])
    if self.miss:
      beside = self.rng.uniform() < 0.75
      above = not beside or self.rng.uniform() < 0.3
      self.miss_beside = self.rng.uniform(0.025, 0.07) * self.rng.choice([-1, 1]) if beside else 0.0
      self.miss_above = self.rng.uniform(0.025, 0.07) if above else 0.0

  def act(self):
    # Read privileged state
    s = self.env.privileged_state()
    ee = s["ee_pos"]
    obj = s["object_pos"]
    place = s["place_pos"]
    table = self.env.cfg.table.height
    gripper = 1.0
    target = ee
    # where the descent aims: off the object on a miss or lateral episode's first try
    aim = obj + self.miss_offset + self.lateral_offset if self.retries == 0 else obj

    # Phases of the policy, like a state machine
    if self.phase == "wander":
      # Anywhere on the table at low height, a few points in a row, the fingers shut on nothing half the time
      if self.wander_points is None:
        self.wander_points = [self._random_table_point() for _ in range(int(self.rng.integers(2, 5)))]
        self.wander_grip = -1.0 if self.rng.uniform() < 0.5 else 1.0
        self.route = self._plan_route(ee[:2], self.wander_points[0][:2])
      # each leg goes around the tall obstacles like the approach does, the last bit down to the point
      point = self.wander_points[0]
      wp = self.route[0] if self.route else point[:2]
      height = self._height_over(ee[:2], wp) if self.route else max(point[2], self._height_over(ee[:2], wp) - 0.08)
      target = np.array([wp[0], wp[1], height])
      gripper = self.wander_grip
      if self._near(ee, target, self.XY_TOL):
        if self.route:
          self.route.pop(0)
        else:
          self.wander_points.pop(0)
          self.wander_grip = -1.0 if self.rng.uniform() < 0.5 else 1.0
          if self.wander_points:
            self.route = self._plan_route(ee[:2], self.wander_points[0][:2])
          else:
            self.route = None
            self.roamed = True
            self.phase = "approach"

    elif self.phase == "approach":
      # Get above the object going around the tall obstacles, and over the low ones
      if self.route is None:
        self.route = self._plan_route(ee[:2], obj[:2])
      wp = self.route[0] if self.route else obj[:2]
      target = np.array([wp[0], wp[1], self._height_over(ee[:2], wp)])
      if self.lateral and not self.route:
        # the last leg at the low hover height, so the hand arrives from the side like the planner
        target[2] = obj[2] + self.hover_height
      if self._near(ee, target, self.XY_TOL):
        if self.route:
          self.route.pop(0)
        else:
          self.phase = "hover"

    elif self.phase == "hover":
      # Hover to a pose above the object
      target = np.array([obj[0], obj[1], obj[2] + self.hover_height])
      # Go down once we are above it and the fingers are turned the right way; on a wander-first episode
      # leave again from here and roam before coming back
      if self._near(ee, target, self.XY_TOL) and abs(self._yaw_action()) < 0.5:
        if not self.roamed:
          self.phase = "wander"
          self.wander_points = None
        else:
          self.phase = "descend"
        # the miss is set now that the fingers are aligned: across the finger axis, and / or too high
        yaw = self.env.finger_axis_yaw()
        self.miss_offset = np.array([-np.sin(yaw) * self.miss_beside, np.cos(yaw) * self.miss_beside, self.miss_above])

    elif self.phase == "descend":
      # Descend onto the object, but not into the table and not with the palm onto a tall object
      z = max(obj[2] - self.GRASP_DEPTH, table + self.MIN_EE_HEIGHT,
              self.env.object_top_z() - self.PALM_CLEARANCE) + aim[2] - obj[2]
      target = np.array([aim[0], aim[1], z])
      if np.linalg.norm(ee[:2] - target[:2]) < self.GRASP_TOL and abs(ee[2] - target[2]) < 0.01:
        self.phase = "grasp"
        self.hold = 0

    elif self.phase == "grasp":
      # Close the gripper and hold for a few steps
      gripper = -1.0
      self.hold += 1
      if self.hold >= self.HOLD_STEPS:
        self.phase = "lift"

    elif self.phase == "lift":
      # Lift straight up to the carry height
      target = np.array([ee[0], ee[1], self.carry_z])
      gripper = -1.0
      if abs(ee[2] - self.carry_z) < 0.02:
        if s["grasped"] or obj[2] > s["object_rest_z"] + 0.02:
          # Got it, plan a route to the place spot, through one random detour point on a wander episode
          if self.carry_detour:
            detour = self._random_table_point()[:2]
            self.route = self._plan_route(ee[:2], detour) + [detour] + self._plan_route(detour, place[:2])
            self.carry_detour = False
          else:
            self.route = self._plan_route(ee[:2], place[:2])
          self.phase = "transport"
        elif self.retries < self.MAX_RETRIES:
          # Missed, try the pick again
          self.retries += 1
          self.phase = "hover"
        else:
          self.phase = "done"

    elif self.phase == "transport":
      # Follow the waypoints, then head for the place spot, rising over any low obstacle on the way
      wp = self.route[0] if self.route else place[:2]
      target = np.array([wp[0], wp[1], self._height_over(ee[:2], wp)])
      gripper = -1.0
      halfway = 0.5 * np.linalg.norm(place[:2] - self.env.layout.pick_pos)
      if self.drop_early and np.linalg.norm(ee[:2] - place[:2]) < halfway:
        # Bad data on purpose, let go midway
        self.phase = "release"
        self.hold = 0
      elif obj[2] < s["object_rest_z"] + 0.01 and not s["grasped"]:
        # Lost it on the way, it is back on the table, go pick it up again if we still can
        self.retries += 1
        self.route = None
        self.phase = "approach" if self.retries <= self.MAX_RETRIES else "done"
      elif self._near(ee, target, self.XY_TOL):
        if self.route:
          self.route.pop(0)
        else:
          self.phase = "lower"

    elif self.phase == "lower":
      # Lower to just above the table
      target = np.array([place[0], place[1], table + self.half_h + 0.03])
      gripper = -1.0
      if abs(ee[2] - target[2]) < 0.015:
        self.phase = "release"
        self.hold = 0

    elif self.phase == "release":
      # Open the gripper and hold
      self.hold += 1
      if self.hold >= self.HOLD_STEPS:
        self.phase = "retreat"
        self.hold = 0

    elif self.phase == "retreat":
      # Move back up out of the way, wait a bit so the object settles, then we are done
      target = np.array([ee[0], ee[1], self.carry_z + 0.05])
      if abs(ee[2] - target[2]) < 0.03:
        self.hold += 1
        if self.hold >= self.HOLD_STEPS:
          self.phase = "done"

    # If the arm is not getting anywhere (blocked by an obstacle, target out of reach) stop wasting steps
    self.ee_hist = (self.ee_hist + [ee.copy()])[-self.STUCK_STEPS:]
    if self.phase in ("wander", "approach", "hover", "descend", "lift", "transport", "lower"):
      if len(self.ee_hist) == self.STUCK_STEPS and np.linalg.norm(ee - self.ee_hist[0]) < self.STUCK_DIST:
        self.phase = "done"
    else:
      self.ee_hist = []

    # Delta for the eef to move towards the target, scaled by speed
    delta = np.clip((target - ee) / self.env.cfg.control.max_delta * self.speed, -1.0, 1.0)
    if self.noise > 0:
      # Slowly drifting noise rather than per step jitter, damped in the delicate phases
      rho = self.env.cfg.data.scripted.noise_correlation
      self.noise_state = (rho * self.noise_state
                          + np.sqrt(1 - rho ** 2) * self.rng.normal(0, self.noise, 3))
      scale = 0.3 if self.phase in ("descend", "grasp", "lower", "release") else 1.0
      delta = np.clip(delta + scale * self.noise_state, -1.0, 1.0)
    action = list(delta)
    if self.env.controller.yaw_enabled:
      action.append(self._yaw_action())
    action.append(gripper)
    return np.array(action)

  # What made this episode different, stored in meta.json
  def describe(self):
    return {"mode": "low" if self.carry_low else "route",
            "speed": round(float(self.speed), 3),
            "carry_z": round(float(self.carry_z), 3),
            "drop_early": self.drop_early,
            "miss_beside": round(float(self.miss_beside), 3), "miss_above": round(float(self.miss_above), 3),
            "wander": self.wander, "lateral": self.lateral, "lateral_turn": self.lateral_turn,
            "lateral_offset": [round(float(v), 3) for v in self.lateral_offset]}

  # Turn the fingers square to the object while hovering / descending
  # every object fits in the 8 cm opening either way, so the nearest face (at most a 45 degree turn) is enough
  def _yaw_action(self):
    max_d = self.env.cfg.control.yaw.max_delta
    if self.phase not in ("hover", "descend") or (self.lateral and self.retries == 0 and not self.lateral_turn):
      return 0.0
    err = wrap_angle(self.env.object_long_axis_yaw() - self.env.finger_axis_yaw())
    err = (err + np.pi / 4) % (np.pi / 2) - np.pi / 4
    # half gain, full gain overshoots and the fingers wobble back and forth
    return float(np.clip(0.5 * err / max_d, -1.0, 1.0))

  # Waypoints around the obstacles that are too tall to carry over, waypoints right next to the place spot are dropped
  def _plan_route(self, start, goal):
    if self.carry_low:
      return []
    cfg = self.env.cfg
    lo = np.array(cfg.table.center) - np.array(cfg.table.half_size) + 0.06
    hi = np.array(cfg.table.center) + np.array(cfg.table.half_size) - 0.06
    blocking = [o for o in self.env.layout.obstacles if o.height > self.max_over]
    circles = [(o.pos, o.footprint_radius + self.margin) for o in blocking]
    reach = (np.array(cfg.robot.base_pos[:2]), cfg.task.reach_range[1])
    return [w for w in plan_route(start, goal, circles, lo, hi, reach)
            if np.linalg.norm(w - goal) > self.PLACE_KEEP_OUT]

  # Height to travel from a to b (xy), the carry height unless a low obstacle is under the hand on that leg
  def _height_over(self, a, b):
    need = self.carry_z
    for o in self.env.layout.obstacles:
      if segment_depth(a, b, [(o.pos, o.footprint_radius + self.HAND_RADIUS)]) > 0:
        need = max(need, self.env.cfg.table.height + o.height + self.hang + self.OVER_CLEARANCE)
    return min(need, self.ceiling)

  # A random reachable point on the table, 2-15 cm above the object's resting height, clear of the obstacles
  # and of the object (the hand would push against them and the episode would stall)
  def _random_table_point(self):
    cfg = self.env.cfg
    lo = np.array(cfg.table.center) - np.array(cfg.table.half_size)
    hi = np.array(cfg.table.center) + np.array(cfg.table.half_size)
    obj = self.env.privileged_state()["object_pos"][:2]
    for _ in range(100):
      point = np.array([self.rng.uniform(lo[0], hi[0]), self.rng.uniform(lo[1], hi[1]), self.env.object_rest_z + self.rng.uniform(0.02, 0.15)])
      clear = all(np.linalg.norm(point[:2] - o.pos) > o.footprint_radius + self.margin for o in self.env.layout.obstacles)
      if self._on_table(point) and clear and np.linalg.norm(point[:2] - obj) > 0.08:
        return point
    return np.array([*cfg.table.center, self.env.object_rest_z + 0.1])

  # Inside the table top with a margin, so a wander point is reachable
  def _on_table(self, point):
    cfg = self.env.cfg
    lo = np.array(cfg.table.center) - np.array(cfg.table.half_size) + 0.06
    hi = np.array(cfg.table.center) + np.array(cfg.table.half_size) - 0.06
    reach = np.linalg.norm(point[:2] - np.array(cfg.robot.base_pos[:2]))
    return bool(np.all(point[:2] > lo) and np.all(point[:2] < hi) and cfg.task.reach_range[0] <= reach <= cfg.task.reach_range[1])

  @staticmethod
  def _near(ee, target, tol):
    return np.linalg.norm(ee[:2] - target[:2]) < tol and abs(ee[2] - target[2]) < 0.025


# Random actions with some momentum and a downward drift so the arm actually reaches the table
class RandomPolicy:

  def __init__(self, env, rng):
    self.env = env
    self.rng = rng
    self.reset()

  def reset(self):
    self.momentum = np.zeros(self.env.action_dim - 1)
    self.gripper = 1.0

  # Runs till max_steps
  done = False

  def act(self):
    self.momentum = 0.8 * self.momentum + 0.6 * self.rng.normal(0, 1, len(self.momentum))
    # Flip the gripper now and then
    if self.rng.uniform() < 0.03:
      self.gripper = -self.gripper
    drift = np.zeros(len(self.momentum))
    drift[2] = -0.25
    return np.clip([*(self.momentum + drift), self.gripper], -1.0, 1.0)

  def describe(self):
    return {"mode": "random"}


# The policy kinds used in data.mix
def make_policy(kind, env, rng, cfg):
  noise = cfg.data.scripted.noise
  if kind == "success":
    # clean demonstration, no noise
    return ScriptedPickPlace(env, rng)
  if kind == "scripted":
    return ScriptedPickPlace(env, rng, noise=noise)
  if kind == "collide":
    return ScriptedPickPlace(env, rng, noise=noise, carry_low=True)
  if kind == "drop":
    return ScriptedPickPlace(env, rng, noise=noise, drop_early=True)
  if kind == "miss":
    return ScriptedPickPlace(env, rng, noise=noise, miss=True)
  if kind == "wander":
    return ScriptedPickPlace(env, rng, noise=noise, wander=True)
  if kind == "lateral":
    return ScriptedPickPlace(env, rng, noise=noise, lateral=True)
  if kind == "random":
    return RandomPolicy(env, rng)
  raise ValueError(f"unknown policy kind: {kind}")
