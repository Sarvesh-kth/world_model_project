import numpy as np
from .route import plan_route, segment_depth

def wrap_angle(a):
  return (a + np.pi) % (2 * np.pi) - np.pi

# Waypoint state machine for pick / carry / place demos
# approach -> hover -> descend -> grasp -> lift -> transport -> lower -> release -> retreat -> done
# the carry always goes around the obstacles using the route planner, unless carry_low is set,
# then it goes straight through them at low height (deliberate collision episodes)
# drop_early lets go of the object halfway through the carry
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

  def __init__(self, env, rng, noise=0.0, carry_low=False, drop_early=False):
    self.env = env
    self.rng = rng
    self.noise = noise
    self.carry_low = carry_low
    self.drop_early = drop_early
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
    self.phase = "approach"
    self.hold = 0
    self.retries = 0
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

  def act(self):
    # Read privileged state
    s = self.env.privileged_state()
    ee = s["ee_pos"]
    obj = s["object_pos"]
    place = s["place_pos"]
    table = self.env.cfg.table.height
    gripper = 1.0
    target = ee

    # Phases of the policy, like a state machine
    if self.phase == "approach":
      # Get above the object going around the tall obstacles, and over the low ones
      if self.route is None:
        self.route = self._plan_route(ee[:2], obj[:2])
      wp = self.route[0] if self.route else obj[:2]
      target = np.array([wp[0], wp[1], self._height_over(ee[:2], wp)])
      if self._near(ee, target, self.XY_TOL):
        if self.route:
          self.route.pop(0)
        else:
          self.phase = "hover"
    elif self.phase == "hover":
      # Hover to a pose above the object
      target = np.array([obj[0], obj[1], obj[2] + self.HOVER_HEIGHT])
      # Go down once we are above it and the fingers are turned the right way
      if self._near(ee, target, self.XY_TOL) and abs(self._yaw_action()) < 0.5:
        self.phase = "descend"
    elif self.phase == "descend":
      # Descend onto the object, but not into the table and not with the palm onto a tall object
      z = max(obj[2] - self.GRASP_DEPTH, table + self.MIN_EE_HEIGHT,
              self.env.object_top_z() - self.PALM_CLEARANCE)
      target = np.array([obj[0], obj[1], z])
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
          # Got it, plan a route to the place spot
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
    if self.phase in ("approach", "hover", "descend", "lift", "transport", "lower"):
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
            "drop_early": self.drop_early}

  # Turn the fingers square to the object while hovering / descending
  # every object fits in the 8 cm opening either way, so the nearest face (at most a 45 degree turn) is enough
  def _yaw_action(self):
    max_d = self.env.cfg.control.yaw.max_delta
    if self.phase not in ("hover", "descend"):
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
  if kind == "random":
    return RandomPolicy(env, rng)
  raise ValueError(f"unknown policy kind: {kind}")
