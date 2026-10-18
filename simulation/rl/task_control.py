import copy

import numpy as np

from environment import EpisodeLayout, PickPlaceEnv
from environment.config import Config
from environment.obstacles import sample_obstacles
from environment.rewards import Rewards
from environment.scene import sample_layout

# What the SAC policy sees and is rewarded with, and the simulator wrapper every controller runs in.
# The observation is the same 41 numbers whether they come from the exact simulator state (rl_true) or
# from the JEPA readout Q (rl_q), so one policy serves both; a policy trained with obstacles also gets a map
# of the table: the 8x8 height map (64 more numbers, the old obstacle-aware SAC) or the 16x16 obstacle-only
# depth map (256 more, what the MPC's obstacle model reads, controller --obstacle-source model). The reward
# is a potential shaping of the environment's staged reward, with a one time bonus for a finished placement.

# the obstacle map is refreshed every this many steps during training (the segmentation render costs 3 ms);
# the controllers refresh it every step, the running maximum makes both the same map a few steps apart
MAP_EVERY = 5
# cells of the obstacle map around the hand's cell the policy sees in "window" mode (the obstacle model's own
# window, world_model/train_obstacle.py): a net over the whole map had to memorise scenes, the same window
# seen from the hand wherever it is makes the geometry learnable
WINDOW = 9


# The WINDOW x WINDOW cell heights of the obstacle map around the hand's cell (0 outside the table) and the
# hand's offset inside its cell (0-1 along x and y): 83 numbers, in metres like the map itself. With
# coarse=True the whole map max-pooled to 8x8 follows (64 more), so the policy also sees the layout beyond
# the window when choosing which way round to carry
def hand_window(obstacle_map, p, cfg, coarse=False):
  cells = obstacle_map.shape[0]
  centre, half = np.asarray(cfg.table.center), np.asarray(cfg.table.half_size)
  rel = (np.asarray(p)[14:16] - (centre - half)) / (2 * half / cells)
  cell = np.clip(np.floor(rel).astype(int), 0, cells - 1)
  r = WINDOW // 2
  padded = np.zeros((cells + WINDOW - 1, cells + WINDOW - 1), np.float32)
  padded[r:r + cells, r:r + cells] = obstacle_map
  window = padded[cell[0]:cell[0] + WINDOW, cell[1]:cell[1] + WINDOW]
  parts = [window.reshape(-1), np.clip(rel - cell, 0, 1) / 4]
  if coarse:
    parts.append(obstacle_map.reshape(8, cells // 8, 8, cells // 8).max(axis=(1, 3)).reshape(-1))
  return np.concatenate(parts).astype(np.float32)


# Reward for the A-to-B task, built on the environment's Rewards
# progress = gamma * potential(next) - potential(now), so holding still earns nothing and the sum over an
# episode telescopes to the potential gained. The 15 point goal bonus is paid once, when the cube has sat
# on the target for settle_steps steps after being lifted and released.
# pull adds the planner's linear distance pull to the potential (the way still to go: hand to cube and cube
# to B before the grasp, cube to B while carried, in metres times pull): the shaped terms saturate beyond
# ~25 cm, this keeps a gradient from anywhere. A potential term, so it changes nothing about what is
# optimal; the controllers leave it 0.
class GoalReward(Rewards):
  gamma = .99

  def __init__(self, cfg, pull=0.0):
    self.settle_required = cfg.episode.settle_steps
    self.pull = pull
    super().__init__(cfg)

  def reset(self):
    super().reset()
    self.ever_lifted = self.task_succeeded = False
    self.settled_steps = 0
    self.potential = 0.0

  def compute(self, state, action):
    old = self.potential
    previous_held = self.prev_grasped
    _, c, _ = super().compute(state, action)

    # reach counts until the cube is grasped, drop fires once when it is let go away from the target
    if not state["grasped"]:
      c["reach"] = 1.0 - float(np.tanh(5 * np.linalg.norm(state["ee_pos"] - state["object_pos"])))
    distance = np.linalg.norm(state["object_pos"][:2] - state["place_pos"][:2])
    c["drop"] = -float(previous_held and not state["grasped"] and distance > 2 * self.success_radius)

    # placed = lifted at least 4 cm at some point, now released, resting within success_radius of B
    lift = state["object_pos"][2] - state["object_rest_z"]
    self.ever_lifted |= bool(state["grasped"] and lift >= .04)
    settled = (self.was_grasped and self.ever_lifted and not state["grasped"]
               and distance < self.success_radius and abs(lift) < .02)
    self.settled_steps = self.settled_steps + 1 if settled else 0
    success = self.settled_steps >= self.settle_required
    goal_bonus = 15.0 if success and not self.task_succeeded else 0.0
    self.task_succeeded |= success

    # the potential is the weighted sum of the shaped terms, zero once the episode is over
    self.potential = sum(self.cfg.weights[k] * c[k] for k in ("reach", "grasp", "lift", "transport"))
    if self.pull:
      way = distance if state["grasped"] else distance + np.linalg.norm(state["ee_pos"] - state["object_pos"])
      self.potential -= self.pull * float(way)
    if self.task_succeeded or state["failed"]:
      self.potential = 0.0
    penalties = sum(self.cfg.weights[k] * c[k] for k in ("collision", "proximity", "table_hit", "drop", "action"))
    if state["failed"] and not self.task_succeeded:
      penalties -= self.cfg.weights["fail"]

    components = {"progress": self.gamma * self.potential - old, "goal": goal_bonus,
                  "penalties": penalties, "time": -.01}
    return sum(components.values()), components, self.task_succeeded


# The 41 numbers the policy sees: scaled robot state (20), cube position relative to the table (3), cube
# relative to the hand (3), goal relative to the table (3) and to the cube (3), held (1), the reward's
# memory (6: was grasped, held last step, ever lifted, settle fraction, succeeded, potential), rest height
# and the fraction of the episode left (2). Fixed scales turn centimetres into numbers near 1. With a
# map (metres, cells x cells) its cells follow, scaled by 4 so a 19 cm wall reads 0.76: the 8x8 height map
# (64 numbers), the 16x16 obstacle-only map (256) or the hand_window of it (83).
def policy_observation(p, xyz, held, memory, goal, rest_z, remaining, heightmap=None, obstacle_map=None):
  p, xyz, goal = (np.asarray(x, np.float32) for x in (p, xyz, goal))
  history = [float(memory.was_grasped), float(memory.prev_grasped), float(memory.ever_lifted),
             memory.settled_steps / memory.settle_required, float(memory.task_succeeded), memory.potential]
  robot = p.copy()
  robot[:7] /= np.pi
  robot[7:14] /= 2
  origin = np.array([0, 0, rest_z], np.float32)
  robot[14:17] = 5 * (p[14:17] - origin)
  robot[17] /= np.pi
  robot[18] *= 25
  observation = np.concatenate((robot, 5 * (xyz - origin), 20 * (xyz - p[14:17]), 5 * (goal - origin),
                                5 * (goal - xyz), [held], history, [rest_z, remaining])).astype(np.float32)
  table = heightmap if heightmap is not None else obstacle_map
  if table is not None:
    observation = np.concatenate((observation, 4 * np.asarray(table, np.float32).reshape(-1)))
  if observation.shape not in ((41,), (105,), (297,), (124,), (188,)) or not np.isfinite(observation).all():
    raise ValueError("invalid policy observation, expected 41, 105, 124, 188 or 297 finite numbers")
  return observation


# The same reward on an estimated state (Q's cube position and held probability), used by the planner
# on imagined states. Contacts are not known there, the penalty head R adds them separately.
def predicted_reward(memory, p, xyz, held, action, goal, rest_z, failed=False):
  state = {"object_pos": np.asarray(xyz), "ee_pos": np.asarray(p)[14:17], "place_pos": np.asarray(goal),
           "object_rest_z": rest_z, "grasped": bool(held >= .5), "failed": bool(failed),
           "gripper_closed": bool(np.asarray(p)[19] < 0), "gripper_width": float(np.asarray(p)[18]),
           "obstacle_contacts": 0, "table_contacts": 0, "obstacle_distance": memory.safe_distance}
  total, components, _ = memory.compute(state, action)
  return float(total), components


# One layout run many times, or with random_ends a whole new scene every episode from the environment's
# own sampler (cube and goal anywhere on the reachable table, the obstacles in between: the distribution
# the world model's data and the controller's --random-ends benchmark use). Wraps PickPlaceEnv, computes
# the GoalReward and the policy observation from the exact state, and keeps the episode statistics.
# obstacles=(low, high) draws that many fresh obstacles every episode; heightmap / obstacle_map ("full" or
# "window") choose what map the observation carries. cube_noise (metres) jitters the cube position the policy is shown, so a
# policy trained on the exact cube is not thrown by Q's centimetre of error at run time. The controllers
# never read the exact state from here except through observation() (rl_true) and the evaluation labels
# in step()'s info.
class TaskSession:

  def __init__(self, cfg, layout, jitter=0.0, obstacles=None, heightmap=False, obstacle_map=False,
               random_ends=False, cube_noise=0.0, pull=0.0):
    self.cfg = Config.nested(copy.deepcopy(cfg))
    self.layout = copy.deepcopy(layout)
    self.jitter = jitter
    self.obstacles = obstacles
    self.obstacle_map = bool(obstacle_map)
    self.window = obstacle_map in ("window", "both")
    self.coarse = obstacle_map == "both"
    self.heightmap = heightmap or (obstacles is not None and not obstacle_map)
    self.random_ends = random_ends
    self.cube_noise = cube_noise
    self.pull = pull
    self.sim = PickPlaceEnv(self.cfg)

  # obstacles overrides the session's (low, high) for this one episode (the validation's empty / obstacle split)
  def reset(self, seed, obstacles=None):
    rng = np.random.default_rng(seed)
    count = obstacles if obstacles is not None else self.obstacles
    if self.random_ends:
      cfg = copy.deepcopy(self.cfg)
      cfg.task.pick_end = "anywhere"
      cfg.task.obstacles.count = list(count) if count is not None else [0, 0]
      layout = sample_layout(cfg, rng).to_dict()
    else:
      layout = copy.deepcopy(self.layout)
      layout["pick"] = (np.asarray(layout["pick"]) + rng.uniform(-self.jitter, self.jitter, 2)).tolist()
      if count is not None:
        cfg = copy.deepcopy(self.cfg)
        cfg.task.obstacles.count = list(count)
        drawn = sample_obstacles(cfg, rng, np.asarray(layout["pick"]), np.asarray(layout["place"]))
        layout["obstacles"] = [o.to_dict() for o in drawn] if drawn else []
    self.obs, _ = self.sim.reset(seed=seed, layout=EpisodeLayout.from_dict(layout))
    self.rest_z = self.sim.object_rest_z
    self.goal = np.array([*layout["place"], self.rest_z], np.float32)
    self.reward = GoalReward(self.cfg, self.pull)
    self.noise = np.random.default_rng(seed + 7)
    # the obstacles as the depth camera sees them, a running maximum over the episode (they do not move,
    # cells the arm hides later keep what was seen)
    self.obstacle_map_now = self.sim.obstacle_map() if self.obstacle_map else None

    # episode statistics
    self.ever_held = self.lifted = self.success = False
    self.settled_steps = self.collisions = self.table_hits = 0
    self.maximum_lift = 0.0
    self.minimum_distance = float(np.linalg.norm(self.obs["state"][:2] - self.goal[:2]))
    self.total = self.control_total = 0.0
    return self.observation()

  # Exact state observation for rl_true and for training, with the map when the policy uses one
  def observation(self):
    held = self.sim._check_contacts()[0]
    remaining = max(0, 1 - self.sim.step_count / self.cfg.episode.max_steps)
    xyz = self.obs["state"][:3]
    if self.cube_noise:
      xyz = xyz + self.noise.normal(0, self.cube_noise, 3).astype(np.float32)
    table = hand_window(self.obstacle_map_now, self.obs["proprio"], self.cfg, self.coarse) if self.window else self.obstacle_map_now
    return policy_observation(self.obs["proprio"], xyz, float(held), self.reward, self.goal, self.rest_z, remaining,
                              self.sim.height_map() if self.heightmap else None, table)

  def step(self, action):
    self.obs, reward, terminated, truncated, info = self.sim.step(action)
    held = self.sim._check_contacts()[0]
    xyz = self.obs["state"][:3]
    lift = float(xyz[2] - self.rest_z)
    distance = float(np.linalg.norm(xyz[:2] - self.goal[:2]))
    if self.obstacle_map and self.sim.step_count % MAP_EVERY == 0:
      self.obstacle_map_now = np.maximum(self.obstacle_map_now, self.sim.obstacle_map())

    # bookkeeping for the result
    self.ever_held |= bool(held or info["grasped"])
    self.lifted |= bool(held and lift >= .04)
    self.maximum_lift = max(self.maximum_lift, lift)
    self.minimum_distance = min(self.minimum_distance, distance)
    self.collisions += int(info["obstacle_contact"])
    self.table_hits += int(info["table_contact"])
    self.total += float(reward)

    # success = lifted, released, resting on the target for settle_steps steps
    settled = (self.ever_held and self.lifted and not held
               and distance < self.cfg.episode.success_radius and abs(lift) < .02)
    self.settled_steps = self.settled_steps + 1 if settled else 0
    self.success |= self.settled_steps >= self.cfg.episode.settle_steps

    # the control reward from the exact state, same formula the planner uses on estimates
    timeout = self.sim.step_count >= self.cfg.episode.max_steps
    fell = xyz[2] < self.cfg.table.height - self.cfg.episode.fall_margin
    state = {"object_pos": xyz, "ee_pos": self.obs["proprio"][14:17], "place_pos": self.goal,
             "object_rest_z": self.rest_z, "grasped": bool(held), "failed": bool(fell or timeout),
             "gripper_closed": bool(self.obs["proprio"][19] < 0), "gripper_width": float(self.obs["proprio"][18]),
             "obstacle_contacts": int(info["obstacle_contact"]), "table_contacts": int(info["table_contact"]),
             "obstacle_distance": self.sim.obstacle_distance()}
    control_reward, components, _ = self.reward.compute(state, action)
    self.control_total += float(control_reward)

    info.update(held_endpoint=bool(held), task_success=bool(self.success), original_reward=float(reward),
                control_reward_components=components, task_failed=bool((fell or timeout) and not self.success))
    # the deadline ends the episode, the policy knows how much time is left so it must not plan past it
    done = bool(self.success or fell or timeout)
    return self.observation(), float(control_reward), done, False, info

  def result(self):
    return {"task_success": bool(self.success), "steps": self.sim.step_count,
            "true_original_return": self.total, "control_return": self.control_total,
            "ever_held": bool(self.ever_held), "ever_lifted_4cm": bool(self.lifted),
            "maximum_lift_cm": 100 * self.maximum_lift,
            "minimum_goal_distance_cm": 100 * self.minimum_distance,
            "final_goal_distance_cm": float(100 * np.linalg.norm(self.obs["state"][:2] - self.goal[:2])),
            "final_held": bool(self.sim._check_contacts()[0]),
            "obstacle_contact_steps": self.collisions, "table_contact_steps": self.table_hits,
            "layout": self.sim.layout.to_dict()}

  def close(self):
    self.sim.close()


# The session a training run uses, the same options for demonstrations, training and validation
def session_for(cfg, layout, args, cube_noise=0.0):
  return TaskSession(cfg, layout, args.position_jitter, args.obstacles, obstacle_map=getattr(args, "obstacle_map", False),
                     random_ends=getattr(args, "random_ends", False), cube_noise=cube_noise, pull=getattr(args, "pull", 0.0))


# Gym wrapper around TaskSession for stable-baselines3, each reset takes the next seed
def make_rl_env(cfg, layout, seed, args):
  import gymnasium as gym

  class SACEnv(gym.Env):

    def __init__(self):
      self.session = session_for(cfg, layout, args, getattr(args, "cube_noise", 0.0))
      self.next_seed = seed
      self.action_space = gym.spaces.Box(-1, 1, (5,), dtype=np.float32)
      width = (188 if self.session.coarse else 124 if self.session.window else 297 if self.session.obstacle_map
               else 105 if self.session.heightmap else 41)
      self.observation_space = gym.spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
      super().reset(seed=seed)
      current = self.next_seed if seed is None else seed
      self.next_seed = current + 1
      return self.session.reset(current), {}

    def step(self, action):
      obs, reward, terminated, truncated, info = self.session.step(action)
      info["is_success"] = info["task_success"]
      return obs, reward, terminated, truncated, info

    def close(self):
      self.session.close()

  return SACEnv()
