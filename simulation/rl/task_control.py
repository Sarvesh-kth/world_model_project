import copy

import numpy as np

from environment import EpisodeLayout, PickPlaceEnv
from environment.config import Config
from environment.rewards import Rewards

# What the SAC policy sees and is rewarded with, and the simulator wrapper every controller runs in.
# The observation is the same 41 numbers whether they come from the exact simulator state (rl_true) or
# from the JEPA readout Q (rl_q, jepa_mpc), so one policy serves both. The reward is a potential shaping
# of the environment's staged reward, with a one time bonus for a finished placement.


# Reward for the A-to-B task, built on the environment's Rewards
# progress = gamma * potential(next) - potential(now), so holding still earns nothing and the sum over an
# episode telescopes to the potential gained. The 15 point goal bonus is paid once, when the cube has sat
# on the target for settle_steps steps after being lifted and released.
class GoalReward(Rewards):
  gamma = .99

  def __init__(self, cfg):
    self.settle_required = cfg.episode.settle_steps
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
# and the fraction of the episode left (2). Fixed scales turn centimetres into numbers near 1.
def policy_observation(p, xyz, held, memory, goal, rest_z, remaining):
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
  if observation.shape != (41,) or not np.isfinite(observation).all():
    raise ValueError("invalid policy observation, expected 41 finite numbers")
  return observation


# The same reward on an estimated state (Q's cube position and held probability), used by the planner
# on imagined states. Contacts are not known there, the penalty head R adds them separately.
def predicted_reward(memory, p, xyz, held, action, goal, rest_z, failed=False):
  state = {"object_pos": np.asarray(xyz), "ee_pos": np.asarray(p)[14:17], "place_pos": np.asarray(goal),
           "object_rest_z": rest_z, "grasped": bool(held >= .5), "failed": bool(failed),
           "obstacle_contacts": 0, "table_contacts": 0, "obstacle_distance": memory.safe_distance}
  total, components, _ = memory.compute(state, action)
  return float(total), components


# One fixed layout, run many times with the cube start jittered. Wraps PickPlaceEnv, computes the
# GoalReward and the policy observation from the exact state, and keeps the episode statistics.
# The controllers never read the exact state from here except through observation() (rl_true) and
# the evaluation labels in step()'s info.
class TaskSession:

  def __init__(self, cfg, layout, jitter=0.0):
    self.cfg = Config.nested(copy.deepcopy(cfg))
    self.layout = copy.deepcopy(layout)
    self.jitter = jitter
    self.sim = PickPlaceEnv(self.cfg)

  def reset(self, seed):
    rng = np.random.default_rng(seed)
    layout = copy.deepcopy(self.layout)
    layout["pick"] = (np.asarray(layout["pick"]) + rng.uniform(-self.jitter, self.jitter, 2)).tolist()
    self.obs, _ = self.sim.reset(seed=seed, layout=EpisodeLayout.from_dict(layout))
    self.rest_z = self.sim.object_rest_z
    self.goal = np.array([*layout["place"], self.rest_z], np.float32)
    self.reward = GoalReward(self.cfg)

    # episode statistics
    self.ever_held = self.lifted = self.success = False
    self.settled_steps = self.collisions = self.table_hits = 0
    self.maximum_lift = 0.0
    self.minimum_distance = float(np.linalg.norm(self.obs["state"][:2] - self.goal[:2]))
    self.total = self.control_total = 0.0
    return self.observation()

  # Exact state observation for rl_true and for training
  def observation(self):
    held = self.sim._check_contacts()[0]
    remaining = max(0, 1 - self.sim.step_count / self.cfg.episode.max_steps)
    return policy_observation(self.obs["proprio"], self.obs["state"][:3], float(held),
                              self.reward, self.goal, self.rest_z, remaining)

  def step(self, action):
    self.obs, reward, terminated, truncated, info = self.sim.step(action)
    held = self.sim._check_contacts()[0]
    xyz = self.obs["state"][:3]
    lift = float(xyz[2] - self.rest_z)
    distance = float(np.linalg.norm(xyz[:2] - self.goal[:2]))

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


# Gym wrapper around TaskSession for stable-baselines3, each reset takes the next seed
def make_rl_env(cfg, layout, jitter, seed):
  import gymnasium as gym

  class SACEnv(gym.Env):

    def __init__(self):
      self.session = TaskSession(cfg, layout, jitter)
      self.next_seed = seed
      self.action_space = gym.spaces.Box(-1, 1, (5,), dtype=np.float32)
      self.observation_space = gym.spaces.Box(-np.inf, np.inf, (41,), dtype=np.float32)

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
