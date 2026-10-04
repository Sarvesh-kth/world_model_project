"""Shared true/estimated task state for the opt-in SAC and JEPA control experiment."""
import copy

import numpy as np

from environment import EpisodeLayout, PickPlaceEnv
from environment.config import Config
from environment.rewards import Rewards


class GoalReward(Rewards):
    """Opt-in potential shaping; repeated holding earns no continuing bonus."""
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
        _, original, _ = super().compute(state, action)
        if not state["grasped"]:
            original["reach"] = 1.0-float(np.tanh(5*np.linalg.norm(state["ee_pos"]-state["object_pos"])))
        distance = np.linalg.norm(state["object_pos"][:2]-state["place_pos"][:2])
        original["drop"] = -float(previous_held and not state["grasped"] and distance > 2*self.success_radius)
        lift = state["object_pos"][2]-state["object_rest_z"]
        self.ever_lifted |= bool(state["grasped"] and lift >= .04)
        settled = (self.was_grasped and self.ever_lifted and not state["grasped"]
                   and np.linalg.norm(state["object_pos"][:2]-state["place_pos"][:2]) < self.success_radius
                   and abs(lift) < .02)
        self.settled_steps = self.settled_steps+1 if settled else 0
        success = self.settled_steps >= self.settle_required
        goal_bonus = 15.0 if success and not self.task_succeeded else 0.0
        self.task_succeeded |= success
        self.potential = sum(self.cfg.weights[k]*original[k] for k in ("reach", "grasp", "lift", "transport"))
        terminal = self.task_succeeded or state["failed"]
        if terminal:
            self.potential = 0.0
        penalties = sum(self.cfg.weights[k]*original[k] for k in
                        ("collision", "proximity", "table_hit", "drop", "action"))
        if state["failed"] and not self.task_succeeded:
            penalties -= self.cfg.weights["fail"]
        components = {"progress": self.gamma*self.potential-old,
                      "goal": goal_bonus, "penalties": penalties, "time": -.01}
        return sum(components.values()), components, self.task_succeeded


def policy_observation(p, xyz, held, memory, goal, rest_z, remaining):
    """Same 41 numbers for SAC's true-state and Q-estimated-state observations."""
    p, xyz, goal = map(lambda x: np.asarray(x, np.float32), (p, xyz, goal))
    history = [float(memory.was_grasped), float(memory.prev_grasped), float(memory.ever_lifted),
               memory.settled_steps/memory.settle_required, float(memory.task_succeeded), memory.potential]
    # Fixed unit scaling gives centimetre-sized grasp offsets useful magnitude.
    # D/Q retain their original raw-unit inputs and checkpoint normalizers.
    robot = p.copy()
    robot[:7] /= np.pi; robot[7:14] /= 2
    origin = np.array([0, 0, rest_z], np.float32)
    robot[14:17] = 5*(p[14:17]-origin)
    robot[17] /= np.pi; robot[18] *= 25
    observation = np.concatenate((robot, 5*(xyz-origin), 20*(xyz-p[14:17]),
                                  5*(goal-origin), 5*(goal-xyz),
                                  [held], history, [rest_z, remaining])).astype(np.float32)
    if observation.shape != (41,) or not np.isfinite(observation).all():
        raise ValueError("invalid SAC state: expected 41 finite numbers")
    return observation


def predicted_reward(memory, p, xyz, held, action, goal, rest_z, failed=False):
    """Use the same goal/progress equations on estimated state, omitting contact terms."""
    state = {"object_pos": np.asarray(xyz), "ee_pos": np.asarray(p)[14:17],
             "place_pos": np.asarray(goal), "object_rest_z": rest_z,
             "grasped": bool(held >= .5), "failed": bool(failed),
             "obstacle_contacts": 0, "table_contacts": 0,
             "obstacle_distance": memory.safe_distance}
    # ponytail: Q has no collision head; this experiment requires an empty layout.
    # Contact terms are omitted in imagined rewards, never in actual evaluation.
    total, components, _ = memory.compute(state, action)
    return float(total), components


class TaskSession:
    """Physical simulator + evaluation only. JEPA action selection receives no sim state."""
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
        self.ever_held = self.lifted = False
        self.settled_steps = self.collisions = self.table_hits = 0
        self.maximum_lift = 0.0
        self.minimum_distance = float(np.linalg.norm(self.obs["state"][:2]-self.goal[:2]))
        self.success = False
        self.total = 0.0
        self.control_total = 0.0
        self.reward = GoalReward(self.cfg)
        self.last_info = {}
        return self.observation()

    def observation(self):
        held = self.sim._check_contacts()[0]
        return policy_observation(self.obs["proprio"], self.obs["state"][:3], float(held),
                                  self.reward, self.goal, self.rest_z,
                                  max(0, 1-self.sim.step_count/self.cfg.episode.max_steps))

    def step(self, action):
        self.obs, reward, terminated, truncated, info = self.sim.step(action)
        held = self.sim._check_contacts()[0]
        xyz = self.obs["state"][:3]
        lift = float(xyz[2]-self.rest_z)
        distance = float(np.linalg.norm(xyz[:2]-self.goal[:2]))
        self.ever_held |= bool(held or info["grasped"])
        self.lifted |= bool(held and lift >= .04)
        self.maximum_lift = max(self.maximum_lift, lift)
        self.minimum_distance = min(self.minimum_distance, distance)
        self.collisions += int(info["obstacle_contact"])
        self.table_hits += int(info["table_contact"])
        settled = (self.ever_held and self.lifted and not held
                   and distance < self.cfg.episode.success_radius
                   and abs(lift) < .02)
        self.settled_steps = self.settled_steps+1 if settled else 0
        self.success |= self.settled_steps >= self.cfg.episode.settle_steps
        self.total += float(reward)
        self.last_info = {**info, "held_endpoint": bool(held), "task_success": bool(self.success)}
        timeout = self.sim.step_count >= self.cfg.episode.max_steps
        # Let evaluation enforce lift+release+settling; the old placement event
        # alone can occur before this stricter criterion has been established.
        fell = xyz[2] < self.cfg.table.height-self.cfg.episode.fall_margin
        state = {"object_pos": xyz, "ee_pos": self.obs["proprio"][14:17], "place_pos": self.goal,
                 "object_rest_z": self.rest_z, "grasped": bool(held), "failed": bool(fell or timeout),
                 "obstacle_contacts": int(info["obstacle_contact"]), "table_contacts": int(info["table_contact"]),
                 "obstacle_distance": self.sim.obstacle_distance()}
        control_reward, components, control_success = self.reward.compute(state, action)
        assert bool(control_success) == self.success, "reward and evaluator success disagree"
        self.control_total += float(control_reward)
        self.last_info.update(original_reward=float(reward), control_reward_components=components,
                              task_failed=bool((fell or timeout) and not self.success))
        # The deadline is part of this finite-horizon task (remaining time is an
        # input), so SAC must not bootstrap a continuation beyond it.
        done = bool(self.success or fell or timeout)
        return self.observation(), float(control_reward), done, False, self.last_info

    def result(self):
        return {"task_success": bool(self.success), "steps": self.sim.step_count,
                "true_original_return": self.total, "control_return": self.control_total,
                "ever_held": bool(self.ever_held),
                "ever_lifted_4cm": bool(self.lifted), "maximum_lift_cm": 100*self.maximum_lift,
                "minimum_goal_distance_cm": 100*self.minimum_distance,
                "final_goal_distance_cm": float(100*np.linalg.norm(self.obs["state"][:2]-self.goal[:2])),
                "final_held": bool(self.sim._check_contacts()[0]),
                "obstacle_contact_steps": self.collisions, "table_contact_steps": self.table_hits,
                "layout": self.sim.layout.to_dict()}

    def close(self):
        self.sim.close()


def make_rl_env(cfg, layout, jitter, seed):
    # Keep Gym/SB3 optional: normal simulation and CPU checks need neither.
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
            self.next_seed = current+1
            return self.session.reset(current), {}

        def step(self, action):
            obs, reward, terminated, truncated, info = self.session.step(action)
            info["is_success"] = info["task_success"]
            return obs, reward, terminated, truncated, info

        def close(self):
            self.session.close()

    return SACEnv()
