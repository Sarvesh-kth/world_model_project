"""Agents: what picks an action in the real env, one step at a time.

An agent turns the current observation into a planner state z, asks a planner (or a fixed
policy) for an action, and returns it. It's the glue between an env (M1), a world model
(oracle, state MLP, M2's JEPA model) and the CEM planner. eval.run_episode() drives any agent.

    agent.reset(adapter, obs)          start of an episode
    agent.act(adapter, obs) -> action  numpy [A], in [-1, 1]
"""

import numpy as np
import torch

from controller.config import CEMConfig
from controller.planner import CEMPlanner

FIXED_YAW = 0.0


def action_bounds(action_dim, plan_yaw=True):
    """[-1, 1] per action dim. plan_yaw=False pins yaw (5-dim M1 actions) to FIXED_YAW. Level E
    plans yaw: with it pinned, some cube yaws gave a corner-to-corner grasp that couldn't lift."""
    low, high = -np.ones(action_dim), np.ones(action_dim)
    if action_dim == 5 and not plan_yaw:
        low[3] = high[3] = FIXED_YAW
    return low, high


class ShowsPlans:
    """For visualizing a planning agent: the gripper paths its world model imagines.

    The agent sets self._plan_start (the planner state z of its latest replan) and implements
    ee_position(z) -> [..., 3] (where that model's state says the gripper is).
    """

    _plan_start = None

    @torch.no_grad()
    def imagined_paths(self, n_elites=8):
        """(chosen plan [H+1, 3], best elites [k, H+1, 3]) gripper paths, world frame; None before
        the first plan. Re-rolls the plan through the agent's own dynamics_fn (for the oracle these
        are extra sim steps in its private copy, never in the real env)."""
        if self._plan_start is None or self.planner.last_plan is None:
            return None
        seqs = torch.cat([self.planner.last_plan[None], self.planner.last_info["elite_actions"][:n_elites]])
        z = self._plan_start.expand(len(seqs), *self._plan_start.shape)
        path = [self.ee_position(z)]
        for t in range(seqs.shape[1]):
            z = self.dynamics(z, seqs[:, t].to(z.dtype if z.is_floating_point() else torch.float32))
            path.append(self.ee_position(z))
        paths = torch.stack(path, dim=1).float().cpu().numpy()
        return paths[0], paths[1:]


class RandomAgent:
    """Uniform random actions: the lower bound."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, adapter, obs):
        pass

    def act(self, adapter, obs):
        return self.rng.uniform(-1, 1, adapter.action_dim)


class ScriptedAgent:
    """M1's scripted expert (reads privileged state): the reference for what's achievable."""

    def reset(self, adapter, obs):
        from controller.adapters.m1_adapter import make_policy

        self.policy = make_policy("success", adapter.env, np.random.default_rng(0), adapter.cfg)
        self.policy.reset()

    def act(self, adapter, obs):
        return self.policy.act()


class OracleCEMAgent(ShowsPlans):
    """CEM planning with the simulator itself as the model (task 4, the upper-bound baseline)."""

    def __init__(self, cost_fn, cfg: CEMConfig, seed=0, plan_yaw=True):
        self.cost_fn, self.cfg, self.seed, self.plan_yaw = cost_fn, cfg, seed, plan_yaw

    def reset(self, adapter, obs):
        from controller.dynamics.oracle import OracleDynamics

        self.dynamics = OracleDynamics(adapter)
        low, high = action_bounds(adapter.action_dim, self.plan_yaw)
        self.planner = CEMPlanner(self.dynamics, self.cost_fn, low, high, cfg=self.cfg, seed=self.seed)

    def act(self, adapter, obs):
        z = self.dynamics.observe(adapter)
        if not self.planner._queue:
            self._plan_start = z
        return self.planner.act(z, torch.zeros(1)).numpy().astype(float)

    def ee_position(self, z):
        return z[..., 0:3]


class StateMLPAgent(ShowsPlans):
    """CEM planning with the learned state MLP (task 5). With audit=True, every new plan is also
    replayed in a simulator clone, logging predicted vs real cost: the planner "exploiting" the
    model shows up as plans that look much better to the model than they are."""

    def __init__(self, model_path, cost_fn, cfg: CEMConfig, seed=0, audit=False, plan_yaw=True):
        self.model_path, self.cost_fn, self.cfg, self.seed, self.audit = model_path, cost_fn, cfg, seed, audit
        self.plan_yaw = plan_yaw
        self.audit_log = []

    def reset(self, adapter, obs):
        from controller.dynamics.state_mlp import StateMLPDynamics, load

        self.dynamics = StateMLPDynamics(load(self.model_path))
        low, high = action_bounds(adapter.action_dim, self.plan_yaw)
        self.planner = CEMPlanner(self.dynamics, self.cost_fn, low, high, cfg=self.cfg, seed=self.seed)
        if self.audit:
            from controller.dynamics.oracle import OracleDynamics

            self.oracle = OracleDynamics(adapter)
        self.t = 0

    def act(self, adapter, obs):
        from controller.dynamics.state_mlp import state_from_adapter

        s = torch.from_numpy(state_from_adapter(adapter, obs))
        replanning = not self.planner._queue
        if replanning:
            self._plan_start = s
        oracle_z = self.oracle.observe(adapter) if self.audit and replanning else None
        action = self.planner.act(s, torch.zeros(1))
        if oracle_z is not None:
            self._audit(s, oracle_z, self.planner.last_plan)
        self.t += 1
        return action.numpy().astype(float)

    def ee_position(self, s):
        return s[..., 0:3]

    def _audit(self, s, oracle_z, plan):
        """Roll the chosen plan through the model and through the real sim; log both outcomes."""
        pred, real = [s[None]], [oracle_z[None]]
        for a in plan:
            pred.append(self.dynamics(pred[-1], a[None]))
            real.append(self.oracle(real[-1], a[None]))
        pred, real = torch.stack(pred, dim=1), torch.stack(real, dim=1)
        pf, rf = pred[0, -1, :14].numpy(), real[0, -1, :14].numpy()
        self.audit_log.append({
            "step": self.t,
            "predicted_cost": round(self.cost_fn(pred, None).item(), 4),
            "real_cost": round(self.cost_fn(real, None).item(), 4),
            "pred_obj_err_cm": round(100 * float(np.linalg.norm(pf[3:6] - rf[3:6])), 2),
            "pred_ee_err_cm": round(100 * float(np.linalg.norm(pf[0:3] - rf[0:3])), 2),
            "pred_grasped": int(pf[9] > 0.5), "real_grasped": int(rf[9] > 0.5),
        })


class JEPACEMAgent(ShowsPlans):
    """The Level E deliverable: CEM on the JEPA world model.

    Sees only what the real system would: the static camera (a rolling 64-frame clip, encoded by
    frozen V-JEPA 2 whenever it replans) and the 20 proprio values. Plans with M2's dynamics
    model D in latent space toward the goal state [z_goal | p_goal]. The goal comes from
    M1's scripted expert solving the task once (adapter.goal_frames), standing in for "a goal
    image is provided"; the agent never reads privileged sim state.
    """

    def __init__(self, checkpoint, encoder, cfg: CEMConfig, z_weight=1.0, p_weight=1.0, seed=0, device="cpu"):
        from controller.adapters.m2_adapter import JEPADynamics

        self.dynamics = JEPADynamics(checkpoint, device=device)
        self.encoder, self.cfg, self.seed, self.device = encoder, cfg, seed, device
        self.z_weight, self.p_weight = z_weight, p_weight
        self.plan_log = []

    def reset(self, adapter, obs):
        from controller.adapters.m2_adapter import ClipBuffer, jpeg_roundtrip
        from controller.costs import latent_goal_cost

        d = self.dynamics
        goal_frames, goal_p, ok = adapter.goal_frames(n=d.clip_frames, camera=d.camera)
        z_goal = self.encoder.encode_clip(np.stack([jpeg_roundtrip(f) for f in goal_frames]))
        self.s_goal = d.state(z_goal, goal_p)
        self.buffer = ClipBuffer(d.clip_frames)
        self.buffer.reset(adapter.render(d.camera))
        self.first = True
        low, high = action_bounds(adapter.action_dim)
        cost = latent_goal_cost(d.z_dim, self.z_weight, self.p_weight)
        self.planner = CEMPlanner(d, cost, low, high, cfg=self.cfg, device=self.device, seed=self.seed)
        self.t = 0

    def act(self, adapter, obs):
        if not self.first:
            self.buffer.push(adapter.render(self.dynamics.camera))  # the frame after the last action
        self.first = False
        s = None
        if not self.planner._queue:  # only encode when a new plan is needed (~1 s on the Mac)
            z = self.encoder.encode_clip(self.buffer.clip())
            s = self.dynamics.state(z, obs["proprio"])
            self._plan_start = s
        action = self.planner.act(s, self.s_goal)
        if s is not None:
            info = self.planner.last_info
            dist = (s - self.s_goal).square()
            self.plan_log.append({
                "step": self.t,
                "z_goal_mse": round(dist[: self.dynamics.z_dim].mean().item(), 4),
                "p_goal_mse": round(dist[self.dynamics.z_dim:].mean().item(), 4),
                "best_cost": round(info["best_cost"][-1], 4), "mean_cost": round(info["mean_cost"][0], 4),
            })
        self.t += 1
        return action.cpu().numpy().astype(float)

    def ee_position(self, s):
        """D predicts proprio too; its values 14..16 are the gripper position."""
        return self.dynamics.proprio(s)[..., 14:17]

    def goal_position(self):
        """Where the goal state puts the gripper (from the expert's final proprio)."""
        return self.ee_position(self.s_goal).cpu().numpy()
