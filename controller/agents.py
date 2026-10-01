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


class OracleCEMAgent:
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
        return self.planner.act(z, torch.zeros(1)).numpy().astype(float)


class StateMLPAgent:
    """CEM planning with the learned state MLP (task 5). With audit=True, every new plan is also
    replayed in a simulator clone, logging predicted vs real cost: the planner "exploiting" the
    model shows up as plans that look much better to the model than they are."""

    def __init__(self, model_path, cost_fn, cfg: CEMConfig, seed=0, audit=False):
        self.model_path, self.cost_fn, self.cfg, self.seed, self.audit = model_path, cost_fn, cfg, seed, audit
        self.audit_log = []

    def reset(self, adapter, obs):
        from controller.dynamics.state_mlp import StateMLPDynamics, load

        self.dynamics = StateMLPDynamics(load(self.model_path))
        low, high = action_bounds(adapter.action_dim)
        self.planner = CEMPlanner(self.dynamics, self.cost_fn, low, high, cfg=self.cfg, seed=self.seed)
        if self.audit:
            from controller.dynamics.oracle import OracleDynamics

            self.oracle = OracleDynamics(adapter)
        self.t = 0

    def act(self, adapter, obs):
        from controller.dynamics.state_mlp import state_from_adapter

        s = torch.from_numpy(state_from_adapter(adapter, obs))
        replanning = not self.planner._queue
        oracle_z = self.oracle.observe(adapter) if self.audit and replanning else None
        action = self.planner.act(s, torch.zeros(1))
        if oracle_z is not None:
            self._audit(s, oracle_z, self.planner.last_plan)
        self.t += 1
        return action.numpy().astype(float)

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


class JEPACEMAgent:
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
