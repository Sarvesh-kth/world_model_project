"""The planner: the loop around cem_plan.

Each act() call returns one action. When no planned actions are left, it runs cem_plan from the
current latent, queues the first `execute_steps` actions of the resulting plan, and keeps the
rest of the plan to warm-start the next search.

Level E executes a few steps per plan open-loop (execute_steps > 1). Grade C MPC is
execute_steps = 1: replan after every action.
"""

import torch

from controller.cem import cem_plan
from controller.config import CEMConfig
from controller.interfaces import CostFn, DynamicsFn


class CEMPlanner:
    def __init__(
        self,
        dynamics_fn: DynamicsFn,
        cost_fn: CostFn,
        action_low,
        action_high,
        cfg: CEMConfig | None = None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ):
        """action_low / action_high: bounds per action dimension [A]. Latents passed to act()
        must be on `device`. seed makes planning reproducible."""
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.cfg = cfg or CEMConfig()
        self.device = torch.device(device)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)
        self.last_info = None  # info dict of the most recent cem_plan call, for logging
        self.reset()

    def reset(self):
        """Forget the current plan. Call at the start of every episode."""
        self._queue = []  # planned actions not executed yet
        self._warm_start = None  # init_mean for the next search

    def act(self, z_now: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Next action [A] from the current latent. Only plans when the queue is empty."""
        if not self._queue:
            self._replan(z_now, z_goal)
        return self._queue.pop(0)

    def _replan(self, z_now, z_goal):
        c = self.cfg
        mean, self.last_info = cem_plan(
            z_now,
            z_goal,
            self.dynamics_fn,
            self.cost_fn,
            horizon=c.horizon,
            n_samples=c.n_samples,
            n_elites=c.n_elites,
            n_iters=c.n_iters,
            action_low=self.action_low,
            action_high=self.action_high,
            init_mean=self._warm_start if c.warm_start else None,
            init_std=c.init_std,
            min_std=c.min_std,
            generator=self.generator,
        )
        k = min(c.execute_steps, c.horizon)
        self._queue = list(mean[:k])
        # Warm start: the k executed actions drop off the front, and the plan is kept H long by
        # repeating its last action at the end (the best guess for what comes after it).
        self._warm_start = torch.cat([mean[k:], mean[-1:].expand(k, -1)])
