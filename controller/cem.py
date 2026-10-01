"""Cross-Entropy Method (CEM) over action sequences.

cem_plan() is one CEM search. It knows nothing about environments, images or world models, only
dynamics_fn and cost_fn (interfaces.py §2). The loop that calls it every few steps and executes
the result is CEMPlanner in planner.py.

Signature, docstring and body: Claude (Calle asked Claude to write the body on 2026-10-01).
"""

import torch

from controller.interfaces import CostFn, DynamicsFn


def cem_plan(
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    dynamics_fn: DynamicsFn,
    cost_fn: CostFn,
    horizon: int,
    n_samples: int,
    n_elites: int,
    n_iters: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    init_mean: torch.Tensor | None = None,
    init_std: float | torch.Tensor = 0.5,
    min_std: float = 0.05,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """Search for a low-cost sequence of `horizon` actions starting from latent z0.

    Keeps a diagonal Gaussian (mean, std) over action sequences [H, A]. Each iteration samples
    n_samples sequences, clips them to the action bounds, rolls each out with dynamics_fn from z0,
    scores the imagined trajectories with cost_fn, and refits mean and std to the n_elites
    sequences with the lowest cost.

    Args:
        z0: current latent [*L], any shape: a vector [D], patch tokens [T, D], a sim state...
            Never looked inside, only stacked into batches.
        z_goal: goal latent, handed to cost_fn untouched.
        dynamics_fn: (z [N, *L], a [N, A]) -> z_next [N, *L]. Called exactly horizon * n_iters
            times per cem_plan call: once per imagined step per iteration, each time with all
            n_samples sequences in one batch.
        cost_fn: (trajectory [N, H+1, *L], z_goal) -> cost [N], lower is better.
            trajectory[:, 0] is z0 (for every sample), trajectory[:, t] the state after t actions.
        horizon: H, number of actions per sequence.
        n_samples: N, sequences sampled per iteration.
        n_elites: K, number of lowest-cost sequences the Gaussian is refit to, 2 <= K <= N.
        n_iters: number of sample-and-refit rounds, >= 1.
        action_low, action_high: bounds per action dimension, shape [A]; A is read from these.
            Every sampled action is clipped into [low, high] before it is rolled out.
        init_mean: starting mean [H, A]. The planner passes the unexecuted rest of its previous
            plan here (warm start). None = the middle of the bounds.
        init_std: starting std, a float or anything that broadcasts to [H, A].
        min_std: floor applied to the std after every refit, so the search can't collapse
            onto one sequence too early.
        generator: torch.Generator used for all sampling, on z0's device. The same generator
            state gives the same plan.

    Returns:
        best_mean: the final mean [H, A]. It lies within the bounds, since it is an average of
            clipped samples. The planner executes its first action(s).
        info: dict with at least
            "best_cost": list of n_iters floats, the lowest sampled cost in each iteration
            "elite_cost": list of n_iters floats, the mean cost of the elites in each iteration
            "mean_cost": list of n_iters floats, the mean cost of all samples in each iteration
                (mean minus best shows whether the model tells good and bad plans apart at all)
            "std": the final std [H, A], after the floor

    Every tensor created here lives on z0.device; actions are float32.
    """
    device = z0.device
    low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
    high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
    action_dim = low.shape[0]

    # 1. The Gaussian over action sequences [H, A]: start from the warm start, or the middle of the bounds
    if init_mean is None:
        mean = ((low + high) / 2).expand(horizon, action_dim).clone()
    else:
        mean = init_mean.to(device=device, dtype=torch.float32).clone()
    std = torch.as_tensor(init_std, dtype=torch.float32, device=device).expand(horizon, action_dim).clone()

    best_cost, elite_cost, mean_cost = [], [], []
    for _ in range(n_iters):
        # 2. Sample N sequences around the mean and clip them into the action bounds
        noise = torch.randn(n_samples, horizon, action_dim, generator=generator, device=device)
        actions = torch.clamp(mean + std * noise, low, high)  # [N, H, A]

        # 3. Imagine: roll every sequence through the model, one batched call per step
        z = z0.expand(n_samples, *z0.shape)  # the same start state for all N samples
        trajectory = [z]
        for t in range(horizon):
            z = dynamics_fn(z, actions[:, t])
            trajectory.append(z)
        trajectory = torch.stack(trajectory, dim=1)  # [N, H+1, *L]

        # 4. Score every imagined trajectory and keep the K cheapest (the elites)
        cost = cost_fn(trajectory, z_goal)  # [N]
        elite_idx = torch.topk(cost, n_elites, largest=False).indices
        elites = actions[elite_idx]  # [K, H, A]

        # 5. Refit the Gaussian to the elites; the floor stops the std from collapsing to zero
        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp_min(min_std)

        best_cost.append(cost.min().item())
        elite_cost.append(cost[elite_idx].mean().item())
        mean_cost.append(cost.mean().item())

    return mean, {"best_cost": best_cost, "elite_cost": elite_cost, "mean_cost": mean_cost, "std": std}
