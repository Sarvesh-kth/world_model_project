"""Tests for cem_plan() on toy problems where the right answer is known.

The toy world is a 2D point that moves by exactly the action (dynamics/toy.py). They fail with
NotImplementedError until cem_plan's body is written.

(a), (b), (c) are the three scenarios from the task list; the rest check the contract in
cem_plan's docstring.
"""

import torch

from controller.cem import cem_plan
from controller.config import CEMConfig
from controller.costs import circle_penalty, goal_distance_cost, mean_goal_distance_cost
from controller.dynamics.toy import point_dynamics
from controller.planner import CEMPlanner

SEED = 0
TOL = 0.25  # how close to the goal counts as reached
LOW = torch.tensor([-1.0, -1.0])
HIGH = torch.tensor([1.0, 1.0])
SETTINGS = dict(horizon=10, n_samples=300, n_elites=30, n_iters=5)
START = torch.zeros(2)
GOAL = torch.tensor([5.0, -3.0])  # 5 steps away at full speed, well inside a 10-step horizon


def plan(z0=START, z_goal=GOAL, dynamics_fn=point_dynamics, cost_fn=goal_distance_cost, device="cpu", seed=SEED, **overrides):
    """cem_plan with the toy settings above. Keyword arguments override any of them."""
    device = torch.device(device)
    kwargs = {**SETTINGS, "action_low": LOW, "action_high": HIGH, **overrides}
    kwargs["action_low"] = kwargs["action_low"].to(device)
    kwargs["action_high"] = kwargs["action_high"].to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    return cem_plan(z0.to(device), z_goal.to(device), dynamics_fn, cost_fn, generator=generator, **kwargs)


def segment_distances(path, point):
    """Closest distance from `point` to each straight segment path[i] -> path[i + 1]."""
    a, b = path[:-1], path[1:]
    ab = b - a
    t = ((point - a) * ab).sum(dim=-1) / (ab * ab).sum(dim=-1).clamp_min(1e-12)
    closest = a + t.clamp(0, 1)[:, None] * ab
    return (closest - point).norm(dim=-1)


# --- the three scenarios ---------------------------------------------------------------------


def test_a_reaches_goal():
    """(a) One plan, executed open-loop, ends within TOL of the goal."""
    best_mean, _ = plan()
    final = START + best_mean.sum(dim=0)  # z + a, applied H times
    assert (final - GOAL).norm() < TOL


def test_b_goes_around_obstacle():
    """(b) A circular obstacle sits straight between start and goal and is penalized over the
    whole imagined trajectory. Closed loop: the executed path must reach the goal without any
    of its segments entering the circle.

    The goal term scores every imagined state, not just the last: with a final-state-only cost the
    closed loop wanders near the goal instead of settling (see mean_goal_distance_cost)."""
    goal = torch.tensor([6.0, 0.0])
    center, radius = torch.tensor([3.0, 0.0]), 1.0

    def cost_fn(trajectory, z_goal):
        # margin 0.3 keeps the states far enough out that a segment between two of them
        # (at most sqrt(2) long) can't clip the circle: 1.3^2 - (sqrt(2) / 2)^2 > 1^2
        return mean_goal_distance_cost(trajectory, z_goal) + 10.0 * circle_penalty(trajectory, center, radius, margin=0.3)

    planner = CEMPlanner(point_dynamics, cost_fn, LOW, HIGH, cfg=CEMConfig(**SETTINGS, execute_steps=2), seed=SEED)
    path = [START]
    for _ in range(30):
        path.append(path[-1] + planner.act(path[-1], goal))
        if (path[-1] - goal).norm() < TOL:
            break
    path = torch.stack(path)

    assert (path[-1] - goal).norm() < TOL, f"ended at {path[-1].tolist()}, goal {goal.tolist()}"
    closest = segment_distances(path, center).min()
    assert closest > radius, f"path enters the obstacle: closest approach {closest:.3f} < radius {radius}"


def test_c_token_latents_work_like_vectors():
    """(c) Latents shaped like patch tokens [T, D] work the same as vectors [D]: the planner only
    stacks latents, it never looks inside them. Here T = 4 copies of the 2D point."""
    T = 4
    z0, goal = START.expand(T, 2), GOAL.expand(T, 2)
    seen = []

    def spy_cost(trajectory, z_goal):
        seen.append(trajectory)
        return goal_distance_cost(trajectory, z_goal)

    best_mean, _ = plan(z0, goal, cost_fn=spy_cost)

    assert best_mean.shape == (10, 2)
    trajectory = seen[0]
    assert trajectory.shape == (300, 11, T, 2)  # [N, H+1, *L]
    assert torch.equal(trajectory[:, 0], z0.expand(300, T, 2)), "trajectory[:, 0] must be z0 for every sample"
    final = z0 + best_mean.sum(dim=0)
    assert (final - goal).norm(dim=-1).max() < TOL


# --- the contract in the docstring -----------------------------------------------------------


def test_output_shapes_and_info():
    best_mean, info = plan()
    assert best_mean.shape == (10, 2) and best_mean.dtype == torch.float32
    assert len(info["best_cost"]) == 5 and len(info["elite_cost"]) == 5
    assert info["std"].shape == (10, 2)


def test_cost_improves_over_iterations():
    _, info = plan()
    assert info["elite_cost"][-1] < info["elite_cost"][0]


def test_every_sampled_action_is_within_bounds():
    """Asymmetric bounds; a spy dynamics_fn sees every action that gets rolled out."""
    low, high = torch.tensor([-0.2, 0.3]), torch.tensor([0.5, 0.9])
    seen = []

    def spy(z, a):
        seen.append(a.clone())
        return point_dynamics(z, a)

    best_mean, _ = plan(dynamics_fn=spy, action_low=low, action_high=high)
    actions = torch.cat(seen)
    assert (actions >= low).all() and (actions <= high).all(), "samples must be clipped before the rollout"
    assert (best_mean >= low).all() and (best_mean <= high).all()


def test_dynamics_called_once_per_step_with_all_samples():
    """One batched call per imagined step per iteration, never one call per sample."""
    calls = []

    def spy(z, a):
        calls.append((tuple(z.shape), tuple(a.shape)))
        return point_dynamics(z, a)

    plan(dynamics_fn=spy, horizon=6, n_samples=50, n_elites=5, n_iters=3)
    assert len(calls) == 6 * 3
    assert all(call == ((50, 2), (50, 2)) for call in calls)


def test_same_seed_gives_same_plan():
    a, _ = plan(seed=1)
    b, _ = plan(seed=1)
    c, _ = plan(seed=2)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_warm_start_is_used():
    """init_mean must be where the search starts: with a tiny std and one iteration it can't get
    far from it, even though it points away from the goal."""
    init = torch.full((10, 2), 0.5)  # heads towards (+5, +5)
    best_mean, _ = plan(z_goal=torch.tensor([-5.0, -5.0]), n_iters=1, init_mean=init, init_std=0.01, min_std=0.01)
    assert torch.allclose(best_mean, init, atol=0.05)


def test_std_never_below_floor():
    """With only 5 elites, 10 refits shrink the std to about 0.01 without a floor (measured on a
    reference CEM, every seed); the floor must hold it at 0.2."""
    _, info = plan(n_elites=5, n_iters=10, min_std=0.2)
    assert (info["std"] >= 0.2 - 1e-6).all()


def test_runs_on_device(device):
    """Runs once per available device (cpu, plus mps / cuda, see conftest.py)."""
    best_mean, _ = plan(device=device)
    assert best_mean.device.type == device.type
    assert (START + best_mean.sum(dim=0).cpu() - GOAL).norm() < TOL
