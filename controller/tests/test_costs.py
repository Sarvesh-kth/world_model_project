"""Tests for the cost functions and the toy dynamics, on hand-computed values."""

import torch

from controller.costs import circle_penalty, goal_distance_cost, mean_goal_distance_cost
from controller.dynamics.toy import point_dynamics


def trajectory(*points):
    """One sample's trajectory [1, H+1, 2] from a list of 2D points."""
    return torch.tensor(points, dtype=torch.float32)[None]


def test_goal_distance_uses_only_final_state():
    t = trajectory([0, 0], [10, 10], [3, 4])
    assert torch.allclose(goal_distance_cost(t, torch.zeros(2)), torch.tensor([5.0]))


def test_mean_goal_distance_skips_start_state():
    t = trajectory([100, 0], [3, 4], [0, 0])  # the start can't be changed, so it doesn't count
    assert torch.allclose(mean_goal_distance_cost(t, torch.zeros(2)), torch.tensor([2.5]))


def test_goal_costs_accept_token_latents():
    t = torch.zeros(3, 4, 2, 5)  # N = 3, H+1 = 4, latent [T = 2, D = 5]
    t[:, -1] = 1.0
    goal = torch.zeros(2, 5)
    assert torch.allclose(goal_distance_cost(t, goal), torch.full((3,), 10**0.5))  # sqrt(T * D)
    assert mean_goal_distance_cost(t, goal).shape == (3,)


def test_circle_penalty_is_zero_outside_margin():
    t = trajectory([0, 2], [2, 0])  # both 2 from the center
    assert circle_penalty(t, torch.zeros(2), radius=1.0, margin=0.5).item() == 0.0


def test_circle_penalty_sums_intrusion_over_states():
    t = trajectory([0, 0.5], [0, 2.0], [0.2, 0])  # r + m = 1, so intrusions 0.5, 0, 0.8
    assert torch.allclose(circle_penalty(t, torch.zeros(2), radius=0.8, margin=0.2), torch.tensor([1.3]))


def test_point_dynamics_vector_and_tokens():
    a = torch.tensor([[1.0, -2.0]])
    assert torch.equal(point_dynamics(torch.zeros(1, 2), a), a)
    tokens = point_dynamics(torch.zeros(1, 3, 2), a)  # the action moves every token
    assert torch.equal(tokens, a[:, None].expand(1, 3, 2))
