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


# --- staged state costs and the latent goal cost ------------------------------------------------

from controller.costs import latent_goal_cost, lift_cost, pick_place_cost, reach_cost  # noqa: E402


def features(ee, obj, target=(0.1, 0.27, 0.75), grasped=0, gripper_open=1, rest_z=0.772):
    """A [1, 2, 12] trajectory: a dummy start state and one imagined state with these features."""
    f = torch.tensor([*ee, *obj, *target, grasped, gripper_open, rest_z], dtype=torch.float32)
    return torch.stack([torch.zeros(12), f])[None]


def test_reach_cost_is_gripper_object_distance():
    assert torch.allclose(reach_cost(features((0, 0, 0), (0.3, 0.4, 0))), torch.tensor([0.5]))


def test_lift_cost_levels():
    obj = (0.18, -0.23, 0.772)
    far_open = lift_cost(features((0.18, -0.23, 0.9), obj))
    at_open = lift_cost(features(obj, obj))
    closed_far = lift_cost(features((0.18, -0.23, 0.9), obj, gripper_open=0))
    grasped = lift_cost(features(obj, obj, grasped=1, gripper_open=0))
    lifted = lift_cost(features((0.18, -0.23, 0.872), (0.18, -0.23, 0.872), grasped=1, gripper_open=0))
    assert lifted < grasped < at_open < far_open < closed_far


def test_edge_pinch_is_not_a_grasp():
    """M1's grasped flag with the gripper 3.5 cm off-centre (a pinch on one edge) must not pay."""
    obj = (0.18, -0.23, 0.772)
    pinch = lift_cost(features((0.18, -0.195, 0.772), obj, grasped=1, gripper_open=0))
    centred = lift_cost(features(obj, obj, grasped=1, gripper_open=0))
    assert pinch > centred + 2.5


def test_pick_place_cost_levels():
    target, rest = (0.1, 0.27, 0.75), 0.772
    placed = pick_place_cost(features((0.1, 0.27, 0.9), (0.1, 0.27, rest), target))
    holding_at_target = pick_place_cost(features((0.1, 0.27, rest), (0.1, 0.27, rest), target, grasped=1, gripper_open=0))
    holding_far = pick_place_cost(features((0.18, -0.23, 0.872), (0.18, -0.23, 0.872), target, grasped=1, gripper_open=0))
    free_far = pick_place_cost(features((0, 0, 1.0), (0.18, -0.23, rest), target))
    assert placed < holding_at_target < holding_far < free_far


def test_latent_goal_cost_hand_value():
    cost = latent_goal_cost(z_dim=2, z_weight=1.0, p_weight=0.5)
    traj = torch.tensor([[[9.0, 9, 9], [1.0, 1, 2]]])  # start ignored; z = (1, 1), p = (2)
    goal = torch.zeros(3)
    assert torch.allclose(cost(traj, goal), torch.tensor([1.0 + 0.5 * 4.0]))
