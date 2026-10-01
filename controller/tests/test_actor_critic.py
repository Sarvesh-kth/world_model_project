"""Tests for the actor-critic skeleton on the toy 2D point (differentiable dynamics z + a)."""

import torch

from controller.actor_critic import Actor, ActorCritic, lambda_returns
from controller.dynamics.toy import point_dynamics

GOAL = torch.tensor([3.0, -2.0])


def reward_fn(s, a):
    return -(s - GOAL).norm(dim=-1)


def test_actor_actions_are_bounded():
    a, entropy = Actor(2, 2)(torch.randn(64, 2) * 10)
    assert a.shape == (64, 2) and a.abs().max() <= 1 and entropy.shape == (64,)


def test_lambda_returns_with_lambda_one_is_discounted_sum():
    rewards, values = torch.ones(3, 1), torch.zeros(4, 1)
    assert torch.allclose(lambda_returns(rewards, values, gamma=0.5, lam=1.0)[0], torch.tensor([1.75]))


def test_update_runs_and_imagined_return_improves():
    torch.manual_seed(0)
    ac = ActorCritic(2, 2, point_dynamics, reward_fn, horizon=8)
    s0 = torch.zeros(64, 2)
    first = ac.update(s0)["imagined_reward"]
    for _ in range(300):
        last = ac.update(s0)
    assert all(torch.isfinite(torch.tensor(list(last.values()))))
    # mean imagined reward = minus the mean distance to the goal over the rollout
    assert last["imagined_reward"] > first + 0.5, (first, last)
