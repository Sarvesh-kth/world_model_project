"""Tests for CEMPlanner, the loop around cem_plan.

cem_plan is swapped for a fake that returns known plans, so these check only the loop logic:
how many actions run per plan, what the warm start is, what reset() does. They pass before
cem_plan's body exists.
"""

import pytest
import torch

from controller import planner as planner_module
from controller.config import CEMConfig
from controller.planner import CEMPlanner

H = 5


@pytest.fixture
def fake_cem(monkeypatch):
    """Replaces cem_plan inside planner.py and records the keyword arguments of every call.

    Plan number p is [100p, 100p + 1, ..., 100p + H - 1] (in both action dims), so every executed
    action shows which plan and which step it came from.
    """
    calls = []

    def fake(z0, z_goal, dynamics_fn, cost_fn, **kwargs):
        steps = 100 * len(calls) + torch.arange(H, dtype=torch.float32)
        calls.append(kwargs)
        return steps[:, None].repeat(1, 2), {"best_cost": [0.0]}

    monkeypatch.setattr(planner_module, "cem_plan", fake)
    return calls


def make_planner(**cfg):
    return CEMPlanner(None, None, action_low=[-1, -1], action_high=[1, 1], cfg=CEMConfig(horizon=H, **cfg))


def run(planner, n):
    """Call act() n times; return the first component of each action."""
    return [planner.act(torch.zeros(2), torch.zeros(2))[0].item() for _ in range(n)]


def test_executes_k_actions_per_plan(fake_cem):
    planner = make_planner(execute_steps=2)
    assert run(planner, 5) == [0, 1, 100, 101, 200]
    assert len(fake_cem) == 3


def test_mpc_replans_every_step(fake_cem):
    planner = make_planner(execute_steps=1)
    assert run(planner, 3) == [0, 100, 200]


def test_warm_start_is_unexecuted_rest_of_plan(fake_cem):
    planner = make_planner(execute_steps=2)
    run(planner, 3)  # plan 0, execute 2 of its actions, then plan 1
    assert fake_cem[0]["init_mean"] is None  # nothing to warm-start from yet
    # plan 0 was [0, 1, 2, 3, 4]; 0 and 1 ran, so [2, 3, 4] is left, padded with its last action
    expected = torch.tensor([2.0, 3.0, 4.0, 4.0, 4.0])[:, None].repeat(1, 2)
    assert torch.equal(fake_cem[1]["init_mean"], expected)


def test_reset_forgets_the_plan(fake_cem):
    planner = make_planner(execute_steps=3)
    run(planner, 1)
    planner.reset()
    run(planner, 1)
    assert len(fake_cem) == 2  # replanned right after reset, didn't use the old queue
    assert fake_cem[1]["init_mean"] is None


def test_no_warm_start_when_disabled(fake_cem):
    planner = make_planner(execute_steps=1, warm_start=False)
    run(planner, 3)
    assert all(call["init_mean"] is None for call in fake_cem)


def test_passes_config_to_cem(fake_cem):
    planner = make_planner(n_samples=77, n_elites=7, n_iters=2, init_std=0.3, min_std=0.02)
    run(planner, 1)
    call = fake_cem[0]
    assert (call["horizon"], call["n_samples"], call["n_elites"], call["n_iters"]) == (H, 77, 7, 2)
    assert (call["init_std"], call["min_std"]) == (0.3, 0.02)
