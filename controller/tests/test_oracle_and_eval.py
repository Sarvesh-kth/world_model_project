"""Tests for the oracle dynamics, the agents and the episode runner on M1's Grade E scene."""

import numpy as np
import pytest
import torch

try:
    from controller.adapters.m1_adapter import grade_e_adapter
except FileNotFoundError:
    pytest.skip("M1's simulation/ folder not found", allow_module_level=True)

from controller.agents import OracleCEMAgent, RandomAgent, ScriptedAgent, action_bounds
from controller.config import CEMConfig
from controller.costs import reach_cost
from controller.dynamics.oracle import OracleDynamics
from controller.eval import run_episode, run_many, summarize

pytestmark = pytest.mark.m1


@pytest.fixture
def adapter():
    a = grade_e_adapter()
    a.reset(seed=0)
    yield a
    a.close()


def test_oracle_predicts_exactly_what_the_sim_does(adapter):
    """dynamics_fn on 3 samples = 3 independent copies of the real env."""
    oracle = OracleDynamics(adapter)
    z0 = oracle.observe(adapter)
    actions = torch.tensor([[1.0, 0, 0, 0, 1], [0, -1.0, 0, 0, 1], [0, 0, -1.0, 0, -1]])
    predicted = oracle(z0.expand(3, -1), actions)
    for i in range(3):
        state = adapter.get_state()
        adapter.step(actions[i].numpy())
        assert torch.equal(predicted[i], oracle.observe(adapter)), f"sample {i} differs from the real step"
        adapter.set_state(state)


def test_action_bounds_plan_or_pin_yaw():
    low, high = action_bounds(5)
    assert low[3] == -1 and high[3] == 1 and low[0] == -1 and high[4] == 1
    low, high = action_bounds(5, plan_yaw=False)
    assert low[3] == high[3] == 0


def test_run_episode_scripted_expert_places_the_cube(adapter):
    row = run_episode(adapter, ScriptedAgent(), seed=0, task="place", max_steps=300)
    assert row["success"] == 1 and row["grasped_ever"] == 1 and row["final_obj_target_xy"] < 0.07


def test_run_many_and_summarize():
    rows = run_many(grade_e_adapter, RandomAgent, seeds=[1, 0], task="reach", max_steps=5, label="random")
    assert [r["seed"] for r in rows] == [0, 1] and all(r["label"] == "random" for r in rows)
    s = summarize(rows)
    assert s["episodes"] == 2 and s["success_rate"] == 0.0


@pytest.mark.slow
def test_oracle_cem_moves_towards_the_cube(adapter):
    cfg = CEMConfig(horizon=4, n_samples=16, n_elites=4, n_iters=2, init_std=0.3, execute_steps=2)
    obs, _ = adapter.reset(seed=0)
    agent = OracleCEMAgent(reach_cost, cfg)
    agent.reset(adapter, obs)
    start = np.linalg.norm(adapter.task_features()[0:3] - adapter.task_features()[3:6])
    for _ in range(8):
        obs, *_ = adapter.step(agent.act(adapter, obs))
    end = np.linalg.norm(adapter.task_features()[0:3] - adapter.task_features()[3:6])
    # a deliberately tiny planner, to stay fast: this checks direction, the experiment checks success
    assert end < start - 0.05, f"gripper-object distance {start:.3f} -> {end:.3f}"
