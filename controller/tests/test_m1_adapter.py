"""Tests for the M1 adapter on the fixed Grade E scene. Skipped when M1's code isn't there."""

import numpy as np
import pytest

try:
    from controller.adapters.m1_adapter import grade_e_adapter
except FileNotFoundError:
    pytest.skip("M1's simulation/ folder not found", allow_module_level=True)

pytestmark = pytest.mark.m1


def rollout(adapter, actions):
    """Step through actions; return proprio and object state after each step."""
    out = []
    for a in actions:
        obs, *_ = adapter.step(a)
        out.append(np.concatenate([obs["proprio"], obs["state"]]))
    return np.stack(out)


@pytest.fixture
def adapter():
    a = grade_e_adapter()
    a.reset(seed=3)
    yield a
    a.close()


def test_reset_and_step_shapes(adapter):
    obs, reward, terminated, truncated, info = adapter.step(np.zeros(adapter.action_dim))
    assert adapter.action_dim == 5
    assert obs["proprio"].shape == (20,) and obs["state"].shape == (13,)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_set_state_replays_exactly(adapter):
    """Save, run 15 actions, restore, run them again: bit-identical."""
    rng = np.random.default_rng(0)
    rollout(adapter, rng.uniform(-1, 1, (10, 5)))  # get away from the reset pose first
    state = adapter.get_state()
    actions = rng.uniform(-1, 1, (15, 5))
    first = rollout(adapter, actions)
    adapter.set_state(state)
    assert np.array_equal(rollout(adapter, actions), first)


def test_clone_replays_exactly(adapter):
    """The oracle imagines in a second env: restoring the state there must give the same future."""
    rng = np.random.default_rng(1)
    rollout(adapter, rng.uniform(-1, 1, (10, 5)))
    state = adapter.get_state()
    actions = rng.uniform(-1, 1, (15, 5))
    real = rollout(adapter, actions)
    other = adapter.clone()
    other.set_state(state)
    assert np.array_equal(rollout(other, actions), real)
    other.close()


def test_task_features_match_observation(adapter):
    obs, *_ = adapter.step(np.zeros(5))
    f = adapter.task_features()
    assert np.allclose(f[0:3], obs["proprio"][14:17])  # ee position
    assert np.allclose(f[3:6], obs["state"][0:3])  # object position
    assert np.allclose(f[6:8], [0.1, 0.27])  # the Grade E place position


@pytest.mark.slow
def test_goal_frames_show_success_and_restore_state(adapter):
    before = adapter.get_state()
    frames, goal_proprio, success = adapter.goal_frames(n=64)
    assert success, "M1's scripted expert should solve the fixed Grade E scene"
    assert frames.shape == (64, 256, 256, 3) and frames.dtype == np.uint8
    assert goal_proprio.shape == (20,)
    assert np.array_equal(adapter.get_state(), before), "goal_frames must leave the env as it was"
