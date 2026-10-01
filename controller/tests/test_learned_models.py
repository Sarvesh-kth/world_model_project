"""Tests for the learned world models: the state MLP and the M2 (JEPA) adapter, with random
weights, so they need neither trained checkpoints nor the V-JEPA download."""

import numpy as np
import pytest
import torch

from controller.dynamics import state_mlp


def fake_episodes(n=6, length=30, seed=0):
    """Random-walk states in the StateMLP layout, constant target / rest height."""
    rng = np.random.default_rng(seed)
    eps = {}
    for i in range(n):
        s = np.cumsum(rng.normal(0, 0.01, (length, state_mlp.STATE_DIM)), axis=0).astype(np.float32)
        s[:, [6, 7, 8, 11]] = [0.1, 0.27, 0.75, 0.772]
        s[:, 9:11] = rng.integers(0, 2, (length, 2))
        eps[f"episode_{i:06d}"] = (s, rng.uniform(-1, 1, (length, 5)).astype(np.float32))
    return eps


def test_state_mlp_shapes_constants_and_clamps():
    eps = fake_episodes()
    s, a, s2 = state_mlp.transitions(eps)
    model = state_mlp.StateMLP(torch.from_numpy(s).mean(0), torch.from_numpy(s).std(0).clamp_min(1e-3))
    x = torch.from_numpy(s[:8])
    out = model(x, torch.from_numpy(a[:8]))
    assert out.shape == (8, state_mlp.STATE_DIM)
    assert torch.equal(out[:, [6, 7, 8, 11]], x[:, [6, 7, 8, 11]]), "target and rest height are copied through"
    assert out[:, 9:11].min() >= 0 and out[:, 9:11].max() <= 1


def test_state_mlp_trains_and_keeps_a_checkpoint(tmp_path):
    eps = fake_episodes()
    model, history = state_mlp.train(eps, {"episode_000000"}, epochs=3, log=lambda *_: None)
    assert len(history) == 3 and np.isfinite(history[-1]["val_mse"])
    state_mlp.save(model, tmp_path / "m.pt")
    again = state_mlp.load(tmp_path / "m.pt")
    x, a = torch.zeros(2, state_mlp.STATE_DIM), torch.zeros(2, 5)
    assert torch.allclose(model(x, a), again(x, a))


m2 = pytest.importorskip("controller.adapters.m2_adapter", exc_type=ImportError)


@pytest.fixture
def fake_checkpoint(tmp_path):
    """A randomly initialized SplitDynamics in M2's checkpoint format (world_model/train_dynamics.py)."""
    from world_model.train_dynamics import SplitDynamics

    torch.manual_seed(0)
    model = SplitDynamics(1024, 20, 5)
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "z_dim": 1024, "p_dim": 20, "a_dim": 5,
                "z_mean": torch.zeros(1024), "z_std": torch.ones(1024) * 2,
                "p_mean": torch.ones(20), "p_std": torch.ones(20),
                "architecture": "SplitDynamics", "width": 512, "p_width": 128,
                "proprio_columns": [], "action_columns": [], "encoder_model": m2.VJEPA_MODEL,
                "pooling": "mean_all_encoder_tokens", "camera": "static", "clip_frames": 64,
                "epoch": 1, "val_z_mse": 0.0, "val_p_mse": 0.0}, path)
    return path


def test_jepa_dynamics_is_a_batched_dynamics_fn(fake_checkpoint):
    d = m2.JEPADynamics(fake_checkpoint)
    s = d.state(np.zeros(1024), np.full(20, 3.0))
    assert s.shape == (1044,) and torch.allclose(d.proprio(s), torch.full((20,), 3.0))
    out = d(s.expand(7, -1), torch.zeros(7, 5))
    assert out.shape == (7, 1044)


def test_clip_buffer_pads_with_first_frame_and_rolls():
    buf = m2.ClipBuffer(n_frames=4, jpeg=False)
    first, second = np.zeros((8, 8, 3), np.uint8), np.full((8, 8, 3), 255, np.uint8)
    buf.reset(first)
    buf.push(second)
    clip = buf.clip()
    assert clip.shape == (4, 8, 8, 3) and (clip[:3] == 0).all() and (clip[3] == 255).all()
