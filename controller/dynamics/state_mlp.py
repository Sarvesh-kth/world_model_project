"""A learned world model on true sim state (task 5): the rehearsal for planning with M2's model.

State s [33] = [14 task features (interfaces.TASK_FEATURES) | joint pos 7 | joint vel 7 |
ee yaw | object quaternion 4]. Features come first, so the oracle's costs work
unchanged. An MLP predicts the change of every dynamic value from (s, action); the target
position and the object's resting height are constant and copied through.

Trained on M1's recorded episodes (data.csv per episode), with M2's alignment: the row at serial
t is the observation after action t, so (row t, action of row t+1) -> row t+1.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

ACTION_COLUMNS = ["action_dx", "action_dy", "action_dz", "action_dyaw", "action_gripper"]
CONSTANT_DIMS = [6, 7, 8, 11]  # target xyz, object resting height
STATE_DIM = 33


def object_yaw(qw, qx, qy, qz):
    """Yaw of the object's x axis, from its quaternion (upright objects)."""
    return float(np.arctan2(2 * (qx * qy + qw * qz), 1 - 2 * (qy * qy + qz * qz)))


def state_from_row(row, rest_z):
    """One data.csv row -> s [33]."""
    from controller.interfaces import TASK_FEATURES  # noqa: F401  (layout documented there)

    v = lambda *names: [float(row[n]) for n in names]  # noqa: E731
    yaw_obj = object_yaw(*v("object_qw", "object_qx", "object_qy", "object_qz"))
    err = (yaw_obj - float(row["ee_yaw"]) + np.pi) % (2 * np.pi) - np.pi
    yaw_err = (err + np.pi / 4) % (np.pi / 2) - np.pi / 4  # same wrap as M1Adapter's grasp_yaw_error
    return np.array(
        v("ee_x", "ee_y", "ee_z", "object_x", "object_y", "object_z", "place_x", "place_y", "place_z", "grasped")
        + [float(float(row["gripper_cmd"]) > 0), rest_z] + v("gripper_width") + [yaw_err]
        + v(*[f"joint_pos_{i}" for i in range(1, 8)], *[f"joint_vel_{i}" for i in range(1, 8)])
        + v("ee_yaw", "object_qw", "object_qx", "object_qy", "object_qz"),
        dtype=np.float32,
    )


def state_from_adapter(adapter, obs):
    """The same s [33] for the live env (M1Adapter + its latest observation)."""
    p, st = obs["proprio"], obs["state"]
    return np.concatenate([adapter.task_features(), p[0:14], p[17:18], st[3:7]]).astype(np.float32)


def load_episodes(episodes_dir):
    """{episode name: (states [T, 32], actions [T, 5])} from M1's episode folders."""
    out = {}
    for path in sorted(Path(episodes_dir).glob("episode_*/data.csv")):
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        rest_z = float(rows[0]["object_z"])  # the object hasn't been touched yet at serial 1
        states = np.stack([state_from_row(r, rest_z) for r in rows])
        actions = np.array([[float(r[c]) for c in ACTION_COLUMNS] for r in rows], dtype=np.float32)
        out[path.parent.name] = (states, actions)
    return out


def transitions(episodes):
    """(s_t, a_{t+1}, s_{t+1}) arrays over all episodes."""
    s, a, s2 = [], [], []
    for states, actions in episodes.values():
        s.append(states[:-1]), a.append(actions[1:]), s2.append(states[1:])
    return np.concatenate(s), np.concatenate(a), np.concatenate(s2)


class StateMLP(nn.Module):
    def __init__(self, mean, std, width=256):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        dynamic = torch.ones(STATE_DIM)
        dynamic[CONSTANT_DIMS] = 0
        self.register_buffer("dynamic", dynamic)
        self.net = nn.Sequential(nn.Linear(STATE_DIM + 5, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, STATE_DIM))

    def forward(self, s, a):
        """Next state from s [N, 32] and a [N, 5]; the net predicts the normalized change."""
        x = torch.cat([(s - self.mean) / self.std, a], dim=-1)
        next_s = s + self.net(x) * self.std * self.dynamic
        # grasped and gripper_open stay in [0, 1] (built as a new tensor: in-place breaks backprop)
        return torch.cat([next_s[..., :9], next_s[..., 9:11].clamp(0, 1), next_s[..., 11:]], dim=-1)


def multistep_error(model, starts, actions, targets):
    """Mean gripper + object position error (cm) after len(actions) open-loop steps, batched.
    starts [B, 32], actions [H, B, 5], targets [B, 32] = the true state H steps later."""
    s = starts
    for a in actions:
        s = model(s, a)
    return 100 * ((s[:, 0:3] - targets[:, 0:3]).norm(dim=1) + (s[:, 3:6] - targets[:, 3:6]).norm(dim=1)).mean().item() / 2


def rollout_batch(episodes, horizon, stride=5):
    """Start states, the recorded actions after them, and the true states `horizon` steps later."""
    starts, acts, targets = [], [], []
    for states, actions in episodes.values():
        for t0 in range(0, len(states) - horizon - 1, stride):
            starts.append(states[t0]), acts.append(actions[t0 + 1:t0 + horizon + 1]), targets.append(states[t0 + horizon])
    return (torch.from_numpy(np.stack(starts)), torch.from_numpy(np.stack(acts)).transpose(0, 1),
            torch.from_numpy(np.stack(targets)))


def train(episodes, val_names, epochs=60, lr=1e-3, batch=256, seed=0, select_horizon=8, log=print):
    """Fit StateMLP on the training episodes. Returns the model from the epoch with the lowest
    held-out `select_horizon`-step rollout error, and the history.

    Not the lowest one-step error: on the Grade E data, the epoch with the best one-step error
    (14) predicted 8-10 steps ahead worse and planned worse than later epochs, and CEM uses
    8-step rollouts."""
    torch.manual_seed(seed)
    tr = transitions({k: v for k, v in episodes.items() if k not in val_names})
    va = transitions({k: v for k, v in episodes.items() if k in val_names})
    s, a, s2 = (torch.from_numpy(x) for x in tr)
    vs, vact, vs2 = (torch.from_numpy(x) for x in va)
    std = torch.cat([s, s2]).std(dim=0).clamp_min(1e-3)
    model = StateMLP(torch.cat([s, s2]).mean(dim=0), std)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    norm_mse = lambda pred, target: (((pred - target) / std) ** 2)[:, model.dynamic.bool()].mean()  # noqa: E731
    val_rollouts = rollout_batch({k: v for k, v in episodes.items() if k in val_names}, select_horizon)
    history, best, best_state = [], float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(s)).split(batch):
            loss = norm_mse(model(s[idx], a[idx]), s2[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val, persistence = norm_mse(model(vs, vact), vs2).item(), norm_mse(vs, vs2).item()
            multi = multistep_error(model, *val_rollouts)
        history.append({"epoch": epoch, "val_mse": val, "persistence_mse": persistence, f"val_{select_horizon}step_cm": multi})
        if multi < best:
            best, best_state = multi, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            log(f"epoch {epoch:3d} val normalized MSE {val:.4f} (persistence {persistence:.4f}), "
                f"{select_horizon}-step error {multi:.2f} cm")
    model.load_state_dict(best_state)
    kept = min(history, key=lambda h: h[f"val_{select_horizon}step_cm"])
    log(f"kept epoch {kept['epoch']}: {select_horizon}-step error {best:.2f} cm, one-step MSE {kept['val_mse']:.4f}")
    return model, history


@torch.no_grad()
def rollout_errors(model, episodes, max_horizon=10, stride=5):
    """Open-loop multi-step error: from each start state, feed the recorded actions and compare.

    Returns {horizon: (ee error cm, object error cm)} averaged over start states.
    """
    errs = {h: ([], []) for h in range(1, max_horizon + 1)}
    for states, actions in episodes.values():
        for t0 in range(0, len(states) - max_horizon - 1, stride):
            s = torch.from_numpy(states[t0:t0 + 1])
            for h in range(1, max_horizon + 1):
                s = model(s, torch.from_numpy(actions[t0 + h:t0 + h + 1]))
                true = states[t0 + h]
                errs[h][0].append(100 * np.linalg.norm(s[0, 0:3].numpy() - true[0:3]))
                errs[h][1].append(100 * np.linalg.norm(s[0, 3:6].numpy() - true[3:6]))
    return {h: (float(np.mean(e)), float(np.mean(o))) for h, (e, o) in errs.items()}


def save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
    model = StateMLP(sd["mean"], sd["std"])
    model.load_state_dict(sd)
    return model.eval()


class StateMLPDynamics:
    """dynamics_fn wrapper: float32 states on the CPU, no gradients."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, s, a):
        return self.model(s.float(), a.float())
