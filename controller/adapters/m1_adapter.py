"""M1's PickPlaceEnv (simulation/environment/env.py) behind the EnvAdapter interface.

The only file in controller/ that imports M1's code. M1's packages (`environment`,
`data_collection`) are top-level modules inside simulation/, so that folder goes on the import
path, which is what running from inside simulation/ does for M1.
"""

import json
import sys

import mujoco
import numpy as np

from controller.config import m1_sim_dir
from controller.interfaces import N_FEATURES

SIM_DIR = m1_sim_dir()
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from data_collection.scripted_policy import make_policy  # noqa: E402
from environment import EpisodeLayout, PickPlaceEnv, load_config  # noqa: E402

# The fixed Grade E scene from M2's branch: one cube, left end to right end, no obstacles
GRADE_E_CONFIG = SIM_DIR / "configs" / "grade_e.yml"
GRADE_E_LAYOUT = SIM_DIR / "configs" / "grade_e_layout.json"

# What the full-physics MuJoCo state includes: qpos, qvel, actuator activations, controls, mocap,
# warm start and time. Restoring it reproduces the next steps exactly.
_MJ_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def grasp_yaw_error(object_yaw, finger_yaw):
    """How far the fingers are from square to the nearest object face, in [-pi/4, pi/4].
    The same wrap as M1's scripted policy (ScriptedPickPlace._yaw_action): objects fit the opening
    either way, so only the angle to the nearest face matters."""
    err = (object_yaw - finger_yaw + np.pi) % (2 * np.pi) - np.pi
    return float((err + np.pi / 4) % (np.pi / 2) - np.pi / 4)


class M1Adapter:
    """One M1 env. `layout=None` samples a new random layout every reset, like M1's collector."""

    def __init__(self, config_path=None, overrides=None, layout=None):
        self.cfg = load_config(config_path, overrides)
        self.layout = layout
        self.env = PickPlaceEnv(self.cfg)
        self.action_dim = 5 if self.cfg.control.yaw.enabled else 4

    # --- Gymnasium-style episode interface ------------------------------------------------------

    def reset(self, seed=None):
        return self.env.reset(seed=seed, layout=self.layout)

    def step(self, action):
        return self.env.step(np.asarray(action, dtype=float))

    def render(self, camera="static"):
        return self.env.render(camera)

    def close(self):
        self.env.close()

    # --- Sim-only extras for the oracle, costs and evaluation -----------------------------------

    def get_state(self):
        """Everything that determines the future, as one float64 vector.

        MuJoCo's state is not enough: M1's ArmController keeps joint targets, yaw and the gripper
        command in Python, and Rewards keeps its stage flags. They are appended after the MuJoCo state.

        Side effect: refreshes the live env's positions and contacts (mj_forward). After mj_step they
        still describe the previous physics substep, while a restored state gets fresh ones; without
        the refresh, the original and the restored run would start about 1 mm apart. Physics state
        (qpos, qvel, warm start) is untouched.
        """
        e, c, r = self.env, self.env.controller, self.env.reward
        mujoco.mj_forward(e.model, e.data)
        mj = np.empty(mujoco.mj_stateSize(e.model, _MJ_SPEC))
        mujoco.mj_getState(e.model, e.data, mj, _MJ_SPEC)
        return np.concatenate([
            mj, c.target_pos, c.step_start, [c.yaw], c.q_des, [float(c.gripper_open)], c.posture, c.base_quat,
            [r.placed, r.succeeded, r.was_grasped, r.prev_grasped],
            [e.step_count, e._success_steps, e.object_rest_z],
        ]).astype(np.float64)

    def set_state(self, state):
        """Restore a get_state() vector. Works on any env built from the same layout."""
        e, c, r = self.env, self.env.controller, self.env.reward
        state = np.asarray(state, dtype=np.float64)
        n = mujoco.mj_stateSize(e.model, _MJ_SPEC)
        mujoco.mj_setState(e.model, e.data, state[:n], _MJ_SPEC)
        rest = state[n:]
        c.target_pos, c.step_start = rest[0:3].copy(), rest[3:6].copy()
        c.yaw, c.q_des, c.gripper_open = float(rest[6]), rest[7:14].copy(), bool(rest[14])
        c.posture, c.base_quat = rest[15:22].copy(), rest[22:26].copy()
        r.placed, r.succeeded, r.was_grasped, r.prev_grasped = (bool(x) for x in rest[26:30])
        e.step_count, e._success_steps, e.object_rest_z = int(rest[30]), int(rest[31]), float(rest[32])
        mujoco.mj_forward(e.model, e.data)  # recompute positions and contacts from the restored state

    def clone(self):
        """A second env with the same layout and config, for imagining rollouts (the oracle).
        Its state is unrelated until you set_state() it."""
        other = M1Adapter(layout=self.env.layout)
        other.cfg, other.env.cfg = self.cfg, self.cfg
        other.env.reset(seed=0, layout=self.env.layout)
        return other

    def task_info(self):
        """Sim-only ground truth (interfaces.EnvAdapter.task_info), read from M1's privileged state."""
        s = self.env.privileged_state()
        return {
            "gripper_pos": s["ee_pos"], "object_pos": s["object_pos"], "target_pos": s["place_pos"],
            "grasped": bool(s["grasped"]), "object_rest_z": float(s["object_rest_z"]),
            "gripper_open": bool(self.env.controller.gripper_open),
            "gripper_width": float(self.env.data.qpos[self.env.finger_qpos_ids].sum()),
            "grasp_yaw_error": grasp_yaw_error(self.env.object_long_axis_yaw(), self.env.finger_axis_yaw()),
        }

    def task_features(self):
        """task_info() as the vector laid out in interfaces.TASK_FEATURES."""
        t = self.task_info()
        f = np.concatenate([t["gripper_pos"], t["object_pos"], t["target_pos"],
                            [t["grasped"], t["gripper_open"], t["object_rest_z"], t["gripper_width"],
                             t["grasp_yaw_error"]]])
        assert len(f) == N_FEATURES
        return f.astype(np.float64)

    def goal_frames(self, n=64, camera="static"):
        """Goal observation: the last n frames of M1's scripted expert finishing the task from the
        current state. The state is restored afterwards. Returns (frames [n, H, W, 3] uint8,
        proprio at the end [20], whether the expert succeeded).

        A clip, not one image, because M2's V-JEPA latent is computed from a 64-frame clip.
        """
        saved = self.get_state()
        policy = make_policy("success", self.env, np.random.default_rng(0), self.cfg)
        policy.reset()
        frames = [self.render(camera)]
        obs, success = None, False
        for _ in range(self.cfg.episode.max_steps):
            obs, _, terminated, truncated, info = self.env.step(policy.act(), give_up=policy.done)
            frames.append(self.render(camera))
            success |= info["success"]
            if terminated or truncated:
                break
        self.set_state(saved)
        frames = frames[-n:]
        frames = [frames[0]] * (n - len(frames)) + frames  # pad at the front like M2 does
        return np.stack(frames), obs["proprio"].copy(), success

    def goal_image(self, camera="static"):
        """Last goal frame: the object resting on the target, arm where the expert left it."""
        return self.goal_frames(n=1, camera=camera)[0][-1]


def grade_e_adapter():
    """M1's env on the fixed Grade E scene (cube from the left end to the right end, no obstacles)."""
    layout = EpisodeLayout.from_dict(json.loads(GRADE_E_LAYOUT.read_text()))
    return M1Adapter(config_path=GRADE_E_CONFIG, layout=layout)
