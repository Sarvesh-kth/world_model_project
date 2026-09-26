"""Where M3 meets the rest of the project.

Section 1: what M3 needs from M1 (simulation/) and M2 (jepa_model/).
    Some values are ASSUMPTIONS (not agreed with the team), some are OBSERVED in M1's code at
    commit bb60e1a (unchanged at c3b66ea). Reconcile as M1's and M2's code evolves: replace
    assumptions with their definitions, or agree to make these the shared ones.
Section 2: contracts M3 owns (dynamics_fn, cost_fn). These stay as they are.

Shapes use N = number of samples, H = planning horizon, A = action dim, *L = latent shape.
"""

from typing import Any, Callable, Protocol

import numpy as np
import torch

# ---------------------------------------------------------------------------------------------
# Section 1: M1 / M2 (reconcile as their code evolves)
# ---------------------------------------------------------------------------------------------

M1_COMMIT = "bb60e1a"  # M1 commit these OBSERVED values were read from

# Action. OBSERVED in M1's configs/default.yml and environment/control.py.
# (Calle's original assumption was 4-dim [dx, dy, dz, gripper], ~2 cm/step, -1 = open.)
ACTION_NAMES = ("dx", "dy", "dz", "dyaw", "gripper")  # 4-dim without dyaw if control.yaw.enabled is false
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
MAX_DELTA_M = 0.04  # metres of end-effector motion per step at |action| = 1
MAX_DYAW_RAD = 0.15  # radians of yaw per step at |action| = 1
GRIPPER_OPEN_IF_POSITIVE = True  # action[-1] > 0 opens, <= 0 closes; binary in effect
CONTROL_HZ = 10  # OBSERVED, matches the assumption

# Cameras. OBSERVED: both exist in M1's env, rendered with env.render(camera).
CAMERAS = ("static", "wrist")
IMAGE_SIZE = (256, 256)  # (height, width), uint8 RGB

# Latent. ASSUMPTION: unknown until M2 decides, pooled [D] or patch tokens [T, D] (D = 1024 for
# ViT-L). The planner must not depend on it, so there is deliberately no constant here.


class EnvAdapter(Protocol):
    """What M3 needs from an environment. adapters/m1_adapter.py implements it for M1's
    PickPlaceEnv (simulation/environment/env.py)."""

    action_dim: int

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """Start an episode. Returns (obs, info), like Gymnasium."""
        ...

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Apply one action [A] in [-1, 1]. Returns (obs, reward, terminated, truncated, info).

        terminated = the episode really ended (success or failure). truncated = time limit. The
        actor-critic (Grade C) needs the difference to bootstrap correctly at truncation.
        """
        ...

    def get_state(self) -> Any:
        """Snapshot everything that determines the future: sim state and any Python-side state
        (M1: the arm controller's joint targets / yaw / gripper flag, and the reward's stage)."""
        ...

    def set_state(self, state: Any) -> None:
        """Restore a get_state() snapshot. Replaying the same actions must then reproduce the
        same trajectory exactly (the oracle test relies on it)."""
        ...

    def render(self, camera: str = "static") -> np.ndarray:
        """Current image [H, W, 3] uint8 from one of CAMERAS."""
        ...

    def goal_image(self, camera: str = "static") -> np.ndarray:
        """Image of the goal: the object resting at the target. ASSUMPTION: the arm stays at its
        reset pose. M1 has no such function yet, so the adapter builds it."""
        ...

    def task_info(self) -> dict[str, Any]:
        """Sim-only ground truth for costs and evaluation, never for the learned pipeline.

        Keys: gripper_pos [3], object_pos [3], target_pos [3] (world frame, metres),
        grasped (bool), object_rest_z (float, object height when resting on the table).
        """
        ...


class Encoder(Protocol):
    """What M3 needs from M2's frozen encoder."""

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images [B, H, W, 3] uint8 (as rendered) -> latents [B, *L].

        ASSUMPTION: preprocessing (resize, normalize, channel order) happens inside encode().
        """
        ...


class DynamicsModel(Protocol):
    """What M3 needs from M2's dynamics model."""

    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Batched one-step prediction: z [N, *L], a [N, A] -> z_next [N, *L].

        ASSUMPTION: Markov in z. If M2's predictor conditions on several past frames, pack the
        frame window into z (L = [W, ...]); the planner doesn't change.
        """
        ...


# ---------------------------------------------------------------------------------------------
# Section 2: contracts M3 owns
# ---------------------------------------------------------------------------------------------

DynamicsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""dynamics_fn(z [N, *L], a [N, A]) -> z_next [N, *L]. One call per planning step, batched over
all N samples. Anything fits behind it: a toy function, MuJoCo itself, an MLP, M2's model."""

CostFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""cost_fn(trajectory [N, H+1, *L], z_goal [*L]) -> cost [N], lower is better.

trajectory[:, 0] is the start state z0, trajectory[:, t] the state after t actions. It gets the
whole trajectory so it can penalize intermediate states (obstacles), not just the last one."""
