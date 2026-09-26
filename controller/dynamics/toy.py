"""Toy dynamics for testing the planner: a point that moves by exactly the action.

The right answer is known in closed form, so tests can check the planner precisely.
"""

import torch


def point_dynamics(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """z_next = z + a.

    z is [N, D] (a point) or [N, T, D] (T copies of it, a stand-in for patch tokens); a is [N, D].
    The action is added to every token.
    """
    return z + a.view(a.shape[0], *([1] * (z.dim() - 2)), a.shape[-1])
