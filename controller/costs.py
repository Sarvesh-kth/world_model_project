"""Cost functions over imagined trajectories.

Every cost has the cost_fn signature from interfaces.py §2, or is a term to add to one:
(trajectory [N, H+1, *L], z_goal [*L]) -> cost [N], lower is better.
"""

import torch


def goal_distance_cost(trajectory: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from the final imagined state to the goal, over all latent dims.

    Works for any latent shape; for patch tokens it is the distance between the flattened latents.
    """
    diff = trajectory[:, -1] - z_goal
    return diff.flatten(start_dim=1).norm(dim=1)



def mean_goal_distance_cost(trajectory: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Mean distance to the goal over every imagined state after the start.

    Unlike goal_distance_cost this rewards arriving early. With a final-state-only cost a
    receding-horizon planner never commits: every plan arrives "at step H", the first actions
    (the only ones executed) are nearly unconstrained, and near the goal it wanders instead of
    settling. Toy obstacle test, 30 seeds: final-only reached the goal 5/30, this 30/30.
    """
    diff = trajectory[:, 1:] - z_goal
    return diff.flatten(start_dim=2).norm(dim=2).mean(dim=1)

def circle_penalty(trajectory: torch.Tensor, center: torch.Tensor, radius: float, margin: float = 0.0) -> torch.Tensor:
    """How far a 2D trajectory [N, H+1, 2] intrudes into a circle, summed over all its states.

    0 when every state keeps at least radius + margin from the center. Only the states are
    checked, not the straight segments between them, so pick a margin that covers a step:
    a segment of length L between two points at distance >= r + m stays outside r if
    (r + m)^2 - (L / 2)^2 >= r^2.
    """
    dist = (trajectory - center).norm(dim=-1)  # [N, H+1]
    return torch.relu(radius + margin - dist).sum(dim=1)
