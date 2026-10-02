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


# --- State-based pick-and-place costs --------------------------------------------------------
# For planning models whose state starts with the task features of interfaces.TASK_FEATURES
# (the oracle, the state MLP). Each scores every imagined state and averages over the plan, so
# arriving early is rewarded (see mean_goal_distance_cost). z_goal is unused: the target position
# is part of the features.

LIFT_HEIGHT = 0.10  # how high to carry the object, metres above its resting height
YAW_WEIGHT = 0.08  # cost at 45 degrees misalignment (feature = 1), while not holding the object
# (with yaw pinned, some cube yaws ended in a corner-to-corner grasp that couldn't lift)
NEAR = 0.05  # "close enough" radius for object-to-target, metres
HELD_WIDTH = 0.03  # a grasp only counts with the fingers at least this far apart, metres
# M1's `grasped` flag only checks that both fingers touch the object while closed, which a pinch
# on one edge or corner also satisfies. The oracle planner found such pinches and then couldn't
# lift. "Gripper within 2 cm of the centre" didn't separate them either (9 of 20 runs held a
# pinch and never lifted). Finger opening does: 4.46-4.6 cm around the cube body in M1's expert
# data, ~0.3 cm closed on nothing or on an edge.


def _features(trajectory):
    """Split the features of every imagined state after the start ([N, H, 14] -> named parts)."""
    from controller.interfaces import (F_EE, F_GRASPED, F_GRIPPER_OPEN, F_OBJ, F_REST_Z, F_TARGET, F_WIDTH,
                                       F_YAW_ERR, N_FEATURES)

    f = trajectory[:, 1:, :N_FEATURES].float()
    grasped = f[..., F_GRASPED].clamp(0, 1) * (f[..., F_WIDTH] > HELD_WIDTH).float()  # held by its body
    misaligned = YAW_WEIGHT * f[..., F_YAW_ERR].clamp(0, 1) * (1 - grasped)
    return (f[..., F_EE], f[..., F_OBJ], f[..., F_TARGET], grasped, f[..., F_GRIPPER_OPEN].clamp(0, 1),
            f[..., F_REST_Z], misaligned)


def reach_cost(trajectory, z_goal=None):
    """Stage (a): get the gripper to the object, fingers square to it."""
    ee, obj, *_, misaligned = _features(trajectory)
    return ((ee - obj).norm(dim=-1) + misaligned).mean(dim=1)


def lift_cost(trajectory, z_goal=None):
    """Stage (b): reach, grasp, lift LIFT_HEIGHT. Levels: lifted < grasped < not grasped, so
    closing on the object and then lifting each lower the cost. A closed gripper without a real
    grasp is penalized, so it doesn't close early or arrive with the fingers shut."""
    ee, obj, _, grasped, gripper_open, rest_z, misaligned = _features(trajectory)
    to_obj = (ee - obj).norm(dim=-1)
    lifted = ((obj[..., 2] - rest_z) / LIFT_HEIGHT).clamp(0, 1) * grasped
    closed_empty = (1 - gripper_open) * (1 - grasped)
    cost = to_obj + misaligned + 3.0 * (1 - grasped) + 2.0 * (1 - lifted) + 1.0 * closed_empty  # 5 / 2 / 0 + distance
    return cost.mean(dim=1)


def pick_place_cost(trajectory, z_goal=None):
    """Stage (c): the whole task, as three levels that each beat the one above:

        not holding the object:  3 + distance gripper -> object   (+ penalty: closed while far)
        holding it:              1 + distance object -> target (xy) + carry-height error
        placed (released on the target, resting):  0

    While holding it far from the target, the wanted height is LIFT_HEIGHT; above the target, 0
    (lower it), after which releasing turns it into "placed".
    """
    ee, obj, target, grasped, gripper_open, rest_z, misaligned = _features(trajectory)
    to_obj = (ee - obj).norm(dim=-1)
    to_target = (obj[..., :2] - target[..., :2]).norm(dim=-1)
    height = obj[..., 2] - rest_z
    want_height = torch.where(to_target > NEAR, torch.full_like(height, LIFT_HEIGHT), torch.zeros_like(height))
    placed = (1 - grasped) * (to_target < NEAR).float() * (height.abs() < 0.02).float()
    holding = 1.0 + 2.0 * to_target + 2.0 * (height - want_height).abs()
    free = 3.0 + to_obj + misaligned + 1.0 * (1 - gripper_open) + to_target  # closed without a grasp: penalized
    cost = placed * 0.0 + (1 - placed) * (grasped * holding + (1 - grasped) * free)
    return cost.mean(dim=1)


STAGE_COSTS = {"reach": reach_cost, "lift": lift_cost, "place": pick_place_cost}


# --- Latent goal cost (Level E with M2's JEPA model) ------------------------------------------


def latent_goal_cost(z_dim, z_weight=1.0, p_weight=1.0):
    """Cost for planner states s = [normalized z | normalized p] (adapters.m2_adapter.JEPADynamics).

    Per imagined state: z_weight * MSE(z, z_goal) + p_weight * MSE(p, p_goal), the same two terms
    M2 trains D with; averaged over the plan (rewards arriving early). z_goal passed to the cost
    is the goal state [z_goal | p_goal] from the encoded goal clip and the goal proprio.
    """

    def cost(trajectory, s_goal):
        diff = trajectory[:, 1:] - s_goal
        z_term = diff[..., :z_dim].square().mean(dim=-1)
        p_term = diff[..., z_dim:].square().mean(dim=-1)
        return (z_weight * z_term + p_weight * p_term).mean(dim=1)

    return cost
