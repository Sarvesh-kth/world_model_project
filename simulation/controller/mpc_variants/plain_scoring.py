# Comparison row: the plain jepa_mpc scoring with the SAC removed, what the standalone planner started from. The
# planner decides everything; no pull, no checks on what may be imagined. Everything not set here comes from
# controller/mpc_config.py.
#   python -m controller.control_pipeline --methods mpc --mpc-config controller/mpc_variants/plain_scoring.py
from controller.mpc_config import *  # noqa: F401,F403

GRIPPER = "sampled"
SCORE = "reward"
REACH_PULL = 0.0
PULL_WHOLE_WAY = False
CUBE_STAYS_PUT = False
HELD_NEEDS_CLOSED = False
GRASP_NEEDS_CONTACT = False
PLACED_AFTER = 15
PLACED_RADIUS = 1.0
PLACED_CLEARANCE = 0.0
EMPTY_GRIP_COST = 0.0
CARRY_STEADY = 0.0
RETREAT_AFTER_PLACE = False
