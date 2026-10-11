# Comparison row: a hand-coded gripper (close at the cube, open over B, rise after placing) around the planner,
# which then only decides the motion (9/20 development seeds on a Mac, 2026-10-10). Not the pure version: label it
# as "MPC with a rule-based gripper". Everything not set here comes from controller/mpc_config.py.
#   python -m controller.control_pipeline --methods mpc --mpc-config controller/mpc_variants/rule_gripper.py
from controller.mpc_config import *  # noqa: F401,F403

GRIPPER = "rule"
SCORE = "reward"
RETREAT_AFTER_PLACE = True
GRASP_NEEDS_CONTACT = False
PLACED_CLEARANCE = 0.0
CARRY_HEIGHT = 0.05
CARRY_WEIGHT = 1.0
EMPTY_GRIP_COST = 0.0
PLACED_RADIUS = 1.0
PULL_WHOLE_WAY = False
HELD_Z_OFFSET = 0.0
CARRY_STEADY = 0.0
