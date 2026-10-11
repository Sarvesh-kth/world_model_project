# The default build with R's imagined obstacle penalties at scale 1 instead of 2: does a weaker penalty let the
# gripper reach a cube next to a wall (obstacle scenes: 4 of 6 grasps failed, closing 2-4 cm from the cube)?
#   python -m controller.control_clutter --methods mpc --new-scenes 20 --mpc-config controller/mpc_variants/penalty_1.py
from controller.mpc_config import *  # noqa: F401,F403

PENALTY_SCALE = 1.0
