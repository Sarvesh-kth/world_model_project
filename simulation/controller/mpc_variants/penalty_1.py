# The default build with R's and O's imagined penalties at scale 1 instead of 2 (v3c jepa_mpc's default). On the
# old models (R only) scale 1 grasped no more and collided 24x more (287 contact steps vs 12 on 20 scenes).
#   python -m controller.control_clutter --methods mpc --new-scenes 20 --mpc-config controller/mpc_variants/penalty_1.py
from controller.mpc_config import *  # noqa: F401,F403

PENALTY_SCALE = 1.0
