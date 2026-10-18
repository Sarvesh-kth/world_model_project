# The default build with R's and O's imagined penalties at scale 2 instead of 1
#   python -m controller.control_pipeline --methods mpc --random-ends --obstacles 1 2 --mpc-config controller/mpc_variants/penalty_2.py
from controller.mpc_config import *  # noqa: F401,F403

PENALTY_SCALE = 2.0
