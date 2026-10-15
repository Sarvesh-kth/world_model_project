# The default build with R's and O's imagined penalties at scale 2 instead of 1 (the default before the v3c
# obstacle test: with O both gave 0 obstacle contact steps on 20 scenes, 2 placed 11/20 and 1 placed 13/20).
#   python -m controller.control_pipeline --methods mpc --random-ends --obstacles 1 2 --mpc-config controller/mpc_variants/penalty_2.py
from controller.mpc_config import *  # noqa: F401,F403

PENALTY_SCALE = 2.0
