# Comparison row: the planner looks 16 steps ahead instead of 8, beyond D's training windows.
# Everything not set here comes from controller/mpc_config.py.
#   python -m controller.control_pipeline --methods mpc --layout configs/middle_wall_layout.json --mpc-config controller/mpc_variants/horizon_16.py
from controller.mpc_config import *  # noqa: F401,F403

HORIZON = 16
