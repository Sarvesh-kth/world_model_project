# Comparison row: the pure version without CARRY_STEADY, to see what the steadiness cost adds.
# Everything not set here comes from controller/mpc_config.py.
#   python -m controller.control_pipeline --methods mpc --mpc-config controller/mpc_variants/pure_no_steady.py
from controller.mpc_config import *  # noqa: F401,F403

CARRY_STEADY = 0.0
