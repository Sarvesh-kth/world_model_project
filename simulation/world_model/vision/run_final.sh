#!/usr/bin/env bash
# Full closed-loop evaluation with the trained models: 20 empty-table scenes (all four controllers), the six held-out
# obstacle scenes with rl_q and the planner, then the planner with the penalty head off.
#   usage: bash world_model/vision/run_final.sh [episodes]
set -euo pipefail
EPISODES=${1:-20}
echo "=== empty table, $EPISODES scenes"
python -u -m world_model.vision.control_pipeline --test-episodes "$EPISODES" --headless --out data/control_final --resume 2>&1 | grep "placements" || true
echo "=== obstacle scenes with the penalty head, then without"
python -u -m world_model.vision.control_clutter --methods rl_q jepa_mpc --headless --out data/control_obstacles 2>&1 | grep "placements" || true
python -u -m world_model.vision.control_clutter --methods jepa_mpc --no-penalties --headless --out data/control_obstacles_nopen 2>&1 | grep "placements" || true
echo "DONE"
