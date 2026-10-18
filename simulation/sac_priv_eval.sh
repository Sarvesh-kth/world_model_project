#!/bin/bash
# The MPC against the simulator-trained SAC on Q's cube estimate (rl_q) on the same fresh random-end scenes,
# 0 / 1 / 2 / 3 obstacles, SCENES per count. Runs from simulation/ with the venv active, needs the GPU for V-JEPA.
#   bash sac_priv_eval.sh mpc 10                       # the planner first, into data/sac_priv_eval_<n>
#   bash sac_priv_eval.sh rl_q 10 data/sac_priv_w1     # then a policy, --resume adds its episodes on the same seeds
# Per-episode results are copied to results/sac_priv_eval_<n>/episodes/<method>/seed_*/result.json.
set -e
method=${1:-mpc}
scenes=${2:-10}
run=${3:-data/sac_priv_w1}
for n in 0 1 2 3; do
  out="data/sac_priv_eval_$n"
  resume=""; [ -d "$out" ] && resume="--resume"
  python -u -m controller.control_pipeline --methods "$method" --baseline-run "$run" --random-ends \
    --obstacles $n $n --test-episodes "$scenes" --headless --out "$out" $resume 2>&1 \
    | grep -v "Exception ignored\|Traceback\|File \"\|EGLError\|^$" | tee -a "data/sac_priv_eval_$n.log" | grep "FINISHED\|placements"
done
