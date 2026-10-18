#!/usr/bin/env bash
# Run controllers on the same fresh scenes and print the table. From the repository root:
#   bash evaluate.sh                          # mpc, rl_q and rl_true, 10 scenes each on 0 / 1 / 2 / 3 obstacles
#   bash evaluate.sh "mpc rl_true" 20         # these controllers, 20 scenes per obstacle count
#   COUNTS="0 2" bash evaluate.sh mpc 5       # only these obstacle counts
#   MODELS=models/v4 SAC=models/sac_v4 bash evaluate.sh     # other model packages
# The controllers run side by side on the same seeds into data/test_<count>obstacles_<controller>/, the
# summaries land in results/. Finished episodes are kept, so the script can be run again to continue.
set -euo pipefail
cd "$(dirname "$0")/simulation"
[ -n "${VIRTUAL_ENV:-}" ] || source ../.venv/bin/activate
METHODS=${1:-"mpc rl_q rl_true"}
SCENES=${2:-10}
COUNTS=${COUNTS:-"0 1 2 3"}
MODELS=${MODELS:-models/world_model}
SAC=${SAC:-models/sac}
SEED=${SEED:-20474010}
mkdir -p data/logs
pids=()
for method in $METHODS; do
  (
    for n in $COUNTS; do
      python -u -m controller.control_pipeline --methods "$method" --models-run "$MODELS" --baseline-run "$SAC" --seed "$SEED" \
        --random-ends --obstacles "$n" "$n" --test-episodes "$SCENES" --headless --resume --out "data/test_${n}obstacles_$method"
    done
  ) > "data/logs/test_$method.txt" 2>&1 &
  pids+=($!)
done
echo "${#pids[@]} controllers running (logs in data/logs/test_<controller>.txt); every finished episode shows up below"
tail -n 0 -F $(for m in $METHODS; do echo "data/logs/test_$m.txt"; done) 2>/dev/null | grep --line-buffered -E "FINISHED|placements|Traceback|Error" &
watcher=$!
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
kill "$watcher" 2>/dev/null || true
[ "$failed" -eq 0 ] || { echo "a controller run failed, see data/logs/"; exit 1; }
python results_table.py test
