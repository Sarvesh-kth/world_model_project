#!/usr/bin/env bash
# Train Q, R, D, the obstacle model and the cost-to-go on an encoded run, then score R on imagined latents and
# check that D / Q / R tell a grasp from a miss. Stages whose model exists are skipped unless the 4th argument is "retrain" (after more data).
#   usage: bash world_model/run_combined.sh data/episodes_v3 v3 60 [retrain]
set -euo pipefail
RUN=${1:-data/combined_v2}
TAG=${2:-combined}
EPOCHS=${3:-60}
RETRAIN=${4:-}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

mkdir -p "$RUN/logs"
for stage in readout reward dynamics; do
  if [ ! -f "$RUN/attempts/$TAG/models/$stage.pt" ] || [ "$RETRAIN" = retrain ]; then
    echo "=== train $stage ($TAG, $EPOCHS epochs)"
    extra=""
    [ "$stage" = dynamics ] && extra="--rollout-steps 8 --task-weight 1.0 --robot-sees-q"
    [ "$RETRAIN" = retrain ] && extra="$extra --overwrite"
    python -u -m world_model.train $stage --run "$RUN" --tag "$TAG" --epochs "$EPOCHS" $extra 2>&1 | tee "$RUN/logs/${TAG}_train_$stage.txt"
  fi
done

if [ ! -f "$RUN/attempts/$TAG/models/obstacle.pt" ] || [ "$RETRAIN" = retrain ]; then
  echo "=== train obstacle ($TAG)"
  python -u -m world_model.train_obstacle --run "$RUN" --tag "$TAG" --epochs 30 --overwrite 2>&1 | tee "$RUN/logs/${TAG}_train_obstacle.txt"
fi
[ -f "$RUN/attempts/$TAG/models/cost_to_go.pt" ] || python -u -m world_model.train_cost_to_go --run "$RUN" --tag "$TAG" 2>&1 | tee "$RUN/logs/${TAG}_train_cost_to_go.txt"

for split in val test; do
  echo "=== reward model on real and imagined latents ($split)"
  python -u -m world_model.eval_reward --run "$RUN" --tag "$TAG" --split $split 2>&1 | tee "$RUN/logs/${TAG}_reward_$split.txt"
done
echo "=== grasp or miss in imagination (test)"
python -u -m world_model.eval_grasp --run "$RUN" --tag "$TAG" --split test 2>&1 | tee "$RUN/logs/${TAG}_grasp_test.txt"
echo "DONE $TAG"
