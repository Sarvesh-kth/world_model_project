#!/usr/bin/env bash
# Merge the three encoded runs into one and train Q, R and D on it, then score R on imagined latents.
#   usage: bash world_model/run_combined.sh [run] [tag] [epochs]
set -euo pipefail
RUN=${1:-data/combined_test1}
TAG=${2:-combined}
EPOCHS=${3:-30}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if [ ! -f "$RUN/manifest.json" ]; then
  echo "=== merge"
  python -m world_model.merge_runs --runs data/full_test1 data/obstacles_test1_fullpca data/episodes_test1_fullpca --out "$RUN"
fi

mkdir -p "$RUN/logs"
for stage in readout reward dynamics; do
  if [ ! -f "$RUN/attempts/$TAG/models/$stage.pt" ]; then
    echo "=== train $stage ($TAG)"
    extra=""
    [ "$stage" = dynamics ] && extra="--rollout-steps 8 --task-weight 1.0 --robot-sees-z"
    python -u -m world_model.train $stage --run "$RUN" --tag "$TAG" --epochs "$EPOCHS" $extra 2>&1 | tee "$RUN/logs/${TAG}_train_$stage.txt"
  fi
done

for split in val test; do
  echo "=== penalty head on real and imagined latents ($split)"
  python -u -m world_model.eval_reward --run "$RUN" --tag "$TAG" --split $split 2>&1 | tee "$RUN/logs/${TAG}_reward_$split.txt"
done
echo "DONE $TAG"
