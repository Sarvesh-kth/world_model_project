#!/usr/bin/env bash
# Test 1 combined models: merge the full-task run, the obstacle run and the full-mix run (all encoded with the
# full_test1 PCA basis), train Q / penalty head R / task-consistent D on everything, score R on imagined latents.
#   usage: bash world_model/vision/run_combined.sh data/combined_test1 combined [epochs]
set -euo pipefail
RUN=${1:-data/combined_test1}
TAG=${2:-combined}
EPOCHS=${3:-30}
PY=python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ ! -f "$RUN/manifest.json" ]; then
  echo "=== merge"
  $PY -m world_model.vision.merge_runs --runs data/full_test1 data/obstacles_test1_fullpca data/episodes_test1_fullpca --out "$RUN"
fi
LOG=$RUN/logs; mkdir -p "$LOG"
for role in readout reward dynamics; do
  if [ ! -f "$RUN/attempts/$TAG/models/$role.pt" ]; then
    echo "=== train $role ($TAG)"
    extra=""; [ "$role" = dynamics ] && extra="--rollout-steps 8 --task-weight 1.0 --robot-sees-z"
    $PY -u -m world_model.vision.train $role --run "$RUN" --tag "$TAG" --epochs "$EPOCHS" $extra 2>&1 | tee "$LOG/${TAG}_train_$role.txt" | grep -v "epoch 0[0-9][0-9] " || true
  fi
done
for split in val test; do
  echo "=== penalty head on real and imagined latents ($split)"
  $PY -u -m world_model.vision.eval_reward --run "$RUN" --tag "$TAG" --split $split 2>&1 | tee "$LOG/${TAG}_reward_$split.txt" | grep -v "Warning\|warn" || true
done
echo "DONE combined $TAG"
