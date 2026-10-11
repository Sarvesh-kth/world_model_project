#!/usr/bin/env bash
# The whole v3 chain on one machine with a GPU, from simulation/ with the venv active:
#   collect N full-mix episodes (cube and goal anywhere on the table, 0-3 obstacles, every policy kind) ->
#   run folder -> encode (the slow step) -> Q, R, D -> offline checks -> three closed-loop runs side by side.
# Every step is restartable: the collector appends, encode --resume continues, a training stage whose model
# exists is skipped (delete attempts/<tag>/models/<stage>.pt to redo it). MuJoCo renders offscreen through
# EGL; on a machine without it: export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa.
#   bash cluster_round.sh v3 400
set -euo pipefail
TAG=${1:-v3}
EPISODES=${2:-400}
RUN=data/episodes_$TAG
EPOCHS=${EPOCHS:-60}
WORKERS=${WORKERS:-4}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$RUN/logs"

have=$(ls -d "$RUN"/episodes_raw/episode_* 2>/dev/null | wc -l)
if [ "$have" -lt "$EPISODES" ]; then
  echo "=== collect $((EPISODES - have)) more episodes ($have there) $(date)"
  python -u -m data_collection.collect --config configs/full_mix.yml --episodes $((EPISODES - have)) --out "$RUN/episodes_raw" --workers "$WORKERS" --seed $((1000 + have)) 2>&1 | tee -a "$RUN/logs/collect.txt"
fi

echo "=== run folder $(date)"
python -u -m world_model.prepare_episodes --episodes "$RUN/episodes_raw" --run "$RUN"

echo "=== encode $(date)"
if [ -f "$RUN/features/meta.json" ]; then
  python -u -m world_model.encode --run "$RUN" --resume 2>&1 | tee -a "$RUN/logs/encode.txt" || true
elif [ -d "$RUN/features" ]; then
  python -u -m world_model.encode --run "$RUN" --resume 2>&1 | tee -a "$RUN/logs/encode.txt"
else
  python -u -m world_model.encode --run "$RUN" 2>&1 | tee -a "$RUN/logs/encode.txt"
fi

for stage in readout reward dynamics; do
  if [ ! -f "$RUN/attempts/$TAG/models/$stage.pt" ]; then
    echo "=== train $stage $(date)"
    extra=""
    [ "$stage" = dynamics ] && extra="--rollout-steps 8 --task-weight 1.0 --robot-sees-z"
    python -u -m world_model.train $stage --run "$RUN" --tag "$TAG" --epochs "$EPOCHS" $extra 2>&1 | tee "$RUN/logs/${TAG}_train_$stage.txt" | grep -E "validation|saved|epoch 0(01|05|10|20|30|40|50|60)"
  fi
done

echo "=== offline checks $(date)"
python -u -m world_model.eval_reward --run "$RUN" --tag "$TAG" --split test 2>&1 | tee "$RUN/logs/${TAG}_reward_test.txt" | grep -E "REAL|IMAGINED h=8" | cut -c1-200
python -u -m world_model.eval_grasp --run "$RUN" --tag "$TAG" --split test 2>&1 | tee "$RUN/logs/${TAG}_grasp_test.txt" | grep -v "^saved"

echo "=== closed loop, three runs side by side $(date)"
M="--models-run $RUN --tag $TAG --headless"
python -u -m controller.control_pipeline --methods jepa_mpc --test-episodes 20 --random-ends $M --out data/control_${TAG}_empty > data/control_${TAG}_empty.log 2>&1 &
python -u -m controller.control_pipeline --methods jepa_mpc --test-episodes 20 --random-ends --obstacles 1 3 $M --out data/control_${TAG}_obstacles > data/control_${TAG}_obstacles.log 2>&1 &
python -u -m controller.control_clutter --methods jepa_mpc $M --out data/control_${TAG}_clutter > data/control_${TAG}_clutter.log 2>&1 &
wait
for run in empty obstacles clutter; do echo "$run: $(cat ../results/control_${TAG}_${run}/summary.txt)"; done
echo "DONE $(date): models in $RUN/attempts/$TAG, summaries in results/control_${TAG}_*"
