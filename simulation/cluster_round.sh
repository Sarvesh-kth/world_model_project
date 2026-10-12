#!/usr/bin/env bash
# The whole v3 chain on one machine with a GPU, from simulation/ with the venv active:
#   collect N full-mix episodes (cube and goal anywhere on the table, 0-3 obstacles, every policy kind) ->
#   run folder -> encode (the slow step) -> Q, R, D, the obstacle model, the cost-to-go -> models/<tag> package ->
#   offline checks -> three closed-loop runs side by side (empty, 0-2 obstacles, held-out clutter).
# Every step is restartable: the collector appends, encode --resume continues, a training stage whose model
# exists is skipped (delete attempts/<tag>/models/<stage>.pt to redo it). MuJoCo renders offscreen through
# EGL; on a machine without it: export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa. The steps by hand are in
# world_model/README.md.
#   bash cluster_round.sh v3 400
set -euo pipefail
TAG=${1:-v3}
EPISODES=${2:-400}
RUN=data/episodes_$TAG
EPOCHS=${EPOCHS:-60}
WORKERS=${WORKERS:-4}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$RUN/logs"

# a clone that pulled the LFS data has the latents in 1.5 GB parts and the hack episodes as one tar: assemble them
f="$RUN/features/latents.npy"
if [ ! -f "$f" ] && ls "$f".part* >/dev/null 2>&1; then
  echo "=== [0/6] assembling $f from its parts"; cat "$f".part* > "$f"
fi
if [ ! -d data/episodes_hack/episodes_raw ] && [ -f data/episodes_hack/episodes_raw.tar ]; then
  echo "=== [0/6] unpacking data/episodes_hack/episodes_raw.tar"; tar xf data/episodes_hack/episodes_raw.tar -C data/episodes_hack
fi

have=$(ls -d "$RUN"/episodes_raw/episode_* 2>/dev/null | wc -l)
if [ "$have" -lt "$EPISODES" ]; then
  echo "=== [1/6] collect $((EPISODES - have)) more episodes ($have there) $(date)"
  python -u -m data_collection.collect --config configs/full_mix.yml --episodes $((EPISODES - have)) --out "$RUN/episodes_raw" --workers "$WORKERS" --seed $((1000 + have)) 2>&1 | tee -a "$RUN/logs/collect.txt"
fi

echo "=== [2/6] run folder $(date)"
python -u -m world_model.prepare_episodes --episodes "$RUN/episodes_raw" --run "$RUN"

echo "=== [3/6] encode $(date)"
if [ -f "$RUN/features/meta.json" ]; then
  python -u -m world_model.encode --run "$RUN" --resume 2>&1 | tee -a "$RUN/logs/encode.txt" || true
elif [ -d "$RUN/features" ]; then
  python -u -m world_model.encode --run "$RUN" --resume 2>&1 | tee -a "$RUN/logs/encode.txt"
else
  python -u -m world_model.encode --run "$RUN" 2>&1 | tee -a "$RUN/logs/encode.txt"
fi

for stage in readout reward dynamics; do
  if [ ! -f "$RUN/attempts/$TAG/models/$stage.pt" ]; then
    echo "=== [4/6] train $stage $(date)"
    extra=""
    [ "$stage" = dynamics ] && extra="--rollout-steps 8 --task-weight 1.0 --robot-sees-q"
    python -u -m world_model.train $stage --run "$RUN" --tag "$TAG" --epochs "$EPOCHS" $extra 2>&1 | tee "$RUN/logs/${TAG}_train_$stage.txt"
  fi
done
# the obstacle model (a window of the obstacle-only depth map around the hand) and the cost-to-go (synthetic scenes)
[ -f "$RUN/attempts/$TAG/models/obstacle.pt" ] || python -u -m world_model.train_obstacle --run "$RUN" --tag "$TAG" --epochs 30 2>&1 | tee "$RUN/logs/${TAG}_train_obstacle.txt"
[ -f "$RUN/attempts/$TAG/models/cost_to_go.pt" ] || python -u -m world_model.train_cost_to_go --run "$RUN" --tag "$TAG" 2>&1 | tee "$RUN/logs/${TAG}_train_cost_to_go.txt"
# the packaged model folder the controllers run with by default (models/, info.json, features/ without the keys)
mkdir -p "models/$TAG/models" "models/$TAG/features"
cp "$RUN/attempts/$TAG/models/"*.pt "models/$TAG/models/" && cp "$RUN/attempts/$TAG/info.json" "models/$TAG/" && cp "$RUN/features/pca.npz" "models/$TAG/features/"
python -c "import json,sys; m=json.load(open('$RUN/features/meta.json')); m.pop('keys',None); json.dump(m, open('models/$TAG/features/meta.json','w'), indent=2)"

echo "=== [5/6] offline checks $(date)"
python -u -m world_model.eval_reward --run "$RUN" --tag "$TAG" --split test 2>&1 | tee "$RUN/logs/${TAG}_reward_test.txt" | grep -E "REAL|IMAGINED h=8" | cut -c1-200
python -u -m world_model.eval_grasp --run "$RUN" --tag "$TAG" --split test 2>&1 | tee "$RUN/logs/${TAG}_grasp_test.txt" | grep -v "^saved"

echo "=== [6/6] closed loop, three runs side by side $(date)"
M="--models-run models/$TAG --headless"
python -u -m controller.control_pipeline --methods jepa_mpc --test-episodes 20 --random-ends $M --out data/control_${TAG}_empty > data/control_${TAG}_empty.log 2>&1 &
P1=$!
python -u -m controller.control_pipeline --methods jepa_mpc --test-episodes 20 --random-ends --obstacles 0 2 $M --out data/control_${TAG}_obstacles > data/control_${TAG}_obstacles.log 2>&1 &
P2=$!
python -u -m controller.control_clutter --methods jepa_mpc $M --out data/control_${TAG}_clutter > data/control_${TAG}_clutter.log 2>&1 &
P3=$!
# the three runs write to their logs; every finished episode and every error is echoed here as it lands
tail -n 0 -F data/control_${TAG}_empty.log data/control_${TAG}_obstacles.log data/control_${TAG}_clutter.log 2>/dev/null \
  | grep --line-buffered -E "FINISHED|placements|Traceback|Error" &
wait $P1 $P2 $P3
kill %tail 2>/dev/null || true
for run in empty obstacles clutter; do echo "$run: $(cat ../results/control_${TAG}_${run}/summary.txt)"; done
echo "DONE $(date): models in $RUN/attempts/$TAG, summaries in results/control_${TAG}_*"
