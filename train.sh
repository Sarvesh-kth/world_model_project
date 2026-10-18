#!/usr/bin/env bash
# Train everything from scratch, or pick up where the last run stopped. From the repository root:
#   bash train.sh                                 # data/episodes, 400 episodes, 60 epochs, the SAC for 300k steps -> models/world_model, models/sac
#   RUN=data/episodes_new TAG=new bash train.sh   # another run folder and package (models/new, models/sac_new)
# Steps: collect -> run folder -> encode with V-JEPA (the slow step) -> Q -> R, D, obstacle model, cost-to-go
# and the SAC side by side -> two checks -> the packages models/<TAG> and models/sac.
# A step whose output exists is skipped, so the script can simply be run again after a stop.
set -euo pipefail
cd "$(dirname "$0")/simulation"
[ -n "${VIRTUAL_ENV:-}" ] || source ../.venv/bin/activate
TAG=${TAG:-world_model}
EPISODES=${EPISODES:-400}
EPOCHS=${EPOCHS:-60}
WORKERS=${WORKERS:-4}
SAC_STEPS=${SAC_STEPS:-300000}
DEMOS=${DEMOS:-500}
COST_SCENES=${COST_SCENES:-20000}
RUN=${RUN:-data/episodes}
MODELS=$RUN/attempts/$TAG/models
SAC=data/sac_$TAG
PACKAGE=models/$TAG
SAC_PACKAGE=$([ "$TAG" = world_model ] && echo models/sac || echo models/sac_$TAG)
LOG=$RUN/logs
mkdir -p "$LOG"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "== 1/6 collect ($(date))"
have=0
[ -d "$RUN/episodes_raw" ] && have=$(find "$RUN/episodes_raw" -maxdepth 1 -name "episode_*" | wc -l)
if [ "$have" -lt "$EPISODES" ]; then
  python -u -m data_collection.collect --config configs/full_mix.yml --episodes $((EPISODES - have)) \
    --out "$RUN/episodes_raw" --workers "$WORKERS" --seed $((1000 + have)) 2>&1 | tee -a "$LOG/collect.txt"
fi

echo "== 2/6 run folder"
python -u -m world_model.prepare_episodes --episodes "$RUN/episodes_raw" --run "$RUN"

echo "== 3/6 encode ($(date))"
states=$(python -c "import json; print(len(json.load(open('$RUN/manifest.json'))['states']))")
if [ -f "$RUN/features/meta.json" ]; then
  encoded=$(python -c "import json; print(len(json.load(open('$RUN/features/meta.json'))['keys']))")
  [ "$encoded" -eq "$states" ] || python -u -m world_model.encode --run "$RUN" --resume 2>&1 | tee -a "$LOG/encode.txt"
elif [ -d "$RUN/features" ]; then
  python -u -m world_model.encode --run "$RUN" --resume 2>&1 | tee -a "$LOG/encode.txt"
else
  python -u -m world_model.encode --run "$RUN" 2>&1 | tee -a "$LOG/encode.txt"
fi

echo "== 4/6 train ($(date))"
# Q first: D is trained through it and the SAC reads the config it writes to attempts/<TAG>/info.json
[ -f "$MODELS/readout.pt" ] || python -u -m world_model.train readout --run "$RUN" --tag "$TAG" --epochs "$EPOCHS" 2>&1 | tee "$LOG/readout.txt"
# the rest side by side; each writes its own log
pids=()
[ -f "$MODELS/reward.pt" ] || { python -u -m world_model.train reward --run "$RUN" --tag "$TAG" --epochs "$EPOCHS" > "$LOG/reward.txt" 2>&1 & pids+=($!); }
[ -f "$MODELS/dynamics.pt" ] || { python -u -m world_model.train dynamics --run "$RUN" --tag "$TAG" --epochs "$EPOCHS" --rollout-steps 8 --task-weight 1.0 --robot-sees-q > "$LOG/dynamics.txt" 2>&1 & pids+=($!); }
[ -f "$MODELS/obstacle.pt" ] || { python -u -m world_model.train_obstacle --run "$RUN" --tag "$TAG" --epochs 30 > "$LOG/obstacle.txt" 2>&1 & pids+=($!); }
[ -f "$MODELS/cost_to_go.pt" ] || { python -u -m world_model.train_cost_to_go --run "$RUN" --tag "$TAG" --scenes "$COST_SCENES" > "$LOG/cost_to_go.txt" 2>&1 & pids+=($!); }
[ -f "$SAC/summary.json" ] || { python -u -m rl.rl_baseline --out "$SAC" --models-run "$RUN/attempts/$TAG" --random-ends --obstacles 0 3 \
    --obstacle-map both --recorded-run "$RUN" --cube-noise 0.01 --net 256 256 --demos "$DEMOS" --rl-steps "$SAC_STEPS" \
    --eval-every 20000 --validation-episodes 30 --validation-empty 10 --validation-obstacles 1 3 --test-episodes 30 > "$LOG/sac.txt" 2>&1 & pids+=($!); }
echo "   ${#pids[@]} trainings running, logs in $LOG/"
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "a training failed, see $LOG/"; exit 1; }

echo "== 5/6 checks: R on real and imagined latents, grasp or miss in imagination"
python -u -m world_model.eval_reward --run "$RUN" --tag "$TAG" --split test 2>&1 | tee "$LOG/reward_test.txt" | grep -E "REAL|IMAGINED h=8" | cut -c1-200
python -u -m world_model.eval_grasp --run "$RUN" --tag "$TAG" --split test 2>&1 | tee "$LOG/grasp_test.txt" | grep -v "^saved"

echo "== 6/6 package"
mkdir -p "$PACKAGE/models" "$PACKAGE/features" "$SAC_PACKAGE/models"
cp "$MODELS"/*.pt "$PACKAGE/models/"
cp "$RUN/attempts/$TAG/info.json" "$PACKAGE/"
cp "$RUN/features/pca.npz" "$PACKAGE/features/"
python -c "import json; m = json.load(open('$RUN/features/meta.json')); m.pop('keys', None); json.dump(m, open('$PACKAGE/features/meta.json', 'w'), indent=2)"
cp "$SAC/models/sac_best.zip" "$SAC/models/validation_history.json" "$SAC_PACKAGE/models/"
cp "$SAC/pipeline.json" "$SAC_PACKAGE/"
echo "DONE ($(date)): $PACKAGE and $SAC_PACKAGE. Evaluate with: MODELS=$PACKAGE SAC=$SAC_PACKAGE bash evaluate.sh"
