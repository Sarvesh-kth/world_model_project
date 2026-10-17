# Calle's imagined SAC: matched empty/obstacle experiment

## Plan and scope

1. Use a separate checkout based on `calle-v3c` commit `1ad414633aeceac32f057c0d653281c178684291`. Reuse existing CUDA packages and model/data files only when their Git LFS hashes match this revision.
2. Evaluate **Calle's actual original `sac_imagined` checkpoint** on 30 empty and 30 obstacle scenes through live static/wrist JEPA → Q → SAC.
3. Continue its saved actor **and critic** weights/optimizers with fresh imagined replay, including both empty and obstacle `episodes_v3` training starts. Keep the 41-input architecture, D/Q/R, reward, grasp checks, BC regularization and other recorded training settings. Skip BC reinitialization of the already trained actor; refill replay and warm up the critic as before. Additional transition budget defaults to the original run's `steps`, overridable with `--steps`.
4. Select the checkpoint using the existing imagined validation criterion, then repeat those exact 60 evaluation scenes. Export success counts, contacts, mean/failure-only distances, per-seed paired changes and ten closest attempts. Save GIFs and trajectories locally.

The user explicitly excluded the earlier proposal-alignment redesign. No larger actor, MPC, ordinary SAC, fresh actual-data replay, new reward or world-model training is introduced. Recorded latents suffice for training; evaluation renders new images.

### What changes and what this cannot prove

- New training fixes the confirmed final-observation/reset alias in `ImaginedEnv`. The original checkpoint stays unchanged. The before/after result includes this correctness fix, additional updates and mixed obstacle starts, so it cannot isolate the effect of obstacle data alone.
- **The actor still has 41 inputs and no explicit obstacle layout.** Q supplies cube position/held probability; robot measurements, numeric goal and task memory complete the observation. Both cameras/depth reach the world model and R supplies learned penalties in imagination. This does not guarantee obstacle avoidance by an actor that cannot distinguish otherwise identical obstacle layouts.
- `--obstacles` includes existing empty + obstacle starts; it does not collect new obstacle images. Train and validation splits are preserved. Final evaluation layouts are new, never used for checkpoint selection.
- The best saved checkpoint can remain `sac_initial.zip` if extra updates do not beat its imagined validation score. This is reported, not called an improvement.
- Original exact-state diagnostic checks remain diagnostic only; final results use live cameras. No forced drops are inserted.

## Required original checkpoint

The reviewed branches **do not contain Calle's imagined SAC run**. The committed `data/rl_baseline_v1` is a different ordinary SAC and is rejected. Obtain these three files from Calle's saved run:

```
simulation/data/sac_imagined_v1/models/sac_best.zip
simulation/data/sac_imagined_v1/pipeline.json
simulation/data/sac_imagined_v1/imagined_training.json
```

The setup links this folder from the old checkout if available. If Calle provides it elsewhere:

```bash
.venv/bin/python scripts/prepare_sac_obstacles.py \
  --existing-root "$HOME/Robots&EmbodiedAI/project/world_model_project" \
  --source-imagined /absolute/path/to/Calle/saved_run
```

The metadata must identify `kind: imagined_sac` and include the actual recipe. Missing files stop preflight. Do not fabricate metadata or retrain a replacement and call it the original. Only these three files are needed from Calle; old frames and replay buffers are unnecessary.

## Block 1 — retrieve code, environment and matching data

```bash
set -e
old_repo="$HOME/Robots&EmbodiedAI/project/world_model_project"
experiment_repo="$HOME/Robots&EmbodiedAI/project/world_model_sac_obstacles"
if [ ! -d "$experiment_repo/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --single-branch --branch codex/calle-sac-obstacles \
    https://github.com/Sarvesh-kth/world_model_project.git "$experiment_repo"
fi
cd "$experiment_repo"
bash scripts/setup_sac_obstacles.sh "$old_repo"
```

Pip writes only into the new environment. Only the old virtualenv's own packages are exposed for CUDA reuse; base Conda site-packages are excluded. Re-running setup repairs environments made by the earlier script, whose base TensorFlow produced NumPy ABI errors on the cluster. Existing CUDA wheels are inherited when usable; otherwise the repo's documented Torch 2.6/torchvision 0.21 CUDA 12.4 pair is installed. SB3 stays at 2.7.1. Installed versions are recorded and cannot change during a resumed run. The old checkout/environment is not edited.

Setup retrieves frozen v3c D/Q/R/O/PCA and `episodes_v3` manifest/settings/meta/latent parts. Raw image archives, historical weights and videos are excluded. Matching existing assets are symlinked after byte checks; they must remain available and immutable. Latents are about 3 GB, plus an assembled copy if no matching existing file is available. The runner's 5 GiB free-space guard cannot detect a private user quota.

## Block 2 — pull code and run before → train → after

```bash
set -e
cd "$HOME/Robots&EmbodiedAI/project/world_model_sac_obstacles"
GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin codex/calle-sac-obstacles
cd simulation
../.venv/bin/python -m controller.sac_obstacles --preflight-only
nohup ../.venv/bin/python -u -m controller.sac_obstacles \
  > data/calle_sac_obstacles_v1_launch.log 2>&1 < /dev/null &
echo "PID: $!"
tail -f data/calle_sac_obstacles_v1_launch.log
```

Default: **120 camera episodes**, 30 per policy/condition. Empty seeds `20495010–20495039`, obstacle seeds `20496010–20496039`, one or two sampled obstacles. The first 10/20 respectively match Calle's documented benchmark ranges/sampler; this does not certify undocumented historical settings. Every full scene is saved once and reused; initial simulator states, jitter and checkpoint identities are checked. Success keeps the existing configured radius (normally 7 cm), held/lifted ≥4 cm, released/resting within 2 cm height error and 15 settled steps, with 300-step limit. Distance alone is not success.

`--evaluate-only` fills the frozen results first. Invoke later without it to continue training. `--episodes 20` gives 80 episodes. Keep all options identical across resumed invocations.

Same command resumes finished episodes and a finished fit. If training is interrupted, `--restart-training` starts another attempt **from the frozen source**, preserving old files; it is not an optimizer/replay resume. `nohup` survives terminal disconnects. A file lock rejects duplicate processes. Use a new `--out data/calle_sac_obstacles_v2` after changing code, inputs, settings or runtime versions.

## Outputs and GIFs

```
results/calle_sac_obstacles_v1/summary.txt         before/after × empty/obstacles
results/calle_sac_obstacles_v1/summary.csv         successes, failures, distances, contacts
results/calle_sac_obstacles_v1/episodes.csv        every result and file path
results/calle_sac_obstacles_v1/paired.csv          per-scene changes
results/calle_sac_obstacles_v1/closest_10.json     ten closest per condition/policy
simulation/data/calle_sac_obstacles_v1/scenes.json
simulation/data/calle_sac_obstacles_v1/training/attempt_*/
simulation/data/calle_sac_obstacles_v1/evaluation/{before,after}/{empty,obstacles}/episodes/sac_imagined/seed_*/
```

Episodes contain `actual.gif`, `trajectory.npz` (qpos for the existing replay renderer), `steps.csv`, `result.json`, `complete.json`. Per-frame JPEGs are removed after safe GIF/trajectory saving to limit disk usage; `--keep-frames` retains them. Pipeline completion is not task success.

Jupyter: show the five closest obstacle episodes after training:

```python
import json
from pathlib import Path
from IPython.display import Image, display
project = Path.home() / "Robots&EmbodiedAI/project/world_model_sac_obstacles"
run = project / "simulation/data/calle_sac_obstacles_v1"
rows = json.loads((project / "results/calle_sac_obstacles_v1/closest_10.json").read_text())
for row in rows["after/obstacles"][:5]:
    print(row["seed"], "success:", row["task_success"], "distance cm:", row["final_goal_distance_cm"])
    display(Image(data=(run / row["folder"] / "actual.gif").read_bytes(), format="gif", width=500))
```

To share small reports later: `git add -f -- results/calle_sac_obstacles_v1`, then commit/push from this branch. GIFs, NPZs and frames remain ignored and are never added by the runner.

## Verification boundary

Preflight validates original checkpoint type/dimensions, complete training recipe, dataset keys and split counts, feature/model compatibility, LFS pointers, finite latents and CUDA/SB3 availability. A resumed run rejects changed code, assets, settings or package versions. These checks do not establish placement performance or improvement. The cluster run supplies those results.
