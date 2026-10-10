# World model: data, encoder, Q, R, D

The world model is three small networks on top of a frozen V-JEPA 2 latent:

```
camera clip (16 frames) --V-JEPA 2 (frozen)--> tokens --time mean, 2x2 pool, PCA 16/cell--> z (1024)

Q(z, p) -> cube xyz, held        the readout: where the cube is and whether it is in the fingers
D(z, p, a) -> z', p'             the dynamics: what the latent and the robot state look like after an action
R(z, p) -> proximity, P(collision), P(table hit)    the penalty head: is the arm about to hit something
```

`p` is the 20-number robot state (`environment/README`), `a` the 5-number action. The controller
(`controller/`) imagines action sequences with `D`, reads them with `Q` and `R`, and scores them.

```
common.py             run folder helpers: manifest, feature cache, checks, metrics, the state record format
collect_full.py       data 1: full task on the SAC's layout, cube jittered 3 cm, drops and recoveries, clutter twins
collect_obstacles.py  data 2: paired scenes with blocking corridor obstacles (for R)
prepare_episodes.py   data 3: a data_collection.collect dataset (full mix) -> run folder
encode.py             V-JEPA 2 -> z for every state, the PCA basis
merge_runs.py         several encoded runs -> one run to train on
train.py              readout (Q), reward (R), dynamics (D)
eval_reward.py        R on real latents and on latents imagined by D
run_combined.sh       merge + the three training stages + eval_reward
```

## A run folder

Every stage reads and writes one folder under `data/`:

```
data/<run>/
  manifest.json          every recorded state and rollout (below)
  scene_settings.json    the layout (and seed) of every rollout
  collection.json        what produced it: settings, config, layout, scene groups with their split
  observations/<rollout>/static_0000.jpg   the static camera frames (prepare_episodes links episodes/ instead)
  features/
    latents.npy          one z per state, manifest order, float32 (N, 1024)
    pca.npz              mean and components of the per-cell PCA
    meta.json            encoder settings: model, revision, pooling name, clip_frames, frame_stride, dtype, spatial_pool, keys
  attempts/<tag>/
    models/{readout,readout_p,reward,dynamics}.pt    the trained networks with their normalisation statistics
    reports/*_training.json, readout_validation.json, reward_validation.json, reward_<split>.json
```

`manifest.json` (schema `vision_consequences_v1`):

- `states[i]`: `key` (`<rollout>:<serial>`), `scene`, `split` (train / val / test), `frames` (the last up to
  64 frame paths ending at this state, the encoder takes the newest `clip_frames` of them), `p` (20),
  `object_xyz`, `object_quat`, `held`, `obstacle_contact`, `table_contact`, `reward`,
  `reward_components` (unweighted, from the environment), `action_from_previous` (the action that led
  here, `None` for the first state), `time`, plus per collector `view` (empty / clutter / obstacles),
  `case` and `phase`.
- `rollouts[j]`: `id`, `scene`, `split`, `view`, `case`, `states` (indices into `states`, one more than
  actions), `actions`, `result` (the TaskSession result: success, steps, contacts ...).
- `config`: the environment config the data was recorded with. `control_hz`, `camera`, `p_columns`,
  `action_columns`.

Scenes are the unit of the train / val / test split, so every rollout of a scene is in one split. The
split is fixed at collection time (`collection.json` `groups`).

## 1. Data

All three collectors need the frozen SAC (`rl/README.md`): they read the layout and config it was
trained with from `data/rl_baseline_v1/pipeline.json` and drive the normal cases with it.

**`collect_full.py`** (defaults 16 / 4 / 6 scenes, `data/full_test1`, 30,642 states). Per scene the cube start
is jittered by 3 cm (`--jitter`), so the cube cannot be guessed from the robot state and `Q` has to look
at the image. Four cases, each recorded in two views:

| case | who drives | what happens |
|---|---|---|
| normal | the SAC | the plain task |
| drop_early / drop_middle / drop_late | the scripted policy | at 15 / 50 / 75 % of the carry the gripper is forced open for 7 steps, the cube drops, the scripted policy restarts and picks it up again |

| view | layout |
|---|---|
| empty | the plain table |
| clutter | 1 to 3 boxes along the far edge of the table; the SAME executed actions are replayed, so the image changes but nothing physical does |

`RecoverySession` is the `TaskSession` (`rl/task_control.py`) with the forced release and a `phase()`
label; it also puts the really executed action into `info`, which is what the clutter twin replays. The
drop cases give `D` and `Q` states where the cube is on the table far from where it started, in the air
without being held, and being regrasped.

**`collect_obstacles.py`** (defaults 14 / 4 / 6 scenes, `data/obstacles_test1`, 10,103 states, 72 rollouts).
Three rollouts per scene: `normal_empty` (SAC on the empty table), `replay_obstacles` (the same actions
with 1 to 3 corridor obstacles from the environment's sampler, so the arm or the cube hits them and the
collision / proximity / table penalties differ from the empty twin only because of what the camera sees)
and `scripted_obstacles` (the scripted route planner going around, few penalties). This is the data `R`
learns from: identical robot trajectories, different penalties, the difference visible only in `z`. The
6 test scenes are the held-out obstacle benchmark `controller/control_clutter.py` runs.

**`prepare_episodes.py`** converts a `data_collection.collect` dataset into a run folder, one episode per
rollout, split 70 / 15 / 15 by episode index. With `configs/full_mix.yml` (success .4, scripted .2, drop .1,
collide .15, random .15, cube, 1 to 3 obstacles, 120 episodes, `data/episodes_test1`, 17,265 states) this
adds collisions, drops, random motion and above all fingers closed on nothing, which the other two
datasets have almost none of.

## 2. Encoder, `encode.py`

`Encoder` wraps `facebook/vjepa2-vitl-fpc64-256` (ViT-L, pinned revision) frozen in fp16 on the GPU.
For one state it takes the newest `clip_frames` frames (16, stride 1, the oldest repeated when the
history is shorter), does the HF preprocessing on the GPU (resize to 292, centre crop 256, normalise),
runs the encoder without its predictor and gets `8 x 16 x 16` tokens of 1024 (16 frames in tubelets of
2). The pooling makes `z`:

- `spatial_mean` (default): mean over the 8 time slices -> a 16x16 grid, 2x2 average pooled -> 8x8
  cells, each cell projected by a PCA to 16 values -> `z` of 8 * 8 * 16 = 1024 numbers that keep WHERE
  things are. The cube is about 10 px, less than one patch; the mean over all tokens (below) lost it.
- `mean_all`: the mean over all tokens -> 1024 numbers, the original M2 latent. Kept for comparison.
- `spatial_last`: the grid of the last time slice only; frame to frame jitter made it unlearnable.

The PCA basis is fitted label-free on `--pca-clips` (300) random training clips with `torch.pca_lowrank`,
saved to `features/pca.npz`, and reused online by the controller. `--pca other/pca.npz` copies another
run's basis so several runs share one latent space and can be merged. Encoding writes into a memmap and
records progress, so `--resume` continues an interrupted run. 0.16 s per state on an RTX 4060.

```bash
python -m world_model.encode --run data/full_test1
python -m world_model.encode --run data/obstacles_test1_fullpca --pca data/full_test1/features/pca.npz
```

`merge_runs.py` concatenates runs encoded with the same basis and clip settings into one run folder
(`data/combined_test1`, 58,010 states): keys and scenes prefixed with the run name, state indices
shifted, frame folders symlinked, latents concatenated.

## 3. Training, `train.py`

One stage per call, all with the same `--run` and `--tag`; the models land in `attempts/<tag>/models/`.
Shared machinery: `z`, `p` and cube `xyz` are normalised by mean and std of the training states (floored
at 1e-3), the statistics are stored in every checkpoint and `normalized()` applies them at inference.
`mlp(inputs, outputs)` is Linear-LayerNorm-GELU-Linear-GELU-Linear with width 128. `fit()` trains with
AdamW (lr 1e-3, batch 64, gradient clipping 5) and keeps the epoch with the lowest validation loss;
`reports/<role>_training.json` has the curve.

**readout** trains two networks with the same loss: `readout` = `Q(z, p)` (1044 -> 4) and `readout_p`
(20 -> 4), the same from `p` alone. Outputs are normalised cube xyz (MSE) and a held logit (BCE).
`readout_p` is the blind control (`--blind` in the controller): on the fixed-scene data it used to beat the
visual `Q` because the cube position was predictable from the arm; on the 3 cm jittered data the visual
`Q` is about twice as accurate (1.5 / 2.9 cm vs 2.6 / 4.7 cm in x / y). Validation metrics go to
`reports/readout_validation.json`. On `combined_test1`: xyz MAE 1.5 / 2.9 / 0.45 cm, held F1 0.96.

**reward** trains `R(z, p)` (1044 -> 4, the fourth output unused): proximity (MSE against the recorded
unweighted proximity component, in [-1, 0]) and collision and table-hit logits (BCE against the recorded
contacts). Only the obstacle data has nonzero targets.

**dynamics** trains `D` on every window of `--rollout-steps` (8) consecutive actions inside a rollout,
rolled out recursively from the window's first state, loss = mean over the window of the normalised MSE
of `z` and of `p`. Two architectures:

- `SplitDynamics`: a robot head `(p, a) -> p + delta` (width 128) and a visual head
  `(z, p, a, p') -> z + delta` (width 512). The robot inputs to the visual head are detached so the
  latent loss cannot bend the robot prediction.
- `SplitDynamicsZ` (`--robot-sees-z`, used for `combined`): the robot head also reads `z` (detached), so
  the finger width it predicts can depend on whether a cube is between the fingers, which `(p, a)` alone
  cannot know.

`--task-weight w` (1.0 for `combined`) adds a task-consistency term: the FROZEN `Q` of the same tag reads
every imagined `(z', p')` and must still give the recorded cube xyz and held label. Without it plain
latent MSE let `D` drop the cube from the latent, because the cube is one of 64 cells while the arm
dominates the loss, and `D` never predicted a lift. `readout` therefore has to be trained before
`dynamics` with the same tag and normalisation.

```bash
python -m world_model.train readout  --run data/combined_test1 --tag combined --epochs 30
python -m world_model.train reward   --run data/combined_test1 --tag combined --epochs 30
python -m world_model.train dynamics --run data/combined_test1 --tag combined --epochs 30 --rollout-steps 8 --task-weight 1.0 --robot-sees-z
```

A checkpoint (`torch.load(..., weights_only=True)`) is a dict: `model` (state dict), `role`,
`z_mean z_std p_mean p_std xyz_mean xyz_std`, `z_dim`, `input_dim`, `architecture`, `rollout_steps`,
`task_weight`, `epoch`, `val_loss`, `encoder`. `load_model(run, role, tag)` rebuilds the network from it;
`readout()` and `imagine()` are the two inference helpers the controller and `eval_reward` use.

## 4. Evaluation, `eval_reward.py`

Can `R` see obstacles, and does it still see them on latents `D` imagined rather than real ones? On
the held-out split it scores `R` on the real latents of the empty and the obstacle scenes, then on
latents imagined over 1 / 4 / 8 steps from 20 windows per obstacle rollout with the recorded actions
(what the planner does). Reports proximity MAE, collision and table-hit precision, recall and AUC to
`attempts/<tag>/reports/reward_<split>.json`. On `combined_test1`: collision AUC 0.995 on real latents,
0.99 imagined at 8 steps.

## Checkpoints: where they are loaded from, and how to get them after a clone

Everything the controllers need from this folder's output is small, and lives in one run folder:

| file | size | who reads it |
|---|---|---|
| `data/combined_test1/attempts/combined/models/readout.pt` | 0.6 MB | Q: `WorldModels.read` via `train.load_model(run, "readout", tag)` |
| `.../models/readout_p.pt` | 0.1 MB | the blind control (`--blind`) |
| `.../models/reward.pt` | 0.6 MB | R: `WorldModels.penalty` (optional, skipped by `--no-penalties`) |
| `.../models/dynamics.pt` | 6 MB | D: `WorldModels.predict` |
| `.../attempts/combined/info.json` | 20 KB | the environment config and resting cube height (`control_pipeline.run_settings`) |
| `.../attempts/combined/reports/*.json` | small | training curves and validation numbers, for reading only |
| `data/combined_test1/features/pca.npz` | 70 KB | the PCA basis the online `Encoder` applies |
| `data/combined_test1/features/meta.json` | 3 MB | encoder settings (model, revision, pooling, clip length, dtype) |
| `data/rl_baseline_v1/models/sac_best.zip` + `pipeline.json` | 1 MB | the frozen SAC and the layout it was trained on (`rl/README.md`) |
| `data/obstacles_test1/{collection.json,scene_settings.json}` | 80 KB | the held-out obstacle layouts for `control_clutter` |

`--models-run` is the run folder and `--tag` the attempt, so Q, R and D are
`<models-run>/attempts/<tag>/models/{readout,reward,dynamics}.pt`. `load_model` rebuilds each network from
the checkpoint dict (`train.py`, `build_model`) and returns the normalisation statistics with it. The
manifest (375 MB) and the latent cache (230 MB) are only needed to train; `info.json`, written by
`train.py` next to the models, carries the two things the controller used to take from the manifest,
and the controller falls back to the manifest only when `info.json` is missing.

These files are in the repository through Git LFS (`.gitignore` lists them as exceptions under
`simulation/data/`, `.gitattributes` routes the binaries through LFS). After cloning, or in an existing
clone that only has LFS pointer files (a `.pt` that is a few hundred bytes of text starting with
`version https://git-lfs.github.com/spec/v1`):

```bash
git lfs install                 # once per machine
git lfs pull --include="simulation/data/combined_test1/**,simulation/data/rl_baseline_v1/**" --exclude=""
```

That is about 70 MB (`rl_baseline_v1` includes its demo episode frames). Then the default commands in
`controller/README.md` run without any training. The V-JEPA weights are not in the repository; they
download from Hugging Face on the first run (`facebook/vjepa2-vitl-fpc64-256`, about 1.2 GB, into
`~/.cache/huggingface`).

To share a newly trained model: train with a new `--tag` (or a new run folder), add the same kind of
exception lines to `.gitignore` for it, `git add` the folder, commit and push; LFS uploads the binaries
with the push. Frames and `latents.npy` should stay out: they are large and are rebuilt by the
collectors and `encode.py`.

## The LeWM comparison, `lewm/` (reference only)

`lewm/` is the M2 branch's planned comparison against the LeWorldModel paper: a small encoder and
transformer predictor trained from scratch on the task frames, planning towards a goal image with no
readout, no reward model and no policy. It was prepared but never run, and it is written against the M2
modules, so it is kept unchanged as a reference rather than wired into this package. `lewm/README.md`
says why it was there, how it differs from the model in use, and how to run it from the commit it
belongs to.

## Reading the code

Start with `common.py` (the formats), then one collector (`collect_obstacles.py` is the shortest),
`encode.py` top to bottom (`Encoder.grid` and `encode` are the latent), then `train.py`: the two
dynamics classes, `fit`, and the three stage blocks in `main`. `run_combined.sh` is the order everything
runs in. `FLOW.md` in this folder goes through every file function by function, and
`diagrams/training_pipeline.png` is the same flow as a picture.
