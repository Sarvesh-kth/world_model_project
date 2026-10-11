# World model: data, encoder, Q, R, D

The world model is three small networks on top of a frozen V-JEPA 2 latent of both cameras plus the table
height map:

```
static clip (16 frames) --V-JEPA 2 (frozen)--> tokens --time mean, 16x16 grid, PCA 32/cell--> 8192  \
wrist clip  (16 frames) --V-JEPA 2 (frozen)--> tokens --time mean, 2x2 pool to 8x8, PCA 16/cell--> 1024   }  z (9280)
static depth image --point cloud--> 8x8 table height map, metres / 0.25                        -->   64  /

Q(z, gripper) -> cube xyz, held  the readout: where the cube is (a spatial softmax over the static grid)
                                 and whether it is in the fingers; it sees no arm pose, only finger width and command
D(z, p, a) -> z', p'             the dynamics: what the latent and the robot state look like after an action
R(z, p) -> reach, grasp, lift, transport, P(placed),          the reward model: the task's own shaped terms and
           proximity, P(collision), P(table hit), P(empty grasp)   the penalties, read from the latent
```

`p` is the 20-number robot state (`environment/README`), `a` the 5-number action. The controller
(`controller/`) imagines action sequences with `D`, reads them with `Q` and `R`, and scores them.
V-JEPA takes RGB video only, so the point cloud does not go through it: the static depth image becomes a
top-down height map of the table (`PickPlaceEnv.height_map`, the highest point in each of 8x8 cells,
clipped at 25 cm) that is appended to `z`. Obstacles, the cube and the arm all stand out in it, `D`
predicts it like the rest of `z`, and the obstacle-aware SAC (`rl/README.md`) reads the same map.

```
common.py             run folder helpers: manifest, feature cache, checks, metrics, the state record format
prepare_episodes.py   a data_collection.collect dataset (the full mix) -> run folder
prepare_control.py    the planner's own closed-loop episodes -> run folder (no encoding, the online latents)
encode.py             V-JEPA 2 -> z for every state, the PCA bases
merge_runs.py         several runs encoded with the same bases -> one run to train on
train.py              readout (Q), reward (R), dynamics (D)
eval_reward.py        R on real latents and on latents imagined by D
eval_grasp.py         does the world model tell a grasp from a miss in imagination
run_combined.sh       the three training stages + the two checks on one run
```
The whole chain, collection to closed loop, is `simulation/cluster_round.sh` (`simulation/CLUSTER.md` step by step).

## A run folder

Every stage reads and writes one folder under `data/`:

```
data/<run>/
  manifest.json          every recorded state and rollout (below)
  scene_settings.json    the layout (and seed) of every rollout
  collection.json        what produced it: settings, config, layout, scene groups with their split
  observations/<rollout>/{static,wrist}_0000.jpg   both camera frames (prepare_episodes links episodes/ instead)
  features/
    latents.npy          one z per state, manifest order, float32 (N, 9280)
    pca.npz              mean and components of the per-cell PCA, one pair per camera (static_mean, static_components, wrist_...)
    meta.json            encoder settings: model, revision, pooling name, cameras, clip_frames per camera, frame_stride,
                         dtype, spatial_pool, heightmap (on/off, cells, scale), layout (block sizes of z), keys
  attempts/<tag>/
    models/{readout,readout_p,reward,dynamics}.pt    the trained networks with their normalisation statistics
    reports/*_training.json, readout_validation.json, reward_validation.json, reward_<split>.json
```

`manifest.json` (schema `vision_consequences_v1`):

- `states[i]`: `key` (`<rollout>:<serial>`), `scene`, `split` (train / val / test), `frames` and
  `wrist_frames` (the last up to 64 frame paths of each camera ending at this state, the encoder takes the
  newest `clip_frames` of them), `heightmap` (64 numbers, metres), `p` (20),
  `object_xyz`, `object_quat`, `held`, `obstacle_contact`, `table_contact`, `reward`,
  `reward_components` (unweighted, from the environment), `action_from_previous` (the action that led
  here, `None` for the first state), `time`, plus per collector `view` (empty / clutter / obstacles),
  `case` and `phase`.
- `rollouts[j]`: `id`, `scene`, `split`, `view`, `case`, `states` (indices into `states`, one more than
  actions), `actions`, `result` (the TaskSession result: success, steps, contacts ...).
- `config`: the environment config the data was recorded with (the camera placement comes from here at
  run time). `control_hz`, `cameras`, `p_columns`, `action_columns`.

Scenes are the unit of the train / val / test split, so every rollout of a scene is in one split. The
split is fixed at collection time (`collection.json` `groups`).

## 1. Data

`data_collection.collect` with `configs/full_mix.yml` records the episodes (`simulation/README.md`): the cube
and the goal anywhere on the reachable table, 0 to 3 obstacles, a random cube yaw, and one policy kind per
episode (success, scripted, miss, lateral, wander, drop, collide, random) so the data has clean picks,
closes beside and above the cube, low sideways approaches, the hand roaming far from the cube, drops,
collisions and random motion, with the reward components of every step. **`prepare_episodes.py`** turns
that folder into a run folder, one episode per rollout, split by episode index modulo 7 (train / val /
test). Collecting more episodes into the same folder and running it again keeps every earlier episode in
its split and appends the new states, which `encode --resume` then encodes.

```bash
python -m data_collection.collect --config configs/full_mix.yml --episodes 400 --out data/episodes_v3/episodes_raw --workers 4 --seed 1000
python -m world_model.prepare_episodes --episodes data/episodes_v3/episodes_raw --run data/episodes_v3
```

**`prepare_control.py`** turns the planner's own episodes (`data/control_*`, which save the frames, the
latents computed online and the true cube state) into a run folder without encoding, to fold the planner's
own situations back into the training data (`merge_runs.py` joins it with the main run).

## 2. Encoder, `encode.py`

`Encoder` wraps `facebook/vjepa2-vitl-fpc64-256` (ViT-L, pinned revision) frozen in fp16 on the GPU.
For one state and one camera it takes the newest `clip_frames` frames (16 for both cameras; stride 1,
the oldest repeated when the history is shorter), does the HF
preprocessing on the GPU (resize to 292, centre crop 256, normalise), runs the encoder without its
predictor and gets `frames/2 x 16 x 16` tokens of 1024. The pooling makes one block per camera:

- `spatial_mean` (default): mean over the 8 time slices -> a 16x16 grid, pooled `--static-pool` /
  `--wrist-pool` patches per cell (1 for the static view, so all 16x16 cells stay; 2 for the wrist view,
  8x8 cells), each cell projected by a PCA to `--static-pca` / `--wrist-pca` values (32 / 16) ->
  16 * 16 * 32 = 8192 static and 8 * 8 * 16 = 1024 wrist numbers that keep WHERE things are. The cube is
  about 15 px from the static camera, one patch: on the old 8x8 / 16-per-cell grid no readout could place it
  better than 3-4 cm whenever the hand was away from it (val, 76k states), on the full grid with 32 per
  cell the spatial-softmax readout gets 0.7 cm from 3,000 frames. The mean over all tokens (below) loses
  the position entirely. `meta.json` records `spatial_pool`, `cells` and `pca_dims` per camera.
- `mean_all`: the mean over all tokens -> 1024 numbers, the original M2 latent. Kept for comparison.
- `spatial_last`: the grid of the last time slice only; frame to frame jitter made it unlearnable.

The two camera blocks are followed by the height map block (64, the recorded map divided by
`heightmap.max_height`, `--no-heightmap` drops it); `meta.json` `layout` records the block sizes. One PCA
basis per camera is fitted label-free on `--pca-clips` (300) random training clips with
`torch.pca_lowrank`, saved to `features/pca.npz`, and reused online by the controller. `--pca other/pca.npz`
copies another run's bases so several runs share one latent space and can be merged. Encoding writes
into a memmap and records progress, so `--resume` continues an interrupted run. 0.26 s per state on an
RTX 4060 (both cameras), the same online in the controller.

```bash
python -m world_model.encode --run data/full_v2
python -m world_model.encode --run data/obstacles_v2 --pca data/full_v2/features/pca.npz
```

`merge_runs.py` concatenates runs encoded with the same bases and clip settings into one run folder:
keys and scenes prefixed with the run name, state indices shifted, frame folders symlinked, latents
concatenated. The v3 chain trains on one run and does not need it.

### The whole chain on a GPU cluster, `cluster_round.sh`

Everything after the environment install runs there: collection (MuJoCo renders offscreen, EGL or
`MUJOCO_GL=osmesa`), `prepare_episodes`, encoding (the slow step, 0.27 s per state on the RTX 4060, about
0.09 s on the cluster's GPU; it needs nothing but the run folder), training, the offline checks and the
closed-loop evaluation. `bash cluster_round.sh v3 400` from `simulation/` with the venv active does all of
it for 400 full-mix episodes into `data/episodes_v3`, models under `data/episodes_v3/attempts/v3`, and
`results/control_v3_{empty,obstacles,clutter}`. Every step is restartable: the collector appends, `encode
--resume` continues from `features/progress.json`, the training stages are skipped when their model exists.
What comes back: `attempts/v3/` (models, reports, info.json) and `features/{pca.npz,meta.json}` make the
controllers run anywhere (`Checkpoints` below); `latents.npy` and the frames stay on the cluster.

Older runs (`data/full_v2`, `obstacles_v2`, `episodes_v2`, `planner_v1`) were encoded with the 8x8 latent and
are not compatible; `full_v2` and `obstacles_v2` also have the cube at one spot (the SAC layout), and all
of `episodes_v2` has it in the four end zones, never in the middle of the table. They are kept for `rl_q`
and as a reference only.

## 3. Training, `train.py`

One stage per call, all with the same `--run` and `--tag`; the models land in `attempts/<tag>/models/`.
Shared machinery: `z`, `p` and cube `xyz` are normalised by mean and std of the training states (floored
at 1e-3; the height map cells at 0.1, so a cell nothing ever stood on in training does not turn a test-time
obstacle into a value in the hundreds), the statistics are stored in every checkpoint and `normalized()`
applies them at inference.
`mlp(inputs, outputs)` is Linear-LayerNorm-GELU-Dropout-Linear-GELU-Dropout-Linear with width 128. `fit()`
trains with AdamW (lr 1e-3 with a cosine decay to 1 %, weight decay 0.05, batch 256, gradient clipping 5)
and keeps the epoch with the lowest validation loss; `reports/<role>_training.json` has the curve. Before
the decay, the weight decay and the dropout every stage had its best validation epoch at 2-3 and a training
loss 7-10x below the validation loss: consecutive states of a scene are near duplicates, so the networks
memorised the ~1,000 scenes instead of learning the task.

**readout** trains `Q` and `readout_p`, the blind control (`--blind` in the controller) that reads the whole
robot state and no image. `Q` is a `GridReadout`: each camera block put back into its grid (static 16x16
of 32 PCA values, wrist 8x8 of 16; the height map resized onto the static grid as one more channel), a 1x1
and two 3x3 convolutions per camera, a spatial softmax on the static features whose expected grid
coordinates say where each of 8 heads looks, and an MLP head on the pooled features, the key points and
the two gripper numbers of `p` (`P_GRIPPER`: finger width, command). `Q` gets no arm pose: the old MLP on
`(z, p)` had learned "the cube is near the hand" and was 3-4 cm off whenever the hand was away from the
cube, exactly when the planner chooses its direction. Seven outputs: the cube ABSOLUTE (used while it is
not held), the cube RELATIVE to the grasp site (used while held: the clip is a time mean and a carried cube
is smeared in it, relative to the hand it is a constant) and the held logit; `readout_loss` trains each
position head on its own states, `readout()` picks by the held probability. Gaussian noise on the
normalised latent while training (`--z-noise`, 0.3). `--mlp-readout` restores the old MLP on `(z, p)`
for comparison. Validation metrics go to `reports/readout_validation.json`.

**reward** trains the reward model `R(z, p)` (the same grid network, with the whole `p`; 9 outputs,
`REWARD_OUTPUTS`): the environment's own
unweighted shaped terms reach, grasp, lift, transport (in [0, 1], MSE), placed (the recorded reward stage
is `done`, BCE), proximity (in [-1, 0], MSE) and the collision, table-hit and empty-grasp flags (BCE; the
empty grasp of runs recorded before it existed is derived from `p`, gripper closed and under 3 cm). The
planner builds its task potential from the first five (`planner.learned_potential`) and its penalties
from the last four; `reward_terms()` returns them by name with the flags as probabilities. Validation MAE per
term and precision / recall per flag go to `reports/reward_validation.json`.

**dynamics** trains `D` on every window of `--rollout-steps` (8) consecutive actions inside a rollout,
rolled out recursively from the window's first state, loss = mean over the window of the normalised MSE
of `z` and of `p`. Two architectures:

- `SplitDynamics`: a robot head `(p, a) -> p + delta` (width 128) and a visual head
  `(z, p, a, p') -> z + delta` (width 512). The robot inputs to the visual head are detached so the
  latent loss cannot bend the robot prediction.
- `SplitDynamicsZ` (`--robot-sees-z`, used for `combined`): the robot head also reads `z` (detached), so
  the finger width it predicts can depend on whether a cube is between the fingers, which `(p, a)` alone
  cannot know.

`--task-weight w` (1.0) adds a task-consistency term: the FROZEN `Q` of the same tag reads every imagined
`(z', p')` and must still give the recorded cube xyz and held label (`readout_loss`). Without it plain
latent MSE let `D` drop the cube from the latent, because the cube is a few cells of the grid while the arm
dominates the loss, and `D` never predicted a lift. `readout` therefore has to be trained before `dynamics`
with the same tag and normalisation. What `D` has to beat: keeping `z` constant over the 8 steps costs a
normalised MSE of 0.27 on the old latent; the old `D` reached 0.16 (validation).

```bash
python -m world_model.train readout  --run data/episodes_v3 --tag v3 --epochs 60
python -m world_model.train reward   --run data/episodes_v3 --tag v3 --epochs 60
python -m world_model.train dynamics --run data/episodes_v3 --tag v3 --epochs 60 --rollout-steps 8 --task-weight 1.0 --robot-sees-z
```

A checkpoint (`torch.load(..., weights_only=True)`) is a dict: `model` (state dict), `role`,
`z_mean z_std p_mean p_std xyz_mean xyz_std rel_mean rel_std`, `z_dim`, `p_columns` (the columns of `p` in
the input), `outputs`, `architecture` (+ `blocks`, `cells`, `p_dim` for the grid network, `input_dim` for
the MLP), `rollout_steps`, `task_weight`, `epoch`, `val_loss`, `encoder`. `load_model(run, role, tag)` rebuilds the network from it;
`readout()` and `imagine()` are the two inference helpers the controller and `eval_reward` use.

## 4. Evaluation, `eval_reward.py` and `eval_grasp.py`

Can `R` see obstacles and the task, and does it still see them on latents `D` imagined rather than real
ones? On
the held-out split it scores `R` on the real latents of the empty and the obstacle scenes, then on
latents imagined over 1 / 4 / 8 steps from 20 windows per obstacle rollout with the recorded actions
(what the planner does). Reports proximity MAE, the MAE of the four task terms, collision and table-hit
precision, recall and AUC (and the same for empty grasp) to `attempts/<tag>/reports/reward_<split>.json`.

**`eval_grasp.py`** asks the question the planner depends on: does the world model know a grasp from a
miss? From real hover states of a split it imagines the planner's go-to-and-close rule aimed at the cube
and at points 2 / 4 / 6 cm beside it and 3 / 5 cm above it, then reads held with `Q`, `D`'s finger width,
and `R`'s grasp term and empty-grasp probability on the imagined end state
(`attempts/<tag>/reports/grasp_<split>.json`). Before the miss data the models called every close a hold
(held 0.83-0.95 at every offset, width stuck at 5 cm); `run_combined.sh` runs it after `eval_reward`.

## Checkpoints: where they are loaded from, and how to get them after a clone

Everything the controllers need from this folder's output is small, and lives in one run folder:

| file | size | who reads it |
|---|---|---|
| `data/combined_v2/attempts/combined/models/readout.pt` | 1 MB | Q: `WorldModels.read` via `train.load_model(run, "readout", tag)` |
| `.../models/readout_p.pt` | 0.1 MB | the blind control (`--blind`) |
| `.../models/reward.pt` | 1 MB | R: `WorldModels.penalty` (optional, skipped by `--no-penalties`) |
| `.../models/dynamics.pt` | 10 MB | D: `WorldModels.predict` |
| `.../attempts/combined/info.json` | 20 KB | the environment config and resting cube height (`control_pipeline.run_settings`) |
| `.../attempts/combined/reports/*.json` | small | training curves and validation numbers, for reading only |
| `data/combined_v2/features/pca.npz` | 140 KB | the PCA bases the online `Encoder` applies |
| `data/combined_v2/features/meta.json` | 3 MB | encoder settings (model, revision, pooling, cameras, clip lengths, height map, dtype) |
| `data/rl_baseline_v1/models/sac_best.zip` + `pipeline.json` | 1 MB | the frozen SAC and the layout it was trained on (`rl/README.md`), for rl_true / rl_q only |
| `data/obstacles_v2/{collection.json,scene_settings.json}` | 80 KB | the held-out obstacle layouts for `control_clutter` |

`--models-run` is the run folder and `--tag` the attempt, so Q, R and D are
`<models-run>/attempts/<tag>/models/{readout,reward,dynamics}.pt`. `load_model` rebuilds each network from
the checkpoint dict (`train.py`, `build_model`) and returns the normalisation statistics with it. The
manifest and the latent cache are only needed to train; `info.json`, written by
`train.py` next to the models, carries the two things the controller used to take from the manifest,
and the controller falls back to the manifest only when `info.json` is missing.

These files are in the repository through Git LFS (`.gitignore` lists them as exceptions under
`simulation/data/`, `.gitattributes` routes the binaries through LFS). After cloning, or in an existing
clone that only has LFS pointer files (a `.pt` that is a few hundred bytes of text starting with
`version https://git-lfs.github.com/spec/v1`):

```bash
git lfs install                 # once per machine
git lfs pull --include="simulation/data/combined_v2/**,simulation/data/rl_baseline_v1/**" --exclude=""
```

That is about 70 MB (`rl_baseline_v1` includes its demo episode frames). Then the default commands in
`controller/README.md` run without any training. The V-JEPA weights are not in the repository; they
download from Hugging Face on the first run (`facebook/vjepa2-vitl-fpc64-256`, about 1.2 GB, into
`~/.cache/huggingface`).

Trained models are not tracked: copy `attempts/<tag>/` and `features/{pca.npz,meta.json}` between
machines. Frames and `latents.npy` stay where they were made; the collector and `encode.py` rebuild them.

## Reading the code

Start with `common.py` (the formats), then `prepare_episodes.py`, `encode.py` top to bottom
(`Encoder.grid` and `encode` are the latent), then `train.py`: `GridReadout`, the two dynamics classes,
`fit`, and the three stage blocks in `main`. `cluster_round.sh` is the order everything runs in.
