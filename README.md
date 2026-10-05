# JEPA World Model Pick and Place

This repository contains a MuJoCo Franka Panda simulation, scripted demonstrations,
data collection, frozen V-JEPA 2 features, learned dynamics/readouts, a trained SAC
baseline and experimental visual control. The `success` demo uses a scripted
policy that reads simulator state. The live comparison uses a fixed reward
formula; a separate learned immediate reward network remains planned work.

**Sharing with teammates:** code/report exports do not include notebook-trained
weights or recordings. See [Share trained runs and videos with Calle](#share-trained-runs-and-videos-with-calle)
for Git LFS upload/download commands and video instructions.

## Install (macOS and Linux)

Clone the repository if you do not already have it, then create a Python virtual
environment and install the dependencies:

```bash
git clone https://github.com/Sarvesh-kth/world_model_project.git
cd world_model_project
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If you already have the repository, start with `cd world_model_project`. The
virtual environment and most generated runs are ignored by Git. Explicitly shared
run folders are allowed and configured for Git LFS; see the sharing section.

## Watch the robot

Run the commands from the `simulation/` directory on a computer with a graphical
desktop and working OpenGL support:

```bash
cd simulation
```

**macOS:** MuJoCo's passive viewer requires the `mjpython` launcher installed in
the virtual environment:

```bash
../.venv/bin/mjpython play.py success --seed 34
```

**Linux:** Use the virtual environment's regular Python interpreter:

```bash
../.venv/bin/python play.py success --seed 34
```

The seed selects a repeatable layout. Replace `success` with `fail`, `collide`,
or `random` to see other scripted/data-collection behaviors. Add `--speed 2` for
faster playback, `--episodes 5` for five consecutive seeds, or
`--layout data/smoke/episode_000000/meta.json` to replay a saved layout.

On macOS the viewer shows the robot and the terminal prints reward events and
the final reward summary. The live Matplotlib window is disabled there because
`mjpython` runs the script off macOS's GUI thread. On Linux the live reward plot
opens alongside the viewer. Do not set `MUJOCO_GL=egl` for the visible viewer;
EGL is for offscreen rendering on supported Linux setups.

## Collect and inspect episodes

From `simulation/`, collect ten episodes and plot the reward of the first one:

```bash
../.venv/bin/python -m data_collection.collect --episodes 10 --out data/smoke --workers 1
../.venv/bin/python plot_rewards.py data/smoke/episode_000000
```

Collection runs without a viewer and saves `index.csv` plus a folder per
episode. Each episode contains step-by-step actions, simulator state, rewards,
the sampled layout, and static/wrist camera JPEGs. Add `--pointcloud` to the
collection command if depth-derived point clouds are needed. See
[simulation/README.md](simulation/README.md) for the file layout and additional
simulation details.

## Grade E notebook run (Linux with an NVIDIA GPU)

This is the first fixed-scene **data and one-step dynamics** run. The cloned
branch has code but no generated episodes, cached features, or trained weights.
Run in a notebook **terminal**, from a fresh checkout of `M2_Kuba`:

```bash
git clone --branch M2_Kuba --single-branch https://github.com/Sarvesh-kth/world_model_project.git
cd world_model_project
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt -r requirements-model.txt
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")'
source .venv/bin/activate
bash setup_mujoco_headless.sh
deactivate
cd simulation
```

The CUDA check must print `True` before encoding. The notebook driver reports
CUDA 12.4 support, so the command above uses PyTorch's matching CUDA 12.4 wheel.
Unqualified `pip install torch` currently selects a CUDA 13 wheel and prints
`False` with a driver-too-old warning on this notebook. To repair an existing
virtual environment, run the same pinned PyTorch install command again; pip
will replace the incompatible torch/torchvision versions. Then repeat the CUDA
check. See [PyTorch's published wheel
commands](https://pytorch.org/get-started/previous-versions/).
The H100 MIG allocation shown for this project has about 20 GiB, so start with
the small run below. V-JEPA's weights are downloaded from Hugging Face on the
first encode; that command needs internet access and enough cache space.
The notebook image has neither usable EGL nor OSMesa for MuJoCo. The setup
script downloads OSMesa with APT into `.mujoco-osmesa/`, without a system-wide
install, and registers a loader in this project's active `.venv`. Run it once
in this environment before collecting images. A Linux machine with working
EGL can instead skip this setup and use `MUJOCO_GL=egl` for collection.

```bash
# 1. Collect 10 episodes on one fixed cube/goal layout. No viewer is opened.
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ../.venv/bin/python -m data_collection.collect \
  --config configs/grade_e.yml --layout configs/grade_e_layout.json \
  --episodes 10 --workers 1 --seed 34 --out data/grade_e/episodes

# 2. Form aligned one-step pairs across each episode; start with four per episode.
../.venv/bin/python -m world_model.prepare \
  --episodes data/grade_e/episodes --out data/grade_e/manifest.json \
  --max-per-episode 4

# 3. Freeze V-JEPA and cache one 1024-value vector per distinct 64-frame clip.
../.venv/bin/python -m world_model.encode \
  --manifest data/grade_e/manifest.json --out data/grade_e/features

# 4. Check whether these visual vectors contain object-position information.
../.venv/bin/python -m world_model.probe \
  --manifest data/grade_e/manifest.json --features data/grade_e/features

# 5. Train a small action-conditioned MLP and print held-out episode errors.
../.venv/bin/python -m world_model.train_dynamics \
  --manifest data/grade_e/manifest.json --features data/grade_e/features \
  --out artifacts/grade_e/dynamics
```

If MuJoCo still reports an OpenGL loading error, check that the setup script
completed in the same virtual environment used by `../.venv/bin/python`.
Collection renders
both static and wrist JPEGs; the current encoder reads only the static view.
The saved CSV also contains privileged simulator object coordinates and rewards,
but `D` receives only the frozen visual vector, 20 proprioception values, and
five bounded action values. `prepare` pairs observation at serial `t` with the
**action stored at serial `t+1`** and observation at `t+1`, because each CSV row
is recorded *after* its action. It splits whole episodes into train/validation.

Outputs are `data/grade_e/episodes/` (JPEG, CSV, metadata),
`data/grade_e/manifest.json` (aligned pairs and split),
`data/grade_e/features/` (vectors and encoder metadata), and
`artifacts/grade_e/dynamics/best.pt` (MLP weights and normalization). All are
ignored by Git. The terminal's validation `z` and `p` errors are normalized
mean squared errors; compare each against the printed persistence baseline.
The probe reports held-out object-position error versus a constant baseline.
The training log also evaluates shifted validation actions as a basic action
sensitivity diagnostic; it does not by itself prove useful control.
The ten-episode run checks the pipeline and is too small to establish a useful
world model or robot success. No learned controller is invoked by these commands.
For clips near episode start, the first observed frame is repeated to make a
64-frame input. The cap selects transitions across each episode, not just its
first steps.

For a larger fixed-scene dataset, collect more episodes into the **same**
`data/grade_e/episodes` directory, then write a **new** manifest path and feature
directory so the small-run artifacts remain reproducible. For example:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ../.venv/bin/python -m data_collection.collect \
  --config configs/grade_e.yml --layout configs/grade_e_layout.json \
  --episodes 100 --workers 1 --seed 35 --out data/grade_e/episodes
../.venv/bin/python -m world_model.prepare \
  --episodes data/grade_e/episodes --out data/grade_e/manifest_full.json
../.venv/bin/python -m world_model.encode \
  --manifest data/grade_e/manifest_full.json --out data/grade_e/features_full
../.venv/bin/python -m world_model.train_dynamics \
  --manifest data/grade_e/manifest_full.json --features data/grade_e/features_full \
  --out artifacts/grade_e/dynamics_full
```

This implementation uses the base [V-JEPA 2 ViT-L 64-frame checkpoint](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256),
with its encoder frozen and mean pooling over output tokens. The pre-trained
predictor is skipped; `D` is the action-conditioned predictor trained here.
Mean pooling and one-step loss are provisional baselines. Before using `D` for
planning, check object-location information, multi-step drift, and whether
changing an action changes its predicted outcome on held-out episodes.

### Diagnose action prediction without removing V-JEPA

After collecting and encoding the paired-action dataset
(`data/grade_e/manifest_branches.json` and `data/grade_e/features_branches`),
train a split dynamics model from `simulation/`:

```bash
../.venv/bin/python -m world_model.train_dynamics \
  --manifest data/grade_e/manifest_branches.json \
  --features data/grade_e/features_branches \
  --architecture split --epochs 30 \
  --out artifacts/grade_e/dynamics_split

../.venv/bin/python -m world_model.rescore_contrasts \
  --source artifacts/grade_e/contrast_before \
  --checkpoint artifacts/grade_e/dynamics_split/best.pt \
  --out artifacts/grade_e/contrast_split_best

../.venv/bin/python -m world_model.compare_contrasts \
  --before artifacts/grade_e/contrast_before \
  --after artifacts/grade_e/contrast_split_best
```

The first head learns `next_p` from current proprioception and action. The
second head learns `next_z` from the current frozen V-JEPA vector, current
proprioception, action, and predicted `next_p`. Both heads are part of one
action-conditioned world model. The training log prints the real and predicted
mean `+x`/`-x` gripper effect on paired **training** states. `best.pt` minimizes
ordinary held-out episode `z+p` error; `last.pt` also saves the final epoch to
show whether checkpoint selection hides a learned action effect.
`rescore_contrasts` uses the previously saved real branch states and visual
vectors, so it does not replay MuJoCo or rerun V-JEPA. If the old contrast data
is missing, run `world_model.action_contrast` over the eight validation states
first. Compare the physical x-effect error and branch x MAE with the old model
and the action-mean baseline. A better x result alone does not demonstrate
object or collision prediction; those need varied scenes and visual ablations.

## Object-consequence experiment: four separate tests

For an automatic follow-up with more varied examples, see
[Automatic varied experiment](#automatic-varied-experiment) below. The original
commands in this section keep their controlled profile and 24-step sequence.

This is the next diagnostic experiment after the split-head gripper-motion test.
Use the notebook GPU for V-JEPA encoding and training. It collects **new data**;
the old `data/grade_e` dataset/checkpoints remain usable by the old commands.
Everything for a run goes under one ignored directory, `simulation/data/vision_v1/`.
No new dependencies are required beyond the existing simulation, PyTorch and
`requirements-model.txt` environment.

### 1. Pull on the notebook

From the repository root:

```bash
git switch M2_Kuba
git pull --ff-only origin M2_Kuba
./.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable in this virtualenv"; print(torch.__version__, torch.cuda.get_device_name(0))'
cd simulation
```

Keep the working CUDA environment you already installed. If this is a fresh
clone, first follow the installation and rootless OSMesa setup above; the setup
script is `setup_mujoco_headless.sh` at the repository root. On the notebook,
OSMesa renders images on the CPU while V-JEPA/training use the GPU.

### 2. Collect and inspect real data

Run each command from `simulation/`:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ../.venv/bin/python -m world_model.vision.collect \
  --run data/vision_v1 --train-scenes 12 --val-scenes 4 --test-scenes 4 --seed 34

../.venv/bin/python -m world_model.vision.audit --run data/vision_v1
```

For a first installation check, use a different run name and counts `2`, `1`,
`1`. Use the 12/4/4 run for the first learning experiment. Larger follow-up runs
can use 40/10/10 and a new run name/seed. Four held-out scene groups are a small
diagnostic sample; they do not establish general manipulation performance.

For **each scene group**, we collect two cube placements and two action sequences:

| Placement | A: `close_lift` | B: `open_lift` |
|---|---|---|
| Cube under gripper | close for 8 steps, then lift for 16 | remain open for 8 steps, then lift for 16 |
| Cube displaced 10 cm in x | exactly the same A sequence | exactly the same B sequence |

A/B start from the **same restored integration and controller state**. The
displaced scene replays the same approach actions; the collector checks that
starting robot proprioception matches. The approach uses the existing scripted
controller. Branches are explicit Cartesian actions executed by MuJoCo/IK,
without a learned planner. Closing action is `[0,0,0,0,-1]`; closed lifting is
`[0,0,1,0,-1]`. Open versions end in `+1`. One step is 0.1 seconds with the default
configuration. Normalized `dz=1` is a controller command; actual displacement is
measured from physics, rather than assumed equal to the target delta.

Training anchors have x in `[0.14,0.22]` m, validation anchors `[0.225,0.245]`,
test anchors `[0.105,0.125]`; y varies in `[-0.26,-0.21]`. These are disjoint
**anchor regions**, with entire scene groups and their branches assigned to one
split. Displaced cube positions may overlap a different region; they are not
claimed to be a globally disjoint set of object x coordinates. Cube geometry,
color, mass and obstacle-free setup stay fixed for this controlled experiment.
It measures grasp/lift consequences, not collision avoidance or task completion.

Each saved state has a camera-history list, 20 robot values, real object xyz and
quaternion, endpoint bilateral finger contact (`held`), step contact flags,
the preceding action, time and reward diagnostics. `held` means both fingers
contact the object while the gripper is commanded closed; it is not a separate
success label. Object rise is checked independently. Every real future branch
frame is saved. A future clip uses the shared real approach history followed by
**that branch's actual frames**, truncated to the last 64; early history is padded
by repeating its first frame.

The default run has 20 groups, 80 real rollouts and 1,960 states/clip endpoints.
The audit checks images, times/action alignment, source restoration, split
leakage, positive/negative contact labels and real A/B height differences.
It writes `reports/audit.json` and one contact sheet per scene. **Inspect the
`reports/scene_*.jpg` images before training.** If the audit fails, stop and
inspect its reported problems; incomplete collection is recorded in the manifest.

### 3. Encode the real clips with frozen V-JEPA

```bash
../.venv/bin/python -m world_model.vision.encode --run data/vision_v1
```

This runs the same base `facebook/vjepa2-vitl-fpc64-256` encoder on each real
64-frame clip, skips its predictor, and mean-pools tokens into a 1,024-value
vector. It saves `features/latents.npy` and `features/meta.json`, including model
revision, ordered state keys and file hashes. It prints progress, seconds/clip
and peak allocated GPU memory; encoding is one clip at a time for the 20 GiB MIG.
No actions or simulator object coordinates are fed to V-JEPA. No encoder weights
are trained. An interrupted encoding can resume safely:

```bash
../.venv/bin/python -m world_model.vision.encode --run data/vision_v1 --resume
```

### 4. Train and validate Q on real observations first

```bash
../.venv/bin/python -m world_model.vision.train readout --run data/vision_v1 --epochs 60
```

`Q(real z, measured p) -> object xyz + grasp probability`. Targets come from the
saved simulator labels. It minimizes normalized xyz MSE plus grasp binary cross
entropy. A companion `Q_p(p)` uses only measured proprioception as a diagnostic.
All normalization and fitting use training scenes; best weights use validation
loss. The test scenes are not used to train or select checkpoints.

Read `reports/readout_validation.json`. The default **Q gate** requires xyz MAE
on each axis <= 2 cm, height MAE <= 1 cm, grasp F1 >= 0.80, Brier score <= 0.15,
and both label classes. These are proposed practical tolerances, not established
exam thresholds. If the gate fails, inspect Q and the data before trusting an
object interpretation of D's latent predictions. The four tests still run for
diagnosis and mark Q-derived results accordingly. A gate pass on real clips is
necessary; it does not guarantee accuracy on D's imagined vectors.

### 5. Train D, then the baseline without vision

```bash
../.venv/bin/python -m world_model.vision.train dynamics \
  --run data/vision_v1 --epochs 60 --rollout-steps 4

../.venv/bin/python -m world_model.vision.train baseline \
  --run data/vision_v1 --epochs 60
```

D reuses `SplitDynamics`: `D_p(p,a) -> next_p`, then
`D_z(z,p,a,predicted_next_p) -> next_z`. It trains on consecutive real branch
states with normalized z/p MSE averaged over **four recursively predicted
steps**. Gradients from the visual head do not override the robot head. For
evaluation, D starts from real `z0,p0`, then repeatedly uses its own predicted
states and the proposed actions; no real intermediate future state is supplied.
Q stays frozen and translates imagined `(z,p)` into object outcomes.

The no-vision baseline is a small direct outcome MLP:
`(starting p, padded action prefix, prefix mask) -> future object xyz/grasp`.
It gets the same training scene/outcome labels and training epoch budget. It
receives neither z, true starting object coordinates, nor measured future p.
This is a direct supervised baseline, **not an architecture-matched latent
ablation**. Comparison alone cannot assign every difference to the encoder;
the matched-position test additionally checks whether identical p/actions need
the visual information to explain different object outcomes.

All models and training curves are saved under `models/` and `reports/`.
Existing model files are preserved. To retrain on the same cached data, pass
`--tag retry_1` to all three training stages and all four tests. This saves a
separate attempt under `attempts/retry_1/` without rerunning simulation/V-JEPA.
Use validation results to adjust settings; keep the test split for final checks.

### 6. Run each test independently

These commands default to **test** scene groups and a 24-step open-loop horizon.
They read cached real vectors/labels and saved model weights; they do not replay
MuJoCo or run V-JEPA again. Each produces its own JSON report and per-rollout CSV.

```bash
# Test 1: does D beat keeping the visual state unchanged?
../.venv/bin/python -m world_model.vision.test persistence --run data/vision_v1
```

```bash
# Test 2: do close/open predictions match their own real outcomes?
../.venv/bin/python -m world_model.vision.test action --run data/vision_v1
```

```bash
# Test 3: does D + Q beat the predictor without vision on object outcomes?
../.venv/bin/python -m world_model.vision.test no-vision --run data/vision_v1
```

```bash
# Test 4: same starting p/actions, cube under versus displaced, unseen anchors.
../.venv/bin/python -m world_model.vision.test positions --run data/vision_v1
```

### Reading results and debugging

| Report field | How to read it |
|---|---|
| `Q_object_interpretation_gate_passed` | Checks Q on real validation and real evaluation clips. If false, object errors cannot diagnose D alone. Latent errors still apply. |
| Persistence `by_horizon` | Compare `D_z_mse` with `persistence_z_mse` at 1/8/16/24 steps. Smaller is better; watch long-horizon drift. p has its own errors. |
| `Q_constant_z_with_same_predicted_p` | Keep the initial visual vector but use the same imagined future robot state. Compare its object errors with `D_Q_outcomes` to check whether changing z helps Q, rather than only predicting robot motion. |
| Action `matched_beats_swapped_fraction` | Among real A/B height differences >= 2 cm, matching should beat swapping. Around 0.5 is weak ranking evidence; aim toward 1. Read individual pairs as well. |
| Action `height_effect_mae_cm` | Error in predicted close-minus-open object-height difference; smaller is better. Small z error alone is insufficient. |
| No-vision `D_Q` versus `no_vision_sequence_baseline` | Compare **the same** xyz/height MAE, grasp F1 and Brier score. MAE/Brier lower, F1 higher. `Q_on_real_future` shows readout error before D is involved. |
| `Q_p_on_measured_future_p` | Diagnostic only: it has the actual future robot values, which imagined predictions do not. It checks whether future proprio alone reveals the object outcome. |
| Positions `D_Q_effect_mae_cm` versus `no_vision_effect_mae_cm` | Can the visual pathway explain under/displaced height differences from matching robot inputs? Source p errors and identical-action checks are included per pair. |
| `eligible_pairs` | If zero, the real outcomes did not differ enough; the contrast is inconclusive. Open/open-position pairs are normally ineligible. |
| `height_mae_when_held_cm` | Check actual lifted/contact examples separately, so many stationary examples do not conceal failure to predict a lift. |

`xyz_mae_cm` is an array for x/y/z. Object heights in CSV are absolute world
heights: a cube on a 75 cm table has center height around 77 cm; **subtract its
source height** to read the rise. All normalized MSEs in a report use that D's
train-only statistics; compare physical-unit object errors across differently
normalized runs. Contact F1/accuracy uses a 0.5 probability threshold. Brier is
the mean squared probability error, which checks calibration as well as class.

For exploratory validation instead of the final test, add `--split val`.
To diagnose an earlier point in a collected sequence, use `--horizon 8` or `16`;
at step 8 there may be grasp contact but little height contrast. A horizon longer
than the collected sequence is rejected. Reports include dataset/checkpoint
hashes, scene counts and Q controls; keep these with the checkpoints.
Feature/model hash mismatches fail instead of silently mixing runs. Collection
rejects nonempty run directories. CUDA encoding is resumable; training writes
best weights during the run and epoch curves at completion.

The runnable implementation regression check is:

```bash
../.venv/bin/python -m world_model.vision.check
```

It uses temporary **toy features on CPU**, exercises alignment/leakage/cache
guards, Q/D/baseline training and each test command, and deletes its temporary
data afterwards. It is a software check, not evidence that V-JEPA predicts
objects accurately. The learned reward model and CEM controller are later work;
none of these commands makes learned action choices for the robot.

## Automatic varied experiment

This follow-up changes the **grasp conditions and action consequences**, rather
than only moving an otherwise identical successful demonstration around the table.
It keeps the cube and obstacle-free table fixed while varying:

| Variable | Varied profile |
|---|---|
| Gripper approach location | x `[0.12,0.24]`, y `[-0.28,-0.16]` metres |
| Cube relative to that location | near offsets 8–18 mm, edge offsets 25–45 mm, far offsets 80–120 mm, in different directions |
| Starting robot pose | small open-gripper yaw changes and vertical preparation offsets |
| Lift command | normalized dz between 0.5 and 1.0, with reference cases at 1.0 |
| Action sequences | close/lift, open/lift, close/lift sideways, close/lift/release |

The source is above contact, then every branch descends for six open-gripper
steps, holds its grip command for eight, and performs its lift for sixteen:
**30 steps total**. Sideways/release variants change the last eight steps.
All four branches restore the same source; both placements use the same actions
and must have matching initial robot values. The actual physics determines
success, failed grasp, contact and object rise. An "under" placement can fail;
a "near offset" can still grasp. The audit checks that both label classes and
genuine outcome contrasts exist. Failed source-matching attempts are resampled,
with reasons saved in the manifest, instead of silently weakening the comparison.

Whole scene groups remain separate across train/validation/test. In this profile
all splits sample the same wider position distribution with independent scenes;
this measures generalization to new scenes from that distribution. The original
controlled profile uses separate anchor regions. These are different experiments;
compare physical errors and report the differing datasets/horizons explicitly.
Obstacles and collision-aware scoring are a subsequent campaign.

### One command on the notebook

From the repository root:

```bash
git switch M2_Kuba
git pull --ff-only origin M2_Kuba
cd simulation
../.venv/bin/python -m world_model.vision.pipeline --run data/vision_v2
```

The defaults are **60 training, 12 validation, 12 test groups**, seed 42:
672 real sequences and 20,328 states/clip endpoints. It runs these stages in order:

1. Check the existing CUDA/Python dependencies.
2. Collect new data, audit it and save contact sheets.
3. Encode every real clip once with frozen V-JEPA, still one clip at a time.
4. Train Q, D and the no-vision baseline for 60 epochs, with eight-step D windows.
5. Run each of the four tests on validation and then test scenes at horizon 30.
6. Repeat model fitting/evaluation with training seeds **0, 1 and 2**, reusing
   the same simulator dataset and feature cache. No model is selected by test results.
7. Export compact reports and logs to `results/vision_v2/` at the repo root.

Models live under `data/vision_v2/attempts/seed_0/`, `seed_1/` and `seed_2/`.
Every stage prints progress and writes its own text log to `data/vision_v2/logs/`.
The encoder reports peak tensor allocation, allocator reservation and GPU capacity.
The larger dataset increases processing time; it does not put all clips into VRAM.
Based on the earlier 0.32 s/clip, encoding alone would take roughly 1.8 hours for
this run; actual speed and simulation/training time depend on the notebook.
Keep the terminal/kernel and teaching GPU session alive for the full run.

To inspect the plan without executing or writing files:

```bash
../.venv/bin/python -m world_model.vision.pipeline --run data/vision_v2 --plan
```

For a smaller pilot, use a **different run name**. A pilot is an installation/data
check, not a strong learning result:

```bash
../.venv/bin/python -m world_model.vision.pipeline \
  --run data/vision_v2_pilot --train-scenes 8 --val-scenes 4 --test-scenes 4 \
  --epochs 10 --training-seeds 0
```

### Resume and failures

```bash
../.venv/bin/python -m world_model.vision.pipeline --run data/vision_v2 --resume
```

Repeat any custom flags from the original command. The runner checks the saved
configuration and artifact hashes, skips completed stages, resumes collection
after its last complete scene and resumes an interrupted encoder cache. An
interrupted training stage preserves its old files under `logs/interrupted/`
and restarts that stage from scratch; optimizer state is not resumed.
Executable errors stop the pipeline with the stage log and traceback. **Q
accuracy gate failures continue into evaluation and remain clearly diagnostic.**
Completion means all computations ran; it does not mean the models passed.
Raw data and earlier runs are preserved. For new settings, use a new run name.

### Read and share the results

Start with `results/vision_v2/summary.txt` and `summary.csv`. The folder also
contains full JSON/CSV reports, training curves with best validation epoch and
checkpoint hashes, stage logs, scene settings and run provenance. At horizon 30,
read the same persistence/action/no-vision/position metrics described above.
The action report additionally compares close/lift with sideways and release
sequences when their real xyz outcomes differ by at least 2 cm; those additional
comparisons use xyz separation, while the original close/open test uses height.
Inspect the contact sheets under `data/vision_v2/reports/scene_*.jpg` for realism.

The exporter copies only reports/logs/settings. Camera datasets, encoded arrays,
model weights and agent notes remain outside the shareable folder. Each training
seed has separate reports; variation across those seeds is a diagnostic of
training stability, not three independent datasets. Do not tune settings on the
final test results; reserve fresh test scenes for later model changes.

To share the evidence through Git, from the repository root after the run finishes:

```bash
git add results/vision_v2
git commit -m "Record varied vision experiment results"
git push origin M2_Kuba
```

Share the commit hash so its reports can be reviewed. Avoid force-adding the
ignored `simulation/data/` directory. All individual stage commands still work;
for this profile use `--horizon 30` and the corresponding `--tag seed_0` when
rerunning an evaluation by hand. The existing `world_model.vision.check` also
exercises runner sequencing, export, resume and artifact guards using toy CPU
features; it makes no real V-JEPA quality claim.

## Live A-to-B control: SAC baseline and JEPA planning

This is an **opt-in experiment**, separate from the original viewer and saved
vision experiments. The existing `environment/rewards.py` defines a reward; it
does **not** contain a trained RL policy. This runner trains an actual policy
using [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/sac.html)
in our existing Panda/MuJoCo scene. It then measures whether the cube really
gets picked up, transported to B and released. Successful computation alone
does not establish a working controller.

### Install and run on the GPU notebook

From the repository root:

```bash
git pull --ff-only origin M2_Kuba
.venv/bin/python -m pip install -r requirements-control.txt
cd simulation
../.venv/bin/python -u -m world_model.vision.control_pipeline \
  --models-run data/vision_v2 --tag width_visual_seed_0 \
  --out data/control_v1
```

The optional dependency is pinned to SB3 **2.7.1**, compatible with the existing
Torch 2.6 CUDA installation. This command requires the original notebook's
`vision_v2` manifest, feature metadata and frozen model checkpoints. It does not
download a policy for a different robot or overwrite the earlier experiments.
Use the existing OSMesa setup; the runner preserves the rendering backend.

### What the single command does

1. **Collect 20 successful full-task scripted demonstrations.** Save the real
   observations, actions, rewards and outcomes. The scripted teacher has exact
   simulator state. These are explicitly labelled demonstrations.
2. **Initialize a small actor by behaviour cloning (120 epochs).** Train it to
   reproduce those demonstration actions. Report its initial validation result
   separately; behaviour cloning alone is not an RL training result.
3. **Fine-tune with SAC for 10,000 real control steps.** Real demonstrations
   initialize replay and 1,000 critic warmup updates. A strong, declared actor
   demonstration loss (`--bc-weight 100`) remains during RL to reduce forgetting.
   Use the opt-in goal/progress objective below, including real contact penalties.
   The original simulator reward is logged separately. This is demonstration-
   assisted RL; a successful fit does not demonstrate an advantage over BC alone.
   Evaluate every 2,500 steps on
   five validation seeds; select the best *RL* checkpoint by actual placement
   success, then final distance. No test episode selects the checkpoint.
4. **Run four controllers from home on the same five fresh test scenes:**

   | Controller | What selects actions / what it observes |
   |---|---|
   | `scripted` | Existing waypoint reference, exact simulator state. |
   | `rl_true` | Trained SAC actor, measured robot state plus exact cube/contact state. This is the privileged RL baseline. |
   | `rl_q` | Same SAC actor, measured robot state plus cube position/held probability estimated by frozen JEPA → Q. |
   | `jepa_mpc` | SAC proposes a sequence; CEM refines candidates; frozen D imagines future z/p; Q reads future cube state; predicted rewards and a terminal SAC critic estimate rank sequences. Execute only the first action, observe and repeat. |

5. **Save and export the evidence.** Success means actually grasped, lifted at
   least 4 cm while held, then released within the configuration's target radius
   and 2 cm of resting height for its settling interval. The default comparison
   jitters cube xy by at most 1 cm per axis around the fixed scene; B stays fixed.
   Five test episodes are a small local check, not general manipulation proof.

### What enters the models and reward

The SAC actor receives the same 41-value observation layout in every mode:
20 robot values, cube xyz, cube-to-gripper offset, explicit world-coordinate B,
B-to-cube offset, held value, six task-history values, known cube resting height
and remaining episode fraction. History records ever-grasped, previous-held,
ever-lifted, settling fraction,
completed placement and current potential. For `rl_true`, cube/contact/history
are exact.
Fixed unit scaling makes centimetre offsets significant to the actor; the D/Q
checkpoint normalization and physical-unit inputs are preserved.
For `rl_q` and `jepa_mpc`, cube/contact/history are estimated from camera-derived
Q outputs. Actual object/contact values are logged for evaluation after actions;
they do not select actions in these two modes.

Online JEPA receives the same static-camera JPEG preprocessing, pinned encoder
revision and causal 64-frame history as the saved model cache. Early clips repeat
the first frame. JEPA remains frozen; D predicts the consequences of proposed
actions. Q translates predicted z/p into cube xyz and held probability. There is
no image decoder and no true future simulator rollout inside CEM.

The original reward pays grasp/lift/transport repeatedly. A local SAC run with
that objective placed **0/5 validation episodes after 100,000 steps**, despite
its BC initialization placing 4/5. It cannot be described as an already working
RL solution. Lingering incentives and forgetting are hypotheses behind the
opt-in correction; neither is established as the sole cause.

The new control reward is:

```text
0.99 * Phi(next) - Phi(current)
+ 15 once for strict completed placement
+ drop/action/failure/contact penalties
- 0.01 per step
```

`Phi` reuses the original bounded reach/grasp/lift/transport terms and weights.
It is zero on terminal states. The explicit episode deadline is a terminal task
failure, so the critic does not bootstrap a continuation past it.
Holding still earns no repeated positive bonus; finishing requires the lift/carry/release/settle criterion above. The same
formula scores real SAC transitions and D/Q imagined transitions. This is
[potential-based shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf),
plus an explicit task objective; it does not guarantee learning success. Original
reward totals and new control totals are reported separately and must not be
compared as identical objectives. Every candidate has independent history;
its placement bonus occurs once and scoring stops at its predicted terminal.
Held probability is thresholded at 0.5; it is an estimate, not perfect contact.
Q has no collision/proximity/table-contact head: those three penalties are
explicitly **omitted from imagined scoring** and are always measured in actual
evaluation. This first runner rejects layouts with obstacles. It does not claim
collision-aware planning or exact reconstruction of full simulator state.

Default CEM uses 64 candidates, 8 steps (0.8 simulated seconds), 3 iterations and
8 elites. Its score is the discounted sum of supported imagined rewards plus
the discounted minimum of SAC's two critic estimates at the final imagined
state. The **SAC critic** is an RL value network; the project's **Q readout** is
the cube-position/grasp network. Set `--terminal-weight 0` in a **new run** for
reward-only scoring. Implausible predicted finger widths invalidate candidates;
if all candidates are invalid, execute the Q-observed actor proposal and count
that fallback explicitly.

**Important scope:** current D/Q were fitted on local grasp/lift branches. This
whole-task run tests their extrapolation into reaching, carrying and placement;
it does not quietly retrain them or assume that they already predict those
regions accurately. A bad baseline or failed learned controller is a result to
inspect, not a successful project milestone.

### Observed local baseline check

[Saved CPU evidence](results/control_reference_check.json): the selected assisted
SAC checkpoint (2,500 of 10,000 training steps, chosen by validation) placed
**5/5 fresh test seeds**. Final cube distance from B was **0.05–1.28 cm**, with
no recorded obstacle/table contact steps. BC alone also placed5/5, so this is
not evidence of an RL improvement. Later RL checkpoints degraded to2/5
validation placements; selecting the final checkpoint blindly would be wrong.
The fixed-scene result does not establish broad robustness or JEPA performance.
Real JEPA/Q/MPC still requires the notebook run above and its saved checkpoints.

### Read, visualize, resume and share

Start with `results/control_v1/summary.txt` and `summary.csv`. Read actual full
placement counts before reward totals. The predeclared local SAC baseline check
is at least 80% actual success over at least five test scenes. The report prints
whether that was observed; smaller/custom checks do not meet this criterion.
`rl_training.json` shows demonstration-only validation, RL validation history and
the selected training step. Each episode has `steps.csv` with actions, true
reward components, actual goal distance/lift/contact and model errors. MPC also
exports candidate scores and selected imagined trajectories. One-step forecast
errors compare the chosen first action with the actual next observation; later
imagined states are **not** compared with a real trajectory that replanned
different actions.

Raw files stay under `simulation/data/control_v1/episodes/<method>/seed_<seed>/`:

- `actual.gif` and `frames/`: actual camera observations, not imagined video.
- `trajectory.npz`: measured p, actual cube/contact labels, executed actions and
  actual rewards; JEPA methods also save real encoded z at every state.
- `trajectory_meta.json`: alignment (`state[t]`, `action[t]`, `state[t+1]`) and
  encoder identity. These recordings can supply future training data with new
  held-out evaluation scenes; this run does not automatically train on tests.
- `forecasts/step_*.npz`: candidate actions, predicted p/cube/held/rewards, scores,
  validity, current z and the selected predicted visual trajectory.

In a Jupyter cell, display an actual run (adjust the root if needed):

```python
from pathlib import Path
from IPython.display import Image, display
root = Path('/home/jovyan/Robots&EmbodiedAI/project/world_model_project/simulation/data/control_v1/episodes')
for method in ('scripted', 'rl_true', 'rl_q', 'jepa_mpc'):
    print(method)
    display(Image(filename=str(sorted((root / method).glob('*/actual.gif'))[0])))
```

For an interrupted run, repeat the original command and add `--resume`.
Completed stages are hash-checked and skipped. SAC resumes its last complete
validation checkpoint, optimizer and replay; its physical episode/RNG stream
restarts, so this is not bit-identical continuation. Interrupted episode files
are preserved under `interrupted/` and that episode is rerun. For changed
settings/code, use a fresh `--out`, such as `data/control_v2`.

The small `../.venv/bin/python -m world_model.vision.control_pipeline --self-check`
checks reward parity, candidate-history isolation, no holding bonus and a real scripted placement
without installing SB3 or loading GPU models. It does not demonstrate SAC/JEPA
control quality.

After completion, from the repository root:

```bash
git add -- results/control_v1
git --no-pager diff --cached --stat
git commit -m "Record live SAC and JEPA control comparison"
git push origin M2_Kuba
git rev-parse --short HEAD
```

Only compact reports/logs are exported. Keep frames, videos, model weights,
replay buffers and agent notes out of Git. Existing Mac/Linux viewer commands
and dependencies remain unchanged.

## Diagnose Q with real versus predicted inputs

After `vision_v2` finishes, use its existing validation cache and checkpoints to
find where imagined object/grasp predictions fail. This command runs D from
each real starting state with the saved actions, then substitutes inputs **only
at Q** at steps 1, 8, 16 and 30:

| Printed combination | Visual input to Q | Robot input to Q |
|---|---|---|
| `pred_z_pred_p` | D's prediction | D's prediction |
| `real_z_pred_p` | Actual encoded future clip | D's prediction |
| `pred_z_real_p` | D's prediction | Actual future robot measurements |
| `real_z_real_p` | Actual encoded future clip | Actual future robot measurements |

From `simulation/`, run the same diagnostic for all three training seeds:

```bash
for seed in 0 1 2; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../.venv/bin/python -m world_model.vision.test hybrid \
    --run data/vision_v2 --tag "seed_${seed}" --split val --horizon 30 || break
done
```

It does not collect, encode or train again, and uses the small trained models
on CPU. It requires the original `data/vision_v2` folder on the notebook, not
just the Git-exported results. The hybrid command defaults to validation;
the original four test commands retain their default test split.

Read **held-only height MAE** (lower is better), **held detected** (recognized
real held objects / actual held objects), false held predictions and grasp F1
(higher is better). A row with no real held objects prints `n/a` for held-only
error and F1; it cannot test detection of positive grasps. Compare substitutions
within the same horizon, not accuracy across changing class counts.

If replacing predicted z by real z helps, the visual forecast is implicated.
If replacing predicted p by real p helps, robot information at Q is implicated.
If real z plus real p is poor, Q itself needs work. Replacing p only at Q does
not repair any p errors already used inside D_z's rollout. Real future inputs
are offline diagnostics and are unavailable to a planner; Q gate failures still
mark object interpretations as diagnostic.

Each seed writes `hybrid_val_h30.json` (metrics, contracts and artifact hashes)
and `hybrid_val_h30.csv` (each rollout/horizon/combination, real/predicted height,
grasp probability, gripper width and command) under
`data/vision_v2/attempts/seed_N/reports/`. It leaves existing weights and the
four earlier reports intact. To share the new evidence from `simulation/`:

```bash
for seed in 0 1 2; do
  cp data/vision_v2/attempts/seed_${seed}/reports/hybrid_val_h30.* \
    ../results/vision_v2/attempts/seed_${seed}/reports/
done
```

Commit those six new report files when ready. They are additional diagnostics;
the original campaign's `files.json` does not list them until a new export.

## Diagnose robot context inside D_z

The hybrid test changes inputs only at Q. The next diagnostic also supplies
**recorded current and next robot values to D_z at every step**, while starting
from one real visual vector and predicting all future visual vectors recursively.
It reuses the same cached clips and trained models; it does not collect, encode
or train again. From `simulation/`:

```bash
for seed in 0 1 2; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../.venv/bin/python -m world_model.vision.test robot-forced \
    --run data/vision_v2 --tag "seed_${seed}" --split val --horizon 30 || break
done
```

The table reports steps 1, 8, 16 and 30. Its four rows per step are:

| Combination | Visual rollout | Robot input to Q |
|---|---|---|
| `pred_z_pred_p` | Ordinary D, predicted robot context | Predicted endpoint p |
| `pred_z_real_p` | Ordinary D, predicted robot context | Recorded endpoint p |
| `forced_z_real_p` | D_z with recorded p at every step; z stays predicted | Recorded endpoint p |
| `real_z_real_p` | Actual encoded future clip, readout reference | Recorded endpoint p |

Compare **`pred_z_real_p` versus `forced_z_real_p`**: Q gets identical robot
values, so any change comes from using accurate robot context inside the visual
rollout. Lower latent `z MSE` and held-only height MAE, plus more real held
objects detected without extra false detections, would support robot-context
errors as a contributor. Little improvement would mean visual forecasting still
fails even with correct robot context. Neither outcome alone identifies the
training/representation cause or proves online control works. The real-z
reference has zero latent error by construction; it is not a learned forecast.
Early horizons without held objects cannot test positive grasp detection.

Recorded **future** robot values are available only for this offline diagnostic;
a planner would have to predict them. No intermediate real visual vectors are
fed back to D_z. Normalization uses the original dynamics checkpoint's training
statistics, and existing Q interpretation gates still apply. JSON/CSV reports
include artifact hashes, per-rollout probabilities, height errors and actual,
predicted and Q-input gripper values. They are saved separately as
`robot-forced_val_h30.json` and `.csv` in each seed's `reports/` directory.

To share just these new reports from `simulation/`:

```bash
for seed in 0 1 2; do
  cp data/vision_v2/attempts/seed_${seed}/reports/robot-forced_val_h30.* \
    ../results/vision_v2/attempts/seed_${seed}/reports/
done
```

## Diagnose contact states and individual robot values

After the robot-forced diagnostic, reuse the same cache/checkpoints to measure
which robot coordinates drift and whether visual predictions fail immediately
or after several imagined steps. From `simulation/`:

```bash
for seed in 0 1 2; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../.venv/bin/python -m world_model.vision.contact \
    --run data/vision_v2 --tag "seed_${seed}" --horizon 30 \
    --reset-horizons 1 4 8 || break
done
```

This command uses **validation only**. It does not collect, encode or train again:

1. Run the original 30-step D rollout and compare all 20 predicted robot values
   with recorded measurements. Report joint/yaw errors in degrees, velocities
   in degrees/s, position/finger width in cm, and gripper command in its native
   -1/+1 units. The command-sign mismatch distinguishes open from close errors.
2. Restart D from **actual z and p at every eligible recorded state**, including
   states around grasping and while holding. Use the corresponding recorded
   action suffix; predict 1, 4 and 8 steps from the same anchors. Starts in the
   last seven steps are excluded so all three horizons use identical anchors.
3. Compare ordinary recursion, ordinary predicted z with real endpoint p,
   robot-forced visual recursion, unchanged starting z with real endpoint p,
   and Q on actual future z/p. All imagined visual paths remain recursive after
   their initial reset; actual future p is used only by offline diagnostic controls.

`transition` means an adjacent held-label change or a recorded transient grasp;
it is a grasp-transition proxy, **not a complete finger-contact measurement**.
`held`/`unheld` are the other starting-state strata. Labels select report groups
and never enter D. Q metrics use actual **future** held labels, so their positive
counts can change with horizon. Many anchors overlap within the same 12 scene
groups: they are correlated measurements, not thousands of independent trials.

### How to read the contact diagnostic

- **Robot errors:** compare `from_start_same_targets` with `reset_one_step` on
  identical target states. Large errors in both implicate immediate prediction;
  much smaller reset errors implicate accumulated drift. Read individual
  coordinates rather than comparing cm, degrees and command errors numerically.
  JSON also includes per-coordinate normalized errors and all original targets.
  Robot CSV `label_step` identifies the actual preceding state used to group
  that error; `start_step` identifies where the forecast itself began.
- **Local visual errors:** check whether ordinary `z MSE` beats
  `unchanged_z_real_p`, especially around transition/held starts. Poor one-step
  forecasts from actual starts indicate a local dynamics issue; degradation
  from 1 to 4/8 steps indicates a recursive limitation. Q on real future states
  reveals readout limitations separately.
- **Robot influence inside D_z:** compare `pred_z_real_p` with
  `forced_z_real_p` at each horizon. Q's robot input is identical in both.
  Corrections can supply future contact information through finger width, so
  improvement does not prove that generic motor drift is the only cause.
- **Empty/failed groups:** `n/a` means no held-positive targets for that row.
  Failed Q gates keep object predictions diagnostic. The JSON retains unheld
  controls even though the terminal omits their separate visual table rows.

Each seed saves `contact_val_h30_reset1-4-8.json` with grouped metrics, contracts,
source/code/checkpoint hashes and anchor counts; `_robot.csv` with all measured
and predicted robot values in native units; and `_forecasts.csv` with each
anchor/horizon/combination, latent error, height and grasp probability. Existing
reports and weights remain unchanged. Progress prints every 200 anchors; the
small D/Q models run on CPU and no V-JEPA download or GPU pass is needed.

To share only these reports from `simulation/`:

```bash
for seed in 0 1 2; do
  cp data/vision_v2/attempts/seed_${seed}/reports/contact_val_h30_reset1-4-8* \
    ../results/vision_v2/attempts/seed_${seed}/reports/
done
```

## Try a vision-conditioned finger-width correction

Use the existing vision_v2 cache and seeds to test one bounded intervention.
The original robot/visual heads and Q stay frozen. A small residual MLP predicts
a correction to **next finger width**, starting from zero correction so its
initial outputs exactly match the saved D. It receives current z, current p and
the proposed action. No actual future robot measurements or object labels are
model inputs; actual next width is the training target.

Two variants use identical head size, seed, 60-epoch budget, eight-step recursive
training windows, original normalization and validation splits:

- `visual`: current JEPA z + p + action → width correction.
- `robot`: zero z + p + action → width correction, a control for extra capacity/training.

The original D_z and Q still use JEPA in **both** variants. This comparison tests
vision specifically inside the width predictor; it is not a no-vision ablation
of the entire system. Only the new head trains, using normalized width MSE; the
best checkpoint is selected by validation width loss. Existing weights/functions
are frozen, but changed width can alter later robot and visual predictions through
their recurrent inputs. Q is copied unchanged so readout changes cannot explain
differences. The final test split is not used.

From `simulation/`, run the complete comparison in one command:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../.venv/bin/python -m world_model.vision.width_compare \
  --run data/vision_v2 --execute --epochs 60
```

It requires your completed original validation/contact reports for seed_0/1/2.
For each seed it trains `width_robot_seed_N` and `width_visual_seed_N`, runs the
contact/reset diagnostic plus persistence/action/positions on validation, and
prints the baseline/control/visual summary. It reuses the frozen cache; no new
collection, encoder download or encoding occurs. Stage logs are under
`data/vision_v2/logs/width_*.txt`. Existing completed width fits/stages are reused
with settings/checkpoint/companion checks; an interrupted training directory is
preserved and rejected rather than overwritten. Move that incomplete directory
aside before retrying. Use the same epoch budget when continuing a comparison.

To run one training stage yourself:

```bash
../.venv/bin/python -m world_model.vision.train width \
  --run data/vision_v2 --from-tag seed_0 --tag width_visual_seed_0 \
  --width-input visual --seed 0 --epochs 60 --rollout-steps 8
```

Run this only before that destination tag exists. All existing diagnostic commands
accept the new tag; old checkpoints/default training remain supported. A runtime
check verifies that zero correction preserves baseline z/p predictions before
training. Original-head weights are excluded from the optimizer.

### Reading the width comparison

`data/vision_v2/reports/width_comparison.json` and `.csv` record source hashes and:

- **Width error:** held-state rollout MAE and transition-state reset-one-step MAE.
- **Useful object forecasts:** normal h8 held-height error, grasp F1/detections and
  false positives; h30 held-height/F1 and action/position height-effect errors.
- **Q gate:** a copied failed gate still marks object interpretations diagnostic.

Compare `visual` against `robot` within each seed. Consistent visual gains would
support JEPA's contribution to contact-sensitive width prediction. If both improve
similarly, the benefit may come from the new head/objective rather than vision.
If width improves but object forecasts do not, fixed D_z/Q remain limiting.
Do not compare the new width training loss numerically with the old combined
z+p training loss; compare the same physical validation metrics. There is no
preclaimed improvement or automatic promotion to a planner.

After this comparison, the next project stage is a validated goal-progress scorer
and a short-horizon Grade E CEM attempt: current cameras → JEPA → imagined D
sequences → score → execute one IK action → observe again. The current campaign
covers grasp/lift/release; it does not establish transport/placement or collision
accuracy. Score ranking and those action-coverage gaps must be checked when
connecting the fixed-scene pick-and-place loop.

To share the compact comparison and validation reports from `simulation/`:

```bash
mkdir -p ../results/vision_v2/width_comparison
cp data/vision_v2/reports/width_comparison.* ../results/vision_v2/width_comparison/
for seed in 0 1 2; do
  for variant in robot visual; do
    tag="width_${variant}_seed_${seed}"
    mkdir -p "../results/vision_v2/width_comparison/$tag"
    cp data/vision_v2/attempts/$tag/reports/*.json "../results/vision_v2/width_comparison/$tag/"
  done
done
```

Only reports are copied: cached clips, model weights and latent arrays remain
outside Git. The original campaign's files.json describes its original export,
not this new width comparison.

## Stable-lift score and candidate-choice experiment: one command

This is the next check before connecting a planner: **can our frozen models
choose a useful action sequence from alternatives?** It tests an explicit task
scorer, not a trained reward network. It does not yet run CEM or complete
pick-and-place. Use the GPU notebook with your completed `data/vision_v2` run,
including the original and paired finger-width checkpoints for seeds 0/1/2.
Those recordings, features and weights are local files; pulling Git alone
does not supply them. No extra packages are needed beyond that working environment.

Pull the code from your notebook's repository root:

```bash
git pull --ff-only origin M2_Kuba
cd simulation
```

Start the entire experiment:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ../.venv/bin/python \
  -m world_model.vision.decision_pipeline \
  --models-run data/vision_v2 --out data/decision_v1 --resume
```

Rerun this **same command** after an interruption. Collection resumes after its
last complete scene; encoding resumes its partial cache; completed stages are
verified and skipped. Changed settings, source checkpoints or relevant code
require a new `--out` name. A software failure stops with its log path; failed
model interpretation/calibration gates are saved scientific results.

### What runs automatically

1. **Collect fresh real outcomes.** Collection seed `20261004`; six fresh
   validation and twelve test scene groups, plus four unused train groups for
   the existing collector/audit contract. No model trains on this new data.
   Each group varies robot preparation, cube placement and motion parameters.
   Both cube-under-gripper and offset placements run six alternatives:
   close/lift, open/lift, close/hold, open/hold, close/sideways/lift,
   and close/lift/release. Each branch restores the same simulator state.
   Defaults give 264 short trajectories and 7,964 recorded states.
2. **Audit.** Check restores, matched robot postures, file/action alignment,
   scene splits and successful/failed outcome coverage. Record real camera
   frames, robot values, cube coordinates, held/contact labels and true simulator
   rewards. Inspect the collection contact sheets if an audit fails.
3. **Encode.** Encode real causal 64-frame camera histories with the same frozen
   V-JEPA model and pinned revision as the original feature cache. Save vectors;
   no encoder fine-tuning or JEPA predictor is used.
4. **Evaluate choices and export.** Keep all existing D/Q checkpoints frozen.
   At the original source and genuine later branch points, compare distinct
   candidate sequences over 4, 8, 16 and 30 control steps. Identical action
   prefixes are merged. Verify that alternatives share starting observations,
   robot values and camera history. Recurse D from the actual starting z/p;
   every future z/p used by the chooser is predicted. Save every forecast,
   choice, abstention and actual answer.

At 10 Hz, 4/8/16/30 steps mean 0.4/0.8/1.6/3 seconds. Branch points permit short
tests while already holding the cube, as well as decisions before grasping.
Reports separate held and unheld starting states. Overlapping decisions remain
correlated within scene groups; their count is not an independent trial count.

### The task score and ground truth

**Actual local success:** the cube is at least **5 cm above its original
pre-grasp height**, is actually held throughout the final **three recorded
states**, and the candidate has no recorded robot/table or obstacle contact.
These are the existing simulator labels: arm/table contact excludes fingers;
obstacle contact includes the robot and object. They are not exhaustive safety checks.
Three end states are a short stability check, not a guarantee of prolonged
holding. This lift objective does not measure reaching the placement target.

**Predicted qualification:** Q reads D's predicted future z/p and supplies cube
height and held probabilities. A candidate must meet the same 5 cm height goal
throughout its final three states, exceed a held-confidence threshold throughout
those states, and predict physically plausible finger openings. The total width
range is 0–8 cm, with a predeclared 2 mm tolerance. Large predicted height cannot
compensate for low held confidence. Among qualified candidates, choose the highest
minimum held confidence, breaking ties by lower action effort then branch name.
If none qualifies, record **ABSTAIN**; no fallback action is executed.

The confidence threshold is selected separately for each seed/horizon using
**only fresh validation Q outputs on actual future states**, then shared by all
D variants. The fixed threshold grid is 0.5/0.7/0.8/0.9/0.95/0.99. Support requires
at least three true positive candidate cases from at least two scene groups and
empirical precision of at least 0.9. Among supported settings, select highest
recall, then lowest threshold. Unsupported calibration forces abstention.
This does not establish a calibrated probability of joint lift/hold success.
Test outcomes never select thresholds or checkpoints.

The scorer's reference height comes from Q on the **past pre-grasp observation**.
Simulator coordinates/contact/rewards are answers used for offline evaluation;
they do not enter D or choose actions. Q is checked separately on real fresh
validation/test states. A failed Q gate marks those object-based choice results
`diagnostic_only`. Real future Q is an explicit diagnostic control, not an
available runtime observation. There is currently no collision predictor: actual
selected collisions are reported, but the chooser cannot forecast them.

### What we compare

| Report version | Future used to choose | Purpose |
|---|---|---|
| `original` | Original D → unchanged Q | Existing baseline |
| `robot` | D with p/action width correction → unchanged Q | Correction without vision in the width head |
| `visual` | D with z/p/action width correction → unchanged Q | Does vision in the width head improve choices? |
| `unchanged_z_predicted_p` | Starting z repeated, original D's predicted p → Q | Does predicted visual change help? |
| `real_Q_diagnostic` | Actual future JEPA z and measured p → Q | Can Q and this rule recognize actual successful outcomes? |

Both corrected complete models still use JEPA in D_z/Q. This is not a full
JEPA-versus-another-encoder comparison. Reports also compare always choosing
close/lift, the expected success of uniformly random candidate choice, and
the best success achievable within the recorded candidate set (oracle bound).
The oracle uses actual outcomes only for evaluation, never for model choices.

### How to read the results

Read `../results/decision_v1/reports/summary.txt`, then `summary.csv` and the
per-seed JSON/CSV files:

- **`candidate_success_coverage`:** fraction of decision sets with at least one
  actually successful candidate. Low coverage means the available moves rarely
  solve the task. All-failure sets cannot demonstrate good action ordering.
- **`chosen_success_rate_when_success_available`:** how often the model selected
  a success when one was available; abstention counts as not selecting success.
  Empty eligible sets return `null`, not perfect agreement.
- **`chosen_success_rate_all_decisions`:** actual successful selections divided
  by all decision sets. Compare with random/oracle baselines on those decisions.
  Some later groups have no close/lift candidate; compare that baseline with
  `chosen_success_rate_on_close_lift_decisions`, using its reported
  `always_close_lift_decisions` count.
- **`false_successes` / `false_success_rate_selected`:** qualified selections
  that actually failed. This exposes confident but incorrect predictions.
- **`abstention_rate`:** fraction with no selected candidate. Very high abstention
  can make false-success rates look good without achieving the task.
- **`candidate_success_precision` / `recall`:** qualification accuracy across
  distinct candidates, separately from the single selected choice.
- **`Q_gate_passed`, `calibration_supported`, `diagnostic_only`:** interpretation
  limits. Check the JSON's Q metrics and threshold sweep before claiming a gain.
- **`candidate_width_invalid_fraction`, `selected_collision_count`:** physical
  forecast failures and actual unsafe selected outcomes. Simulator return is
  saved separately; it is not this new binary local-success definition.

`seed_N_candidates.csv` includes candidate actions, aliases, predicted scoring
features and actual success/contact/return. `seed_N_forecast_steps.csv` includes
actual versus predicted height, held probability and finger width at each step.
`seed_N_decisions.csv` shows selected candidate or ABSTAIN, coverage and regret
(binary success lost relative to the best candidate, for selected actions).
For example, predicted close/lift qualifying but actually dropping the cube
counts as false success, even if its latent prediction error is small.

Full stage logs remain in `data/decision_v1/logs/`; raw frames and encoded
vectors remain in `data/decision_v1/observations/`. The automatic export contains
reports, full text logs, audit, scene/action settings, source/model/code hashes,
timings and a file hash index. It excludes raw frames, latent arrays and weights.
Wait for **`COMPLETE: reports/logs exported`**, then share from the repository root:

```bash
cd ..
git add -- results/decision_v1
git --no-pager diff --cached --stat
git commit -m "Record stable-lift candidate-choice experiment"
git push origin M2_Kuba
git rev-parse --short HEAD
```

The scorer has a small dependency-light logic check, runnable from `simulation/`:

```bash
../.venv/bin/python -m world_model.vision.decision --check
```

### Visualize saved decisions, predictions and real movements

From `simulation/`, after pulling the visualization helper:

```bash
../.venv/bin/python -m world_model.vision.visualize_decisions
```

This reads the exported seed-0 reports and the notebook's existing recordings.
It loads no models and needs no GPU. Under
`artifacts/decision_v1/visualizations/` it creates:

- `choices_seed_0.png`: successful selections, chosen failures and abstentions
  at each horizon, with candidate coverage and Q-gate status.
- `scene_0011_under_close_lift_seed_0.png`: actual versus predicted cube rise,
  held probability and finger width for a successful real lift that D largely misses.
- `scene_0011_under_real_branches.gif`: the **recorded real** close/lift,
  open/lift and close/lift/release camera sequences side by side.

Open the GIF/PNGs in Jupyter's file browser. If only the Git export is available,
charts still work and the real GIF is skipped. The helper verifies that the
recording manifest matches the report before generating a video. The rise
curves use the same references as the scorer: actual pre-grasp height for real
labels, Q's estimate on the past pre-grasp observation for predictions.
Q on real future observations is plotted only as a diagnostic.

Try the contrasting false-positive example or another model-fitting seed:

```bash
../.venv/bin/python -m world_model.vision.visualize_decisions --scene scene_0012
../.venv/bin/python -m world_model.vision.visualize_decisions --seed 1
```

Choose `--placement offset` or `--branch close_lift_release` to inspect other
recorded candidates. GIFs show physical simulator outcomes. D predicts numeric
latent/robot states; no decoder currently converts those predictions into an
imagined robot video. The plots are how we inspect those forecasts.

## Establish the exact-state RL baseline first

`control_v1` on the notebook placed 3/5 with exact-state SAC, and 0/5 with
both visual controllers. We now establish reliable A-to-B control separately
before adding JEPA. This runner needs no encoder, latent cache, D or Q weights.

From the repository root on the notebook:

```bash
git pull --ff-only origin M2_Kuba
.venv/bin/python -m pip install -r requirements-control.txt
cd simulation
../.venv/bin/python -u -m world_model.vision.rl_baseline \
  --out data/rl_baseline_v1
```

If already in `simulation/`, install with
`../.venv/bin/python -m pip install -r ../requirements-control.txt` and run the
last command. Use the notebook's existing OSMesa setup; this runner requires
no CUDA. Training and inference use CPU. On macOS, ordinary Python works for
these saved offscreen videos; `mjpython` is only required for the interactive
MuJoCo viewer.

The single command:

1. Collects **100 successful scripted demonstrations** of the complete task.
2. Initializes the SAC actor by **240 epochs of behaviour cloning (BC)**.
   BC means learning to copy recorded actions. Its score is reported separately.
3. Fits critics on demonstrations for 2,000 updates, then runs **20,000 real
   SAC environment steps**, retaining the declared demonstration regularizer.
   This is **demonstration-assisted RL**, not training from scratch.
   The learning rate is `3e-5`, initial automatic entropy coefficient `0.005`;
   these are proposed stabilizing settings, not a guaranteed fix.
4. Evaluates SAC every 2,500 training steps on **20 validation episodes**.
   Only validation placement success, then goal distance, selects the checkpoint.
5. Freezes the selected SAC checkpoint and evaluates **BC and SAC separately
   on the same 100 fresh episodes**. Final-test scores do not select weights.
6. Saves actual videos/frames/trajectories for the first five fresh seeds for
   scripted, BC-only and SAC control. No scripted actions or fallback are used
   during BC/SAC evaluation: the learned actor selects every action.
7. Exports JSON/CSV reports and logs to `results/rl_baseline_v1`.

The arm, original IK and strict success check remain the same: cube grasped,
lifted at least 4 cm, then released and settled within 7 cm of B for 15 steps.
Scope: a fixed unit cube, empty table, fixed B, A jittered by ±1 cm per axis.
It does not establish reliability on obstacles, different shapes or arbitrary
positions. JEPA stays disconnected during this run.

### Reading the baseline output

- `complete=true`: the procedure finished; this is not a success claim.
- `gate_passed=true`: SAC placed successfully in **at least 90/100** fresh
  episodes, with at least 20 validation episodes and 100 final-test episodes.
- `BC_test_successes` and `SAC_test_successes`: imitation versus reward-trained
  policy. Better SAC performance must be observed, not assumed.
- `selected_rl_steps`: validation-selected SAC checkpoint, not the last weights.
- `evaluations.json`: per-seed `ever_held`, `ever_lifted_4cm`, final distance,
  contacts, steps and actual placement outcome, so pick failures and misplaced
  releases can be distinguished.
- `rl_training.json` and `run_info.json`: learning settings, training/validation
  curve, checkpoint hash, exact code/configuration hashes and seed lists.

Saved videos are on the notebook, outside Git:

```python
from pathlib import Path
from IPython.display import Image, display
root = Path('data/rl_baseline_v1/episodes')  # notebook cwd: simulation/
for method in ('scripted', 'bc_true', 'rl_true'):
    video = sorted((root / method).glob('*/actual.gif'))[0]
    print(method, video)
    display(Image(filename=str(video)))
```

Resume an interrupted run with the same command plus `--resume`. The runner
checks code/settings and completed checkpoint/evaluation hashes. Preserve old
runs when changing settings; use a new output name. Raw frames/checkpoints
stay under `simulation/data/rl_baseline_v1` and are not pushed.

Once complete, from `simulation/`:

```bash
cd ..
git add -- results/rl_baseline_v1
git --no-pager diff --cached --stat
git commit -m "Record exact-state RL baseline evaluation"
git push origin M2_Kuba
```

**Do not add JEPA until the baseline passes.** Then freeze this exact SAC
checkpoint and compare true-state inputs, JEPA/Q observations, and D/Q planning
on common new scenes. A separately retrained actor is a different baseline.

### Observed local baseline reference

The new Mac CPU run (`results/rl_baseline_local_v1`) completed: BC-only93/100,
selected SAC100/100 fresh placements, mean final distance0.369cm and maximum
0.972cm. All eight SAC validation checkpoints scored20/20; validation selected
the20,000-step checkpoint. This is one fixed-scene demonstration-assisted fit.
The notebook reproduction is reported below. The assisted stage retained BC
training, so its improvement cannot be attributed solely to reward gradients.

## Reconnect JEPA using the passing notebook SAC checkpoint

The notebook baseline terminal output reported **SAC 100/100** and **BC 99/100**,
with selected SAC checkpoint SHA256
`b091f73ce7952bb1abaa2f663705a277df3f1c1e8c780ef8d0ed99e1f36f8248`.
The next command reuses that exact checkpoint; it does not train SAC again.
Existing frozen JEPA, visual readout Q and corrected dynamics D are reused too.

From the notebook repository root:

```bash
git pull --ff-only origin M2_Kuba
cd simulation
../.venv/bin/python -u -m world_model.vision.control_pipeline \
  --baseline-run data/rl_baseline_v1 \
  --models-run data/vision_v2 --tag width_visual_seed_0 \
  --seed 20364005 --test-episodes 5 \
  --terminal-weight 0 \
  --out data/control_frozen_v1
```

If already in `simulation/`, run `git pull --ff-only origin M2_Kuba` there and
then the Python command. No installation or latent re-encoding stage is needed.
The existing CUDA/Transformers and OSMesa virtualenv must still work.

### What runs

1. Verify the saved baseline passed at least 90/100 actual placements, completed
   at least 20 validation episodes, and its checkpoint/evaluation hashes match.
   Verify the same physical configuration, reward/observation code, layout and
   starting-position variation. Reject overlapping seeds and source/output paths.
2. Copy the frozen SAC actor/critics into the new output. Print
   `REUSED FROZEN SAC ... no training`. Never modify the source baseline.
3. Execute these four controllers from home on the **same five new scenes**:

   | Method | Current object information | How actions are selected |
   |---|---|---|
   | `scripted` | Exact simulator coordinates | Original waypoint policy; reference only. |
   | `rl_true` | Exact simulator coordinates/contact | Frozen successful SAC actor. |
   | `rl_q` | Actual camera clip → frozen JEPA → visual Q | Same SAC actor using Q estimates. No D forecasting. |
   | `jepa_mpc` | Actual camera clip → frozen JEPA → visual Q | SAC-guided CEM candidates, frozen D forecasts, visual Q readout and task-reward scoring. |

4. Save actual images/GIFs, per-step chosen actions and task outcomes, Q estimates,
   predicted-versus-actual next states, candidate scores and rejection reasons.
5. Export shareable reports/logs under `results/control_frozen_v1`.

No obstacles are present: the original cube/empty-table task is retained.
Known destination B is explicitly provided to all controllers. True future
cube/contact values are evaluator answers and never enter JEPA action selection.
`rl_true` uses true **current** state as its declared privileged baseline.

### There is a reward formula, not a reward network

`GoalReward` is the fixed reach/grasp/lift/transport/placement progress formula,
plus strict-placement bonus and penalties. It was written in code, not fitted
from labels. During SAC training, its **actual** rewards train the actor and
SAC's two critics. The critics estimate future cumulative reward; they are not
our visual readout Q and are not a learned instantaneous reward function.

In planning, each candidate uses:

```text
Current camera clip → JEPA → z0
Measured robot values → p0
Known destination → B

SAC proposal + sampled variations → candidate action sequences
    for each sequence:
        D(current z, current p, proposed action) → next imagined z and p
        visual Q(next imagined z, next imagined p) → cube xyz / held probability
        reward formula(previous history, imagined state, action, B) → reward
        repeat with the next candidate action
    sum discounted rewards across the sequence
Choose the highest-scoring valid sequence
Execute only its FIRST action through the original IK controller
Observe new real camera frames and robot values; repeat
```

CEM samples 64 sequences of 8 actions, refines them for 3 iterations, retaining
8 elites. These are proposed alternatives, not a list of actions proven best
by SAC. Future states are predictions and can be wrong. Candidates with invalid
numeric/physical forecasts are rejected; if every candidate is rejected, the
Q-observed SAC proposal is executed and the fallback is reported.

The command uses `--terminal-weight 0`: scores contain discounted predicted
rewards only. A separately named future run may use `--terminal-weight 1` to
add the pretrained SAC critic's estimate of reward beyond the 8-step horizon.
Reward history is copied separately for every candidate. Imagined obstacle,
proximity and table-contact penalties remain omitted because Q cannot predict
those contacts; actual evaluation retains them. No success of JEPA control is
implied by the working exact-state SAC baseline.

### What to inspect and share

- `summary.txt`/`summary.json`: placements, goal distance, fallback counts; confirm
  the frozen checkpoint hash is identical for `rl_true`, `rl_q` and `jepa_mpc`.
- `frozen_baseline.json`: source checkpoint and passing 100-episode evidence.
- `episodes/<method>/seed_<seed>/steps.csv`: actual motion, Q estimates and
  one-step forecast errors. A small error while the cube stays stationary is
  not evidence of successful manipulation.
- MPC `candidates.csv`: valid flags and `invalid_reasons`, such as
  `finger_width`, `D_nonfinite` or `Q_nonfinite`. Reasons accumulate over the
  candidate horizon. Only the retained CEM pool per decision is archived.
- Raw `actual.gif`, frames and forecast NPZs remain under
  `simulation/data/control_frozen_v1/episodes`; use the earlier GIF display
  snippet with this run's root.

Read the three learned modes in order: `rl_true` checks the frozen controller
on the new scenes; `rl_q` exposes current-perception errors; `jepa_mpc` adds
learned forecasting/search. The final comparison does not isolate JEPA from D/Q
or prove it beats other encoders. Current D/Q full-task coverage remains limited.

Exact interrupted resume: repeat the command with `--resume`.
Once complete, from `simulation/`, export both the new baseline evidence and
this comparison so the results can be inspected locally:

```bash
cd ..
git add -- results/rl_baseline_v1 results/control_frozen_v1
git --no-pager diff --cached --stat
git commit -m "Record frozen SAC versus JEPA control comparison"
git push origin M2_Kuba
```

## Share trained runs and videos with Calle

The earlier `results/` exports contain reports/logs, not trained weights, cached
latents or raw frames. These remain on the notebook until explicitly uploaded.
The sharing policy now permits these exact folders under `simulation/data/`:
`vision_v2`, `rl_baseline_v1`, `control_v1`, `control_frozen_v1`, and
`control_frozen_v2`. Their entire contents use Git LFS. Agent notes, virtualenvs,
other generated runs and temporary artifacts retain their ignore rules.

### Upload from Kuba's GPU notebook

Work in the **repository root**, not `simulation/`. The data is on this notebook;
running the upload from the Mac will not supply missing notebook files.

```bash
cd "$HOME/Robots&EmbodiedAI/project/world_model_project"
git lfs version
git lfs install --local
git pull --ff-only origin M2_Kuba
git branch --show-current
```

The branch must be `M2_Kuba`. If `git lfs` is missing, install the
[official Git LFS client](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md)
first. The Ubuntu command is `sudo apt-get install git-lfs` when sudo is available.
On the teaching notebook's Linux x86_64 environment, this installs the official
v3.7.1 binary without sudo, checking its published SHA256:

```bash
(
  set -e
  lfs_tmp=$(mktemp -d)
  trap 'rm -rf "$lfs_tmp"' EXIT
  cd "$lfs_tmp"
  curl -fL --retry 3 -o git-lfs.tar.gz \
    https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz
  printf '1c0b6ee5200ca708c5cebebb18fdeb0e1c98f1af5c1a9cba205a4c0ab5a5ec08  git-lfs.tar.gz\n' | sha256sum -c -
  tar -xzf git-lfs.tar.gz --strip-components=1
  mkdir -p "$HOME/.local/bin"
  install -m 755 git-lfs "$HOME/.local/bin/git-lfs"
)
export PATH="$HOME/.local/bin:$PATH"
git lfs version
```

Repeat the PATH export in future shells if `$HOME/.local/bin` is not already on
PATH. Other architectures should use their matching official package. Git LFS
is a Git client tool, not a Python package or CUDA dependency. The binary/checksum
come from the [official release](https://github.com/git-lfs/git-lfs/releases/tag/v3.7.1).

First check size:

```bash
du -sh simulation/data/vision_v2 simulation/data/rl_baseline_v1 \
  simulation/data/control_v1 simulation/data/control_frozen_v1
```

Ordinary GitHub pushes reject individual files over100MiB. Git LFS has its own
per-file limits and storage/download allowances. The repository owner's quota
applies; check available allowance before uploading all frames/forecasts. GitHub
Free/Pro currently include10GiB storage and10GiB download bandwidth, while the
Free/Pro per-file LFS limit is2GB. If these runs exceed the available quota, upload
the compact runtime set first and share bulk data through an agreed data store or
GitHub release assets instead of putting large binaries into ordinary Git.
See [file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
and [LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).

**Minimum runtime set:** enough for `control_pipeline --baseline-run ...`; it
does not include raw training frames or the cached latent array.

```bash
(
  set -e
  git add -- .gitattributes .gitignore README.md
  git add -- simulation/data/vision_v2/manifest.json \
    simulation/data/vision_v2/features/meta.json \
    simulation/data/vision_v2/attempts/width_visual_seed_0/models/dynamics.pt \
    simulation/data/vision_v2/attempts/width_visual_seed_0/models/readout.pt
  git add -- simulation/data/rl_baseline_v1/pipeline.json \
    simulation/data/rl_baseline_v1/summary.json \
    simulation/data/rl_baseline_v1/evaluations.json \
    simulation/data/rl_baseline_v1/models
  git --no-pager diff --cached --stat
  git lfs status
  git commit -m "Share frozen SAC and JEPA control checkpoints"
  git push origin M2_Kuba
  git rev-parse --short HEAD
)
```

Include **all baseline model files**: `frozen_baseline()` verifies every entry in
`pipeline.json`'s `trained` map, including `bc_initial.zip`, even though the live
controller executes `sac_best.zip`. Preserve original JSON/checkpoint bytes so
their recorded SHA256 hashes stay valid. Do not edit absolute provenance paths
to another person's home directory. Those describe the source run; new experiments
use new output folders instead of resuming an archived pipeline on another host.

**Full data and recordings:** after checking available storage, upload the
completed run trees. This supplies cached `features/latents.npy`, all model tags,
raw images, simulator states, trajectories, actual GIFs and candidate forecasts.
That permits D/Q retraining with the same encoder without recollection/re-encoding.

```bash
(
  set -e
  git add -- simulation/data/vision_v2 simulation/data/rl_baseline_v1
  for run in control_v1 control_frozen_v1; do
    if [ -d "simulation/data/$run" ]; then
      git add -- "simulation/data/$run"
    fi
  done
  git --no-pager diff --cached --stat
  git lfs status
  git commit -m "Share vision training data and controller recordings"
  git push origin M2_Kuba
  git rev-parse --short HEAD
)
```

Add `simulation/data/control_frozen_v2` the same way **after its running pipeline
finishes**. Commit a finished snapshot, not files changing during training or
collection. A normal `git push` uploads the required LFS objects through the
installed hook. If authentication fails, use a GitHub token/SSH authentication;
an account password is not accepted. Never put tokens into files or remote URLs.
Do not use `git add .` or force-add the entire repository for these uploads.

### Download on Calle's computer

Install Git LFS first. For a fresh checkout, this avoids downloading every raw
frame before Calle chooses which runs he needs:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch M2_Kuba --single-branch \
  https://github.com/Sarvesh-kth/world_model_project.git
cd world_model_project
git lfs install --local
git lfs pull --include="simulation/data/vision_v2/**,simulation/data/rl_baseline_v1/**,simulation/data/control_frozen_v1/**" --exclude=""
```

For an existing clone:

```bash
git switch M2_Kuba
GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin M2_Kuba
git lfs install --local
git lfs pull
```

For only the minimal runtime assets, use this selection instead of downloading
all raw data (after cloning/pulling pointer files):

```bash
git lfs pull --include="simulation/data/vision_v2/manifest.json,simulation/data/vision_v2/features/meta.json,simulation/data/vision_v2/attempts/width_visual_seed_0/models/**,simulation/data/rl_baseline_v1/pipeline.json,simulation/data/rl_baseline_v1/summary.json,simulation/data/rl_baseline_v1/evaluations.json,simulation/data/rl_baseline_v1/models/**" --exclude=""
```

If a supposed `.pt`, `.npy` or `.jpg` is a tiny text file beginning with
`version https://git-lfs.github.com/spec/v1`, it is still an LFS pointer. Run
`git lfs pull` for that path. The upload is complete only after both Git objects
and LFS objects are successfully pushed; a commit hash alone is not proof.
Downloading the pinned base V-JEPA model from Hugging Face remains automatic;
the pretrained encoder is not copied into this Git repository.

### Watch and rebuild videos without GPU inference

From a notebook whose current directory is `simulation/`:

```python
from pathlib import Path
from IPython.display import Image as DisplayImage, display

root = Path("data/control_frozen_v1/episodes")
for method in ("scripted", "rl_true", "rl_q", "jepa_mpc"):
    video = root / method / "seed_20384005" / "actual.gif"
    print(method, video)
    if video.is_file():
        display(DisplayImage(filename=str(video)))
    else:
        print("Recording not downloaded; select this episode with git lfs pull.")
```

The latest comparable movies are in `control_frozen_v1`; `control_v1` contains
the older weaker-baseline experiment. The20-episode `control_frozen_v2` batch uses
seeds20394005..20394024. `actual.gif` shows actual executed motion, not decoded
JEPA predictions. All four controllers automatically create it from their saved
static camera frames. No CUDA or MuJoCo viewer is needed to display it.

To rebuild a GIF from an episode's existing JPEGs, run from `simulation/`:

```bash
../.venv/bin/python - <<'PY'
from pathlib import Path
from PIL import Image

episode = Path("data/control_frozen_v1/episodes/jepa_mpc/seed_20384005")
paths = sorted((episode / "frames").glob("*.jpg"))[::2]
if not paths:
    raise SystemExit("No frames: download this episode's frames with git lfs pull")
frames = []
for path in paths:
    with Image.open(path) as im:
        frames.append(im.convert("RGB").resize((256, 256)))
out = Path("artifacts/control_videos/jepa_mpc_seed_20384005.gif")
out.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(out, save_all=True, append_images=frames[1:], duration=200, loop=0)
print(out.resolve())
PY
```

This matches the runner's10Hz frames, every-second-frame sampling and200ms GIF
duration. Generated copies stay under ignored `artifacts/`. To inspect decisions,
use each episode's `steps.csv`, `trajectory.npz`, `candidates.csv`,
`selected_forecasts.csv`, and `forecasts/` NPZs. Only MPC has candidate forecasts.

### Rerun the frozen control comparison on a CUDA machine

Follow the Linux/CUDA and headless-rendering setup earlier in this README,
including `requirements-control.txt`. From `simulation/`, after downloading the
runtime set, use a **new** output directory:

```bash
../.venv/bin/python -u -m world_model.vision.control_pipeline \
  --baseline-run data/rl_baseline_v1 \
  --models-run data/vision_v2 --tag width_visual_seed_0 \
  --seed 20364005 --test-episodes 5 \
  --horizon 8 --population 64 --elites 8 --iterations 3 \
  --terminal-weight 0 --out data/control_calle_v1
```

This reuses the frozen SAC and D/Q, generates actual recordings for all four
methods and evaluates seeds20384005..20384009. It does not retrain the encoder,
actor or readout. Source-contract/hash checks still apply. Ordinary GIF viewing
works on Mac/Linux; this live JEPA runner requires CUDA. Reproducing the experiment
does not imply visual placement succeeds; inspect the saved outcomes.


## One-command full-task Q and rectangular-clutter experiment

This is a **new experiment**, not a claim that visual control or RL recovery already works. It preserves the original passing SAC checkpoint and the previous `vision_v2` models. If the original SAC fails a validation-only recovery probe, it trains a separate recovery actor and freezes that actor for all final comparisons. The new entry point is `world_model.vision.clutter_pipeline`.

### What it runs

1. Verify the frozen SAC source: completed baseline, >=90/100 fresh placements, checkpoint hashes and unchanged physics/observation/reward. Check CUDA and render one frame before collection.
2. Collect 22 whole scene groups by default: 12 training, four validation, six test. Each has the same cube/A/B/reset seed in empty and rectangular-clutter variants, and four cases. Thus there are 176 collected trajectories. Normal empty trajectories use the frozen SAC; forced-release trajectories use the existing scripted controller, restarted after the intervention to demonstrate recovery. Replay each empty trajectory's **executed** actions in its clutter pair. Failures are retained; unexpected collection contacts or changed intervention timing stop the audit.
3. Audit complete-scene splits, paired starting states/actions, camera histories, labels and files. All variants of a scene stay in the same split.
4. Probe the original actor on validation scenes in all eight empty/clutter × normal/recovery conditions. If the minimum condition success is below 80%, train a **new** SAC in `recovery_rl/`, initialized from the source checkpoint: successful training-scene demonstrations, 240 BC epochs, 2,000 critic warmup updates and 20,000 online RL steps by default. Select by minimum per-condition validation success, then mean goal distance. External forced-opening actions are excluded from actor imitation; Q/D still learn their visual consequences. Online recovery episodes start after a real scripted release, so SB3 always stores the action actually executed by the actor. The original checkpoint is never overwritten. A bounded fit can still fail; the final gate reports that failure.
5. Freeze the selected actor (original if the probe passed, otherwise the new fit) and run exact-state SAC and scripted reference on final test scenes. Save actual GIFs/frames/actions/contacts/outcomes. No final test result chooses the actor or a checkpoint.
6. Encode every saved causal static-camera history with the same pinned, frozen V-JEPA encoder: 64 frames, first-frame padding when necessary, mean of encoder tokens, normally 1,024 numbers. Neither actions nor simulator object coordinates enter JEPA.
7. Fit **Q-empty** on complete empty-table paths and **Q-mixed** on both variants, including carry, release and recovery. Both predict cube xyz and held probability from `(z, p)`. Normalize and train on the training split; select checkpoints using validation only. Each also trains a diagnostic p-only readout. Q-mixed has twice as many view examples: this comparison measures the practical mixed-data recipe, not an isolated effect of obstacles with matched update counts.
8. Train a shared split D on mixed full-task transitions using eight-step recursive loss. Then fit the existing visual finger-width correction with D frozen. Both Q variants use this **same** corrected D in control. Also fit a blind action-sequence baseline on varied starting states, using only measured starting p and proposed actions.
9. Evaluate both Q models on identical real held-out clips, by view and phase: approach, grasp/lift, carry, lower, release/settling and recovery. Report xyz/height errors, false-held/missed-held counts, F1, calibration and delay before Q notices a real release.
10. Evaluate D at 1/4/8/16 steps against unchanged latents, deliberately wrong actions and the blind baseline. Simulator outcomes are answers, never D inputs. Incorrect-action forecasts are compared with the recorded outcome of the original action sequence; this is an action-dependence diagnostic, not a new simulator counterfactual.
11. Compare six controllers on the same five test scene groups, two views and four cases: scripted, exact-state SAC, SAC+Q-empty, SAC+Q-mixed, CEM+D+Q-empty, CEM+D+Q-mixed. This makes 240 control episodes. Execute only the first selected action, observe again and replan. Save predictions, rewards, validity/fallback reasons and actual videos.

Rectangles are fixed within each scene group and vary across groups. The rectangles are physical boxes with randomized positions, yaw and heights **5–20 cm**, in a strip at the far edge of the existing table. They stay away from the intended A-to-B corridor. This first campaign tests clutter sensitivity; it does not establish avoidance of blocking obstacles. The current 41-number actor observation contains no obstacle geometry, and D/Q cannot predict collision penalties. Actual obstacle contacts are still measured and penalized by the environment. The reports state this scoring limitation.

**Forced release:** after a held cube has moved 15%, 50% or 75% of its original distance towards B, the evaluator overrides seven actions with “stay here and open”. Gravity drops the cube; no cube teleport, robot reset or camera-history reset occurs. Then the controller regains control. Commands and forced interventions are recorded separately. A case that never reaches the trigger is a failure to reach that test condition, not a recovery success. The visual controller receives camera history and measured p, not the evaluator's true cube/contact labels.

**Goal and reward:** B is still passed explicitly through the existing policy observation and the fixed `GoalReward` scoring function. No reward network is trained. A separate recovery SAC actor is fitted only if the validation probe fails; the reward equations remain unchanged. Success still requires lift, arrival at B, release and 15 settling steps. The new planner horizon defaults to **20 steps (2 s)** so that a release near the beginning can include settling. This setting is an experiment, not a proved fix. Terminal critic weight remains zero. No bonus is added merely for attempting recovery.

### Start on the notebook

The existing CUDA virtualenv, OSMesa setup, `data/rl_baseline_v1` and `data/vision_v2/features/meta.json` are required. Raw `vision_v2` images are not used: this campaign collects new full-task images. Run from `simulation/`:

```bash
cd "$HOME/Robots&EmbodiedAI/project/world_model_project/simulation"
git -C .. pull --ff-only origin M2_Kuba
../.venv/bin/python -m pip install -r ../requirements-control.txt

../.venv/bin/python -u -m world_model.vision.clutter_pipeline
```

That last command runs every stage and exports results automatically. It can take several hours: encoding every observation and running the live JEPA/CEM comparisons are the expensive parts. A 20 GB MIG GPU is enough for the existing single-clip frozen encoder/bounded training batches; the preflight verifies the actual CUDA stack. Run the exact same command again after an interruption: finished stages are authenticated and skipped, encoding continues from its saved progress, collection continues at trajectory boundaries, and interrupted Q/D fits restart that fit while preserving its partial files. Interrupted SAC fitting restores its last committed weights and replay buffer; the simulator resets on resume. Completed control episodes are reused; an unfinished episode restarts from reset and its old recordings are archived. Settings, source checkpoint or code changes require a fresh `--run` **and** `--export`. Do not update code during an active run.

To survive an SSH/browser disconnect, start the same pipeline in the background once:

```bash
mkdir -p data
nohup ../.venv/bin/python -u -m world_model.vision.clutter_pipeline \
  > data/q_clutter_v1.console.log 2>&1 < /dev/null &
tail -f data/q_clutter_v1.console.log
```

Do not launch two processes into the same run. The pipeline takes an OS file lock, rejects a concurrent run and releases the lock when the process exits. `Ctrl-C` exits `tail`, not the background run. A notebook server shutdown can still stop its processes.

Optional smaller real-GPU pilot, with separate paths:

```bash
../.venv/bin/python -u -m world_model.vision.clutter_pipeline --pilot \
  --run data/q_clutter_pilot --export ../results/q_clutter_pilot
```

The pilot uses 3/1/1 groups, three fit epochs and one control scene. It exercises the full procedure; it cannot pass the five-episode reference criterion. Defaults for Q/D are one training seed, 60 epochs, 64 candidates, eight elites and three CEM iterations. Optional recovery fitting uses `--rl-steps 20000 --bc-epochs 240 --critic-warmup 2000`; the pilot reduces these to 1,000/2/20. Change scene/epoch counts only on a fresh run, for example `--train-scenes 36 --val-scenes 8 --test-scenes 8 --control-scenes 5 --run data/q_clutter_v2 --export ../results/q_clutter_v2`. One training seed is a first comparison; repeat with fresh runs/seeds before claiming a stable advantage.

### Where to read the results

| File or directory | Meaning |
| --- | --- |
| `simulation/data/q_clutter_v1/pipeline.json` | Exact settings, source/code hashes and completed stages. `complete` means the jobs finished, not that the models succeeded. |
| `results/q_clutter_v1/reports/summary.txt` | Controller placement/recovery/contact/fallback summary. |
| `results/q_clutter_v1/reports/control_summary.csv` | Each method × Q × empty/clutter × release case, with denominators. |
| `results/q_clutter_v1/reports/offline.json` | Phase/view Q errors, release lag and multi-step D comparisons. |
| `results/q_clutter_v1/reports/Q_real_states.csv` | Real xyz/held and both Q estimates at every evaluation state. |
| `results/q_clutter_v1/recovery_rl/` | Source validation probe, separate SAC fitting/validation record and selected actor hash. |
| `results/q_clutter_v1/attempts/*/reports/` | Training curves, validation selection and Q gates. |
| `simulation/data/q_clutter_v1/attempts/q_empty/models/readout.pt` | Empty-only Q checkpoint. |
| `simulation/data/q_clutter_v1/attempts/q_mixed/models/readout.pt` | Mixed-view Q checkpoint. |
| `simulation/data/q_clutter_v1/attempts/shared_width/models/dynamics.pt` | Shared corrected visual/robot dynamics checkpoint. |
| `simulation/data/q_clutter_v1/control/*/*/episodes/*/seed_*/` | Actual GIF, frames, step CSV, trajectory NPZ, candidate forecasts. These large files stay in the raw run. |
| `simulation/data/q_clutter_v1/logs/` | Full sequential stage logs/tracebacks. |

Interpretation order: first check exact-state SAC placements and triggered recovery, then Q on real observations, then imagined outcomes, then closed-loop placement. **Final reference gate:** >=80% placements over >=5 episodes in every view/case, zero rectangle-contact steps, and >=5 triggered recovery episodes for each release case. If it fails, the visual results are still exported, but they do not show a successful RL reference under that condition; improve the recovery curriculum or add obstacle-aware RL in a fresh experiment before claiming the full robot solution works. Q gates remain unchanged; failed gates are diagnostic results, not software crashes. Nonfinite offline forecasts are counted in `invalid_windows`; reported errors cover only the common finite windows. Inspect those counts before comparing errors. Lower errors or higher F1 alone are not successful A-to-B control. `empty_p`/`mixed_p` in offline reports are diagnostic readouts without z. When attached to the corrected D, its predicted robot width can still depend on vision; the separate `blind` action-sequence model never sees z and is not an architecture-matched latent ablation.

Display an actual video in a notebook:

```python
from pathlib import Path
from IPython.display import display, Image
run = Path("data/q_clutter_v1")  # notebook working directory: simulation/
for video in sorted(run.glob("control/*/*/episodes/*/seed_*/actual.gif"))[:6]:
    print(video)
    display(Image(filename=str(video)))
```

Commit the compact results after the campaign finishes:

```bash
cd ..
git add -- results/q_clutter_v1
git --no-pager diff --cached --stat
git commit -m "Record full-task empty and clutter Q comparison"
git push origin M2_Kuba
git rev-parse --short HEAD
```

The default raw run is also allowlisted for Git LFS if Calle needs the weights and recordings. Follow the LFS installation/quota/setup instructions above before adding `simulation/data/q_clutter_v1`; compact reports alone do not include checkpoints.

### CPU implementation check

```bash
# Requires Torch/SB3 plus simulation dependencies, but no CUDA or HF download.
../.venv/bin/python -m world_model.vision.check_clutter --run data/q_clutter_check
```

This check runs real MuJoCo collection/drop/re-grasp/recording, audits paired actions/splits, fits tiny Q/D/blind/width models, verifies normalizers and report/export contracts, and rejects a false reference pass. It also exercises tiny recovery SAC updates, source initialization, replay resume and the no-fit copy branch; it checks that external opening commands are excluded from imitation and online stored actions match execution. It intentionally uses **synthetic, privileged visual vectors**. It tests code execution; its fit errors are not evidence about JEPA.
