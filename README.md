# JEPA World Model Pick and Place

This repository currently contains a MuJoCo simulation of a Franka Panda arm,
scripted pick-and-place demonstrations, episode data collection, and an initial
frozen V-JEPA 2 feature/dynamics training pipeline. A learned reward model and
learned controllers are planned work; the
`success` demo uses a scripted policy that reads simulator state.

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
virtual environment and generated `simulation/data/` are ignored by Git.

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
