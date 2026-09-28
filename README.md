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
cd simulation
```

The CUDA check must print `True` before encoding. The notebook driver reports
CUDA 12.4 support, so the command above uses PyTorch's matching CUDA 12.4 wheel.
Unqualified `pip install torch` currently selects a CUDA 13 wheel and prints
`False` with a driver-too-old warning on this notebook. To repair an existing
virtual environment, run the same pinned PyTorch install command with
`--force-reinstall`, then repeat the CUDA check. See [PyTorch's published wheel
commands](https://pytorch.org/get-started/previous-versions/).
The H100 MIG allocation shown for this project has about 20 GiB, so start with
the small run below. V-JEPA's weights are downloaded from Hugging Face on the
first encode; that command needs internet access and enough cache space.

```bash
# 1. Collect 10 episodes on one fixed cube/goal layout. No viewer is opened.
MUJOCO_GL=egl ../.venv/bin/python -m data_collection.collect \
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

If MuJoCo reports an EGL initialization error, rerun only the collection command
with `MUJOCO_GL=osmesa` if the cluster has OSMesa installed. Collection renders
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
MUJOCO_GL=egl ../.venv/bin/python -m data_collection.collect \
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
