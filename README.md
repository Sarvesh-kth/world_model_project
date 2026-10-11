# JEPA World Model Pick and Place

This repository contains a MuJoCo Franka Panda simulation, scripted demonstrations,
data collection, frozen V-JEPA 2 features, learned dynamics/readouts, a trained SAC
baseline and visual control. The `success` demo uses a scripted
policy that reads simulator state. The live comparison uses a fixed reward
formula plus a learned penalty head for contacts (see the current architecture section).

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

For the GPU parts install the CUDA build of torch first (see below). If you already have the repository, start with `cd world_model_project`. The
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

## Current architecture (branch `world_model_testing`)

This is the state of the project after the "New architecture" work and the v2 perception: a spatial
V-JEPA latent of the static camera (now looking down at about 60 degrees) and the wrist camera, the static
depth image as a table height map inside the latent, a readout `Q`, a penalty head `R`, a task-consistent
dynamics model `D`, and two controllers on that perception: the frozen SAC (the actor-critic, one forward
pass per step) and the latent MPC (Cartesian waypoint candidates imagined by `D`, scored by `Q` and `R`,
no policy in the loop). The day-by-day log, with what was measured and why each change was made, is
`STATUS.txt`. The code is organised in three packages under `simulation/`, each with a README:

```
simulation/
  environment/, data_collection/   the simulator and the episode collector       -> simulation/README.md
  world_model/     run folder, encode with V-JEPA, train Q R D, evaluate         -> simulation/world_model/README.md
  rl/              the exact-state SAC baseline (demos -> BC -> SAC)              -> simulation/rl/README.md
  controller/      the four controllers and the latent MPC (planner.py)          -> simulation/controller/README.md
  configs/         default.yml (every knob), grade_e.yml, grade_e_layout.json, full_mix.yml
  data/            runs: datasets, latents, trained models, controller episodes (git-ignored, see data/README.md)
  cluster_round.sh the whole chain (collect, encode, train, evaluate) in one command; CLUSTER.md step by step
results/           summaries the controllers write (git-ignored)
```

Commands run from `simulation/` with the venv active (`python` then is the venv's). The encoder and
the two camera-based controllers need Linux with an NVIDIA GPU; install the CUDA build of torch first:

```bash
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd simulation
```

### Rendering backend and environment variables

MuJoCo picks its OpenGL backend from `MUJOCO_GL` when `mujoco` is first imported, so the variable has to
be right for what you run:

- **Collectors, encoder, controllers, SAC training (offscreen camera):** they import
  `simulation/offscreen.py` before MuJoCo, which sets `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` and
  drops `__EGL_VENDOR_LIBRARY_FILENAMES` for that process. So on Linux nothing needs exporting. The
  manual equivalent, for your own scripts:

  ```bash
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  unset __EGL_VENDOR_LIBRARY_FILENAMES
  ```

  The `unset` matters on a laptop whose `.bashrc` pins the NVIDIA EGL vendor for other tools: when the
  NVIDIA userspace libraries and the loaded kernel driver disagree (`nvidia-smi` then prints
  "Driver/library version mismatch", typically after a driver update without a reboot) that vendor
  cannot open a headless display and MuJoCo fails with "Cannot initialize a EGL device display";
  without the pin EGL falls back to a vendor that works. A reboot fixes the mismatch itself.

- **The teaching notebook (no EGL):** run `bash setup_mujoco_headless.sh` once with the venv active
  (it downloads OSMesa into `.mujoco-osmesa/` without a system install), then export the OSMesa
  backend before any offscreen command:

  ```bash
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  ```

- **The viewer (`play.py`, the live window of `control_pipeline`):** needs a display and a working GLX.
  `simulation/window.py` probes which GLX vendor can open a window (the default, then Mesa) and starts
  the window with that environment, without the EGL variables and without the NVIDIA PRIME offload
  variables (`__NV_PRIME_RENDER_OFFLOAD`, `__GLX_VENDOR_LIBRARY_NAME`, which leave the window blank).
  When the NVIDIA driver and its libraries disagree, NVIDIA GLX fails ("GLX: Failed to create context:
  BadValue", `glxinfo` fails the same way) and the window comes from Mesa on the other GPU instead. If
  neither works the controller says so and runs without the window. Do not export `MUJOCO_GL=egl`
  in a shell you start `play.py` from.
- **Hugging Face:** the V-JEPA weights are fetched into the default cache (`~/.cache/huggingface`) on
  the first encode; set `HF_HOME` to move the cache, and have internet access for that first run.
- **CPU threads:** the training scripts cap torch at 4 threads; `OMP_NUM_THREADS=4` is exported by
  `run_combined.sh` and is a good default when sharing a machine.

### Trained models

The controllers load everything from a run folder (`--models-run data/episodes_v3 --tag v3`):

| what | where |
|---|---|
| Q, R, D weights | `data/episodes_v3/attempts/v3/models/{readout,reward,dynamics}.pt` |
| PCA bases and encoder settings, used online too | `data/episodes_v3/features/{pca.npz,meta.json}` |
| the environment config and rest height | `data/episodes_v3/attempts/v3/info.json` |
| frozen SAC policy (rl_true / rl_q only) | `data/rl_baseline_v1/models/sac_best.zip` (tracked, Git LFS) |
| held-out obstacle layouts for `control_clutter` | `data/obstacles_v2/{collection.json,scene_settings.json}` (tracked) |

Trained models are not in the repository: the v3 chain produces them on the cluster (`simulation/CLUSTER.md`),
and `attempts/v3` plus `features/{pca.npz,meta.json}` (about 60 MB) is all a machine needs to run the
controllers. After cloning, for the SAC:

```bash
git lfs install
git lfs pull --include="simulation/data/rl_baseline_v1/**" --exclude=""
```

Which file is read by what, and how to share a new model, is in
[simulation/world_model/README.md](simulation/world_model/README.md), section "Checkpoints".

### Run the controllers

```bash
M="--models-run data/episodes_v3 --tag v3"
python -m controller.control_pipeline --methods jepa_mpc --random-ends $M                      # 5 fresh scenes, live window
python -m controller.control_pipeline --test-episodes 20 --random-ends --headless $M           # all four: scripted rl_true rl_q jepa_mpc
python -m controller.control_pipeline --methods jepa_mpc --random-ends --obstacles 1 3 --test-episodes 20 --headless $M
python -m controller.control_clutter --methods jepa_mpc $M --only scene_0019 scene_0021        # held-out obstacle scenes
```

Always `--random-ends` (cube and goal anywhere on the table); the layout file's default is one fixed cube
spot. Frames, gifs and per-step csv land in `data/control/episodes/<method>/seed_*/`, the summary in
`results/control/summary.{txt,json,csv}`; `--out` names another run, `--resume` continues an interrupted
one. The numbers of each model version are in `STATUS.txt`.

### Train from scratch

```bash
# the exact-state SAC (about 1 h, only for rl_true / rl_q): demos -> BC -> SAC, 100-episode evaluation
python -m rl.rl_baseline --out data/rl_baseline_v1

# the world model: 400 full-mix episodes -> run folder -> encode -> Q, R, D -> checks -> closed loop
bash cluster_round.sh v3 400          # or step by step: simulation/CLUSTER.md
```

