# JEPA world model for pick and place

A Franka Panda in MuJoCo picks a cube anywhere on a table, carries it past obstacles and places it on a
marker, driven by a latent MPC on frozen V-JEPA 2 features: a readout `Q`, a reward model `R`, a dynamics
model `D`, an obstacle model `O` on the depth image with the robot and the cube removed, and an optional
learned cost-to-go `C`. No policy in the loop, no geometric avoidance rule, no route planner. A frozen SAC
on the exact state is the actor-critic baseline.

![architecture](diagrams/architecture.png)

![pipeline](diagrams/pipeline.png)

```
simulation/
  environment/        the simulator: scene, obstacles, arm controller, rewards, cameras, obstacle map   -> environment/README.md
  data_collection/    scripted and random policies, the collector, the episode format                   -> data_collection/README.md
  world_model/        run folder, V-JEPA encoding, Q R D O C training, evaluation, the model package   -> world_model/README.md
  rl/                 the exact-state SAC baseline                                                      -> rl/README.md
  controller/         the four controllers and the latent MPC (planner.py)                             -> controller/README.md
  configs/            default.yml (every knob), full_mix.yml / hack_mix.yml (collection mixes), grade_e.yml (the SAC's scene)
  models/v3c/         the packaged models the controllers run with (Git LFS)
  data/               datasets, latents, controller episodes (data/README.md; the training data is in Git LFS, the rest local)
  cluster_round.sh    the whole chain in one command
  play.py, plot_rewards.py, offscreen.py, window.py    viewer, reward plot, EGL / GLX setup helpers
diagrams/             the two figures above and the script that draws them
results/              summaries the controllers write (local)
STATUS.txt            the day-by-day log of what was measured and why each change was made
NEXT.md               what could be done next, with the commands
```

## Install

```bash
git lfs install                                                            # once per machine
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:Sarvesh-kth/world_model_project.git   # code only
cd world_model_project
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # Linux + NVIDIA GPU
pip install -r requirements.txt
git lfs pull --include="simulation/models/**,simulation/data/rl_baseline_v1/**" --exclude=""      # 50 MB: run the controllers
git lfs pull --include="simulation/data/episodes_v3/**,simulation/data/episodes_hack/**" --exclude=""   # 3.3 GB: only to retrain
cd simulation
```

Every command below runs from `simulation/` with the venv active. The encoder and the camera controllers
need Linux with an NVIDIA GPU (8 GB is enough); the V-JEPA weights (1.2 GB) download from Hugging Face on
the first run. The LFS data comes as `latents.npy.part00/01` and `episodes_raw.tar`; `cluster_round.sh`
assembles them, or by hand:

```bash
cat data/episodes_v3/features/latents.npy.part* > data/episodes_v3/features/latents.npy
tar xf data/episodes_hack/episodes_raw.tar -C data/episodes_hack
```

Rendering: the collectors, the encoder and the controllers render offscreen through EGL (`offscreen.py`
sets `MUJOCO_GL=egl` and drops a pinned EGL vendor from the shell; on a machine without EGL export
`MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`). The viewer (`play.py`, the controllers' live window) needs
a display and GLX; `window.py` finds a vendor that works. Do not export `MUJOCO_GL=egl` in a shell you
start the viewer from.

## Run

```bash
python play.py success --seed 34                                                                     # watch the scripted policy
python -m controller.control_pipeline --methods jepa_mpc --random-ends                               # the planner, 5 scenes, live window
python -m controller.control_pipeline --methods jepa_mpc --random-ends --obstacles 0 2 --test-episodes 20 --headless   # the obstacle table
python -m controller.control_pipeline --test-episodes 20 --random-ends --headless                    # all four controllers
```

Always `--random-ends` (cube and goal anywhere on the table). Summaries land in `results/<out>/summary.txt`,
frames and per-step csv in `data/<out>/episodes/`. Current numbers (20 scenes, 0-2 obstacles): the learned
obstacle model 16/20 with zero contacts, the old geometric map 14/20 with 233 contact steps; the full
table is in `controller/README.md`, the history in `STATUS.txt`.

## Train

```bash
bash cluster_round.sh v3 400          # collect 400 episodes -> run folder -> encode -> Q R D O C -> models/v3 -> checks -> 3 closed-loop runs
python -m rl.rl_baseline --out data/rl_baseline_v1       # the SAC baseline, about 1 h
```

Step by step, and how to add data to the existing run, in `world_model/README.md`; the next things to do in
`NEXT.md`.
