# JEPA world model for pick and place

A Franka Panda in MuJoCo picks a cube from anywhere on a table, carries it past 0-3 obstacles and puts it
down on a green marker. The controller is a model predictive controller (MPC) that plans in the latent space
of a frozen V-JEPA 2 encoder: a readout `Q` says where the cube is and whether it is held, a dynamics model
`D` imagines the latent after an action, a reward model `R` predicts the penalties, an obstacle model `O`
reads an obstacle-only depth map, and an optional cost-to-go `C` knows the way around the obstacles. No
policy is in the loop and no geometric avoidance rule. The comparison is a SAC policy trained in the
simulator on the exact state, run on the exact state as `rl_true` or on the cameras through `Q` as `rl_q`.

![architecture](diagrams/architecture.png)

![pipeline](diagrams/pipeline.png)

## Repository structure

```
simulation/             the code; every command runs from here with the venv active
  environment/          the MuJoCo simulator: scene, obstacles, arm control, rewards, cameras
  data_collection/      scripted and random policies that record episodes
  world_model/          run folder, V-JEPA encoding, training and checks of Q R D O C
  rl/                   the SAC policies and the simulator wrapper every controller runs in
  controller/           the controllers and the two latent planners
  configs/              default.yml (every knob) and the override files and fixed scenes
  models/               the packaged models the controllers load: world_model/ and sac/ (Git LFS)
  data/                 run folders and controller episodes (local, except the held-out scenes)
results/                the controllers' summaries
diagrams/               the two figures above and the script that draws them
presentation/           the notes for the slides, the clips and the results tables
train.sh, evaluate.sh   the whole training chain, and the controllers against each other
```

Each folder has a README. Runs are named by what they are: `data/test_<count>obstacles_<controller>` for
the evaluation script, `data/episodes` for the training data, `models/world_model` and `models/sac` for
the packages the controllers load.

## Install

```bash
git lfs install                                                            # once per machine
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:Sarvesh-kth/world_model_project.git   # code only
cd world_model_project
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # Linux + NVIDIA GPU
pip install -r requirements.txt
git lfs pull --include="simulation/models/**" --exclude=""                 # the packaged models the controllers run with
cd simulation
```

The encoder and the camera controllers need Linux with an NVIDIA GPU; the V-JEPA weights download from
Hugging Face on the first run. `train.sh` collects the training data.

Rendering: collectors, encoder and controllers render offscreen through EGL (`offscreen.py` sets it up). On
a machine without EGL install `libosmesa6` and export `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`. The viewer (`play.py`, the controllers' live window) needs a display and GLX;
do not export `MUJOCO_GL=egl` in a shell you start it from.

## Run

Everything runs from `simulation/` with the venv active, except the two scripts at the root, which run
from here.

### Watch the scripted policy

```bash
cd simulation
python play.py success --seed 34      # a clean pick, around the obstacles, place
python play.py fail --seed 34         # picks, then drops the cube mid carry
python play.py collide --seed 34      # picks, then carries straight into the obstacles
python play.py random --seed 34       # random actions
```

A MuJoCo window plus a live reward plot; every bonus and penalty is printed as it happens. `--speed 2`
runs it twice as fast, `--episodes 3` plays three scenes, `--layout <episode>/meta.json` replays a
recorded scene.

### Test a controller

```bash
python -m controller.control_pipeline --methods mpc --random-ends --obstacles 2 2 --test-episodes 5 --headless --out data/mpc_2
python -m controller.control_pipeline --methods rl_true rl_q --random-ends --obstacles 0 3 --test-episodes 10 --headless --out data/sac_03
python -m controller.control_pipeline --methods mpc --random-ends --test-episodes 3 --out data/look     # live window
python -m controller.control_pipeline --methods mpc rl_true --layout configs/middle_wall_layout.json --test-episodes 3 --headless --out data/wall
```

`--methods` picks the controller, `--obstacles LOW HIGH` how many obstacles a scene gets, `--test-episodes`
how many scenes; `--random-ends` puts the cube and the goal anywhere on the table (the distribution the
models were trained on), `--layout` runs one fixed scene instead. Summaries land in
`results/<out>/summary.txt`, frames, gifs and a per-step csv in `data/<out>/episodes/<method>/seed_*/`;
`--resume` continues an interrupted run.

The scenes are seeded: episode `i` is drawn from `--seed + 20000 + i`, and `--seed` has a fixed default, so
the same command gives the same cube, goal and obstacles every time and every controller is compared on
the same scenes. For a fresh scene pass another seed (`--seed $RANDOM`) or ask for more episodes, each
gets its own.

| method | perception | action |
|---|---|---|
| `scripted` | exact state | the scripted policy, the simulator's upper bound |
| `rl_true` | exact state | the simulator-trained SAC (`models/sac`) |
| `rl_q` | cameras through `Q` | the same SAC on the estimate |
| `mpc` | cameras through `Q`, `D`, `R`, `O` | CEM over raw action sequences imagined by `D` (`controller/mpc.py`) |
| `jepa_mpc` | cameras through `Q`, `D`, `R`, `O` (and `C`) | CEM over waypoints, a go-to rule makes the actions (`controller/planner.py`) |

### Evaluate controllers against each other

```bash
bash evaluate.sh                              # mpc, rl_q and rl_true: 10 scenes on 0 / 1 / 2 / 3 obstacles, same seeds, side by side
bash evaluate.sh "mpc rl_true" 20             # these two, 20 scenes per count
COUNTS="0 3" bash evaluate.sh mpc 5           # only these obstacle counts
python simulation/results_table.py            # the table, also printed at the end of the script
```

The controllers run in parallel, every one over the obstacle counts in turn, into
`data/test_<count>obstacles_<controller>/`; finished episodes are kept, so the script can be run again to continue.
`MODELS=models/new SAC=models/sac_new` evaluates other packages.

## Train

```bash
bash train.sh                                  # 400 episodes -> encode -> Q, then R D O C and the SAC side by side -> checks -> models/world_model, models/sac
RUN=data/episodes_new TAG=new EPISODES=600 bash train.sh   # another run; EPOCHS, SAC_STEPS, DEMOS, WORKERS can be set the same way
```

Encoding is the slow step; the SAC trains on the CPU while the other models train on the GPU. Every
step is skipped when its output exists, so a stopped run continues where it was. Step by step, and how to add
data to an existing run, in `simulation/world_model/README.md` and `simulation/rl/README.md`.
