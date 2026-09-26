# JEPA World Model Pick and Place

This repository currently contains a MuJoCo simulation of a Franka Panda arm,
scripted pick-and-place demonstrations, and episode data collection. The JEPA
encoder, learned world model, and learned controllers are planned work; the
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
