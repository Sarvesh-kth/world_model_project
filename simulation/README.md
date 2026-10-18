# simulation

Everything runs from this folder with the venv active (`cd simulation && source ../.venv/bin/activate`).
The packages are the pipeline in order, each with its own README:

```
environment/       the MuJoCo simulator: scene, obstacles, arm controller, rewards, cameras and maps
data_collection/   scripted and random policies that record episodes for the world model
world_model/       run folder, V-JEPA encoding, the models Q R D O C, their evaluation
rl/                the SAC policies: the fixed-layout one and the simulator-trained one for the MPC's scenes
controller/        the controllers (scripted, rl_true, rl_q, jepa_mpc, mpc) run on fresh scenes and compared
configs/           default.yml and the override files
models/            the packaged models the controllers load (Git LFS)
data/              run folders and controller episodes (local, except what the data README lists)
```

Scripts at this level:

```
play.py            watch the scripted policy in a MuJoCo window with a live reward plot:  python play.py success --seed 34
plot_rewards.py    plot the rewards of a recorded episode
results_table.py   the markdown table of the controllers per obstacle count, from the runs of ../evaluate.sh
offscreen.py       imported first by everything that renders without a window (selects EGL)
window.py          finds an OpenGL setup that can open a window (the viewer, the controllers' live window)
```

Rendering: the collectors, the encoder and the controllers render offscreen through EGL. On a machine
without EGL install `libosmesa6` and export `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`. The viewer needs a display and GLX; do not export `MUJOCO_GL=egl` in a shell you start it from.
