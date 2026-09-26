# SIMULATION MUJOCO PICK AND PLACE

Pick and place in MuJoCo with a Franka Panda, used to collect data for a JEPA style world model.
The arm picks a random object from one end of the table, carries it past three to five obstacles
and places it on the green marker at the other end. At least one obstacle is too tall to lift
over, so the route goes around it; the low ones get carried over, with the carry height rising
and falling along the way. The scripted demo succeeds on about 97% of layouts (117 of 120 seeds),
the rest are layouts where tall obstacles leave no gap for the hand.

```
simulation/       the MuJoCo env and the data collection
  environment/    scene builder, obstacles, controller, rewards, env, config loader
  data_collection/  scripted / random policies, route planner, episode writer, collector
  configs/        default.yml, every knob lives here
  play.py         watch a scenario in the viewer
  plot_rewards.py plot the reward of a recorded episode over time
controller/       (todo) the planner that uses the world model
jepa_model/       (todo) the world model itself
```

## Setup

Enter the repository

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Everything runs from inside `simulation/`, with the venv active and headless rendering on:

```bash
cd simulation
source ../.venv/bin/activate
export MUJOCO_GL=egl          # osmesa if there is no GPU
```

Watch it (opens a window, needs a display):

```bash
python play.py success        # pick, move between the obstacles, place
python play.py fail           # drops the object mid way
python play.py collide        # carries straight through the obstacles
python play.py random         # random actions
```

A second window plots the reward live while it runs (reward per step and the running return,
then the bonus terms, then the penalties), and every bonus or penalty that fires is printed
in the terminal with the time and the stage. Add `--seed 34` to pick a layout, `--speed 2`
to play faster, `--layout data/smoke/episode_000003/meta.json` to replay a recorded
episode's layout.

An episode ends when the object has sat on the target for a bit (success), or as a failure
when the object falls off the table, the scripted policy gives up (grasp missed too many
times, arm stuck), or time runs out. Failures get a one time `fail` penalty in the reward.

Collect a dataset:

```bash
python -m data_collection.collect --episodes 100 --data/test --workers 4
```


## Data Saving

```
data/<run>/
  index.csv                one row per episode (policy, object, ends, success, collisions)
  episode_000000/
    data.csv               one row per step, keyed by serial
    meta.json              layout, policy variant, seed, goal, config
    images/static_1.jpg    static camera, one per serial
    images/wrist_1.jpg     wrist camera
    pcl/static_1.npy       point clouds, only with --pointcloud
```

Every row has the action, joint state, object pose, every reward component unweighted,
the grasped / collision flags and the camera poses. `data_collection.read_episode(folder)`
loads a folder back.

Reward over time for one episode (total on top, the components below):

```bash
python plot_rewards.py data/smoke/episode_000003          # opens a window
python plot_rewards.py data/smoke/episode_000003 --save   # writes rewards.png into the folder
```

