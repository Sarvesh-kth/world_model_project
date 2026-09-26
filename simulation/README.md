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

## Setup and run

See the [root README](../README.md) for installation and the separate macOS and
Linux viewer commands. Run the commands below from `simulation/`.

The viewer opens a window and needs a graphical desktop. On Linux a second
window plots reward live. On macOS `mjpython` is required for the viewer, so the
live Matplotlib window is disabled; reward events and the final component
totals are printed in the terminal. Add `--seed 34` for a repeatable layout,
`--speed 2` for faster playback, or
`--layout data/smoke/episode_000003/meta.json` to replay a recorded layout.

An episode ends when the object has sat on the target for a bit (success), or as a failure
when the object falls off the table, the scripted policy gives up (grasp missed too many
times, arm stuck), or time runs out. Failures get a one time `fail` penalty in the reward.

Collect a dataset:

```bash
../.venv/bin/python -m data_collection.collect --episodes 100 --out data/test --workers 4
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
../.venv/bin/python plot_rewards.py data/test/episode_000003          # opens a window
../.venv/bin/python plot_rewards.py data/test/episode_000003 --save   # writes rewards.png into the folder
```
