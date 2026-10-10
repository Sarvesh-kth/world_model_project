# Simulation: environment and data collection

Pick and place in MuJoCo with a Franka Panda. The arm picks an object at one end of the table, carries it
past obstacles and places it on the green marker at the other end. This folder holds the simulator
(`environment/`), the scripted and random policies that generate data (`data_collection/`) and the two
viewing tools. The world model, the SAC baseline and the controllers live next to it and have their own
READMEs (`world_model/`, `rl/`, `controller/`). Run every command from `simulation/`.

```
environment/
  config.py      Config (dict with attribute access) and load_config: default.yml + an override file + a dict
  scene.py       EpisodeLayout (everything that changes per episode) and build_scene (MjSpec -> MjModel)
  obstacles.py   Obstacle, the corridor sampler, add_obstacle
  control.py     ArmController: 5 numbers (dx dy dz dyaw gripper) -> joint targets through damped least squares IK
  rewards.py     Rewards: the staged, unweighted reward components
  env.py         PickPlaceEnv: reset / step / observation / render / contacts
data_collection/
  scripted_policy.py  ScriptedPickPlace (waypoint state machine), RandomPolicy, make_policy
  route.py            A* over a coarse grid around obstacle footprints, line of sight smoothing
  writer.py           EpisodeWriter: one folder per episode (data.csv, meta.json, images/), read_episode
  collect.py          collect a dataset with a mix of policies, optionally with worker processes
play.py              watch a scenario in the viewer with the reward plotted live
plot_rewards.py      plot the reward of a recorded episode
offscreen.py         imported first by every script that renders without a window: selects EGL and drops a
                     pinned EGL vendor from the shell (see the root README, "Rendering backend")
window.py            finds a GLX setup that can open a window (default vendor, then Mesa); play.py and the
                     controller's live window use it
configs/default.yml  every knob, commented
```

## The environment, `environment/env.py`

`PickPlaceEnv(cfg)` owns one MuJoCo model at a time. `reset(seed, layout=None, holdout=False)` samples an
`EpisodeLayout` (or takes the one given), builds the scene for it, puts the arm at home with a small random
posture, drops the object onto the table, lets physics settle for half a second and remembers the
object's rest height. It returns the observation and the layout summary.

`step(action, give_up=False)` runs one control step at `control.hz` (10 Hz): the controller turns the action
into a position / yaw / gripper target, then `ik.hz / control.hz` (25) IK updates each followed by the
simulator steps that fill one IK period (2 steps of 2 ms). Contacts are checked after every IK update
so a brief brush is not missed. The step returns `(obs, reward, terminated, truncated, info)`:

- `obs["proprio"]`, 20 numbers: 7 joint positions, 7 joint velocities, grasp-site xyz, finger-axis yaw,
  gripper width, gripper command (+1 open, -1 closed). This is `p` everywhere else in the project.
- `obs["goal"]`: the place position relative to the robot base. `obs["state"]`: privileged, object pose
  (7), grasp site (3), place position (3). The controllers only read it for `rl_true` and for the labels.
- `info`: `reward_components` (unweighted), `stage` (reach / grasp / transport / done), `grasped`,
  `obstacle_contact`, `table_contact`, `success`, `failed`, `object_pos`.
- `terminated` when the object has sat on the target for `episode.settle_steps` (if
  `terminate_on_success`) or on a failure: fell off the table (`fall_margin`), the policy gave up, or
  time ran out (`max_steps`, 300).

Contacts (`_check_contacts`): grasped means both finger pads touch the object while the gripper is
commanded closed; an obstacle contact is any robot or object geom touching an obstacle geom; a table
contact is any arm geom except the fingers touching the table. `obstacle_distance()` is the closest
distance between any arm collision geom and any obstacle, capped at `rewards.safe_distance` (6 cm).

Rendering: `render(camera)` for `static` (256x256, fixed above the far side of the table) and `wrist`;
`render_depth` and `point_cloud` for the optional point clouds. The renderer is created lazily and
rebuilt on every reset because the model changes.

## Scene and layout, `environment/scene.py`

`EpisodeLayout` holds the object (name, scale, mass, friction, colour), the pick and place ends and
positions and the obstacle list; `to_dict` / `from_dict` make it replayable (every recorded episode's
`meta.json` has one). `sample_layout` picks an object from the pool, a pick end (`task.pick_end`, random
by default) and the opposite place end, then tries up to 200 times for pick and place positions that are
reachable (`task.reach_range` from the base), at least `min_pick_place_dist` apart, and admit an
obstacle layout between them.

`build_scene` writes the MuJoCo spec: textures and materials, floor, table with legs and the robot
pedestal, the green place marker (no collision), the obstacles, the object mesh dropped 6 cm above the
table, the Panda from the menagerie with a stiffer gripper (`robot.grip_scale`), a grasp site 10.3 cm
below the hand frame (the point the controller moves), the wrist camera tilted 30 degrees towards the
fingertips, and the static camera. Friction uses an elliptic cone with `sim.impratio` 10, otherwise a held
object creeps out of the fingers.

## Obstacles, `environment/obstacles.py`

Three kinds, all sitting on the table: `wall` (thin box across the corridor), `box` (small square box with a
random yaw) and `cylinder`. `sample_obstacles` places `task.obstacles.count` of them along the corridor
between pick and place (`corridor_span` fraction along, `lateral_offset` across), at least `clearance`
from the ends and `spacing` from each other, on the table. The first one is always tall
(`kinds.*.height`, 16 to 19 cm, taller than the arm can carry over); the rest get a height from
`extra_height` (4 to 19 cm), so some can be carried over. `footprint_radius` is the circle the route
planner keeps away from. The config `full_mix.yml` uses 1 to 3 obstacles for the world model data; the
controllers' standard layout `grade_e_layout.json` has none.

## Arm controller, `environment/control.py`

Actions are 5 numbers in [-1, 1]. `set_action` moves the position target by `action[:3] * max_delta`
(4 cm per step), clipped to the `workspace` box, adds `action[3] * yaw.max_delta` to the hand yaw (kept
within a quarter turn, the gripper is symmetric) and opens or closes the gripper by the sign of
`action[4]`. Each `update_ctrl` slides the target from the step's start pose towards the commanded one,
computes position and orientation error of the grasp site, a damped least squares inverse of the site
Jacobian (`ik.damping`), a null space pull towards the posture (`posture_gain`), caps linear, angular and
joint velocities, integrates into the joint target and never lets that target run more than `max_lead`
ahead of the real joints. The orientation is always top down, rotated by the yaw.

## Rewards, `environment/rewards.py`

`Rewards.compute(state, action)` returns the weighted total, the unweighted components and the success
flag. Components (weights in `rewards.weights`):

| component | when | value |
|---|---|---|
| reach | object not yet grasped | `1 - tanh(5 * distance hand to object)` |
| grasp | object grasped | 1 |
| lift | grasped | height above the table / `lift_height` (4 cm), clipped to [0, 1] |
| transport | grasped | `1 - tanh(3 * distance object to target)` |
| place | once | object released, resting within `success_radius` (7 cm) of the target after a grasp |
| success | once | the first step the placed object sits at the target |
| fail | once | episode ends without success |
| collision, table_hit | per step | -1 while touching an obstacle / the table |
| proximity | per step | ramps from 0 at `safe_distance` to -1 when an arm link touches an obstacle |
| drop | once | released away from the target |
| action | per step | mean squared motion command |

The components are stored unweighted on purpose so training can pick any subset later; the penalty
head `R` in `world_model/` learns proximity, collision and table_hit. The SAC in `rl/` uses its own
shaping built on top of this class (`GoalReward`).

## Data collection, `data_collection/`

`collect.py` collects a dataset with a mix of policy kinds set in `data.mix`:

| kind | policy |
|---|---|
| success | clean scripted pick and place, retried on fresh layouts up to `success_attempts` times until it succeeds |
| scripted | the same with slowly drifting action noise (`scripted.noise`, `noise_correlation`) |
| drop | noisy scripted that lets go halfway through the carry |
| collide | noisy scripted that carries low, straight through the obstacles |
| random | random actions with momentum and a downward drift |

`allocate_kinds` turns the fractions into a shuffled list, one kind per episode. With `--workers N` each
worker owns an env and a writer; only the parent appends to `index.csv`. The scripted demo succeeds on
about 97 % of layouts, the rest are layouts where tall obstacles leave no gap for the hand.

```bash
python -m data_collection.collect --episodes 100 --out data/test --workers 4
python -m data_collection.collect --config configs/full_mix.yml --episodes 120 --out data/episodes_test1/episodes_raw
```

The scripted policy (`scripted_policy.py`) is a waypoint state machine: approach (around tall obstacles,
over low ones, planned by `route.py`), hover 12 cm above the object, turn the fingers square to the
object's long axis, descend, close and hold 8 steps, lift to the carry height, transport along the planned
route rising over low obstacles, lower, release, retreat. It retries a missed grasp twice and gives up
when the hand has not moved 3 cm in 30 steps. `route.py` runs A* on a 2 cm grid where cells inside an
obstacle circle (footprint plus `route_margin`) cost 1000 instead of 1, so a start inside a circle still
finds a way out, then straightens the path by line of sight.

### What a recorded episode looks like

```
data/<run>/
  index.csv                one row per episode (policy, object, ends, steps, success, collisions, seed)
  episode_000000/
    data.csv               one row per control step: action, the 20 proprio numbers, object pose, place and
                           goal, reward_total and every reward_<component>, grasped / contact / success flags,
                           stage, camera poses
    meta.json              layout (replayable), policy kind and its parameters, seed, goal, the config
    images/static_1.jpg    static camera, one per row (serial), and wrist_1.jpg
    pcl/static_1.npy       point clouds, only with --pointcloud
```

`data_collection.writer.read_episode(folder)` loads a folder back. `world_model/prepare_episodes.py` turns a
dataset like this into the run format the world model trains on.

## Watching and plotting

```bash
python play.py success --seed 34          # viewer window plus a live reward plot (Linux)
python play.py collide --speed 2
python play.py success --layout data/test/episode_000003/meta.json   # replay a recorded layout
python plot_rewards.py data/test/episode_000003 --save                # rewards.png into the folder
```

On macOS the viewer needs `mjpython` and the live plot is disabled. The viewer window needs a display;
every collector and controller renders offscreen through EGL instead.

## Config

`configs/default.yml` is the base; another yml (`grade_e.yml`, `full_mix.yml`) overrides only the keys it
lists, and code can pass a dict of overrides on top. The important sections: `sim` (timestep, friction
impratio), `control` (hz, max_delta, yaw, workspace, the IK gains), `cameras`, `robot` (base, home posture,
posture noise, grip scale), `table`, `task` (ends, reach range, object pool and ranges, obstacles),
`episode` (max_steps, success radius, settle steps, fall margin, terminate_on_success), `rewards`
(lift height, safe distance, weights), `data` (jpeg quality, workers, the policy mix, scripted policy
parameters). `grade_e.yml` is the fixed-scene setup the SAC and the world model use: no posture noise,
cube only.
