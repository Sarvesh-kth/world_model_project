# Environment

Pick and place in MuJoCo with a Franka Panda: the arm picks a cube anywhere on the table, carries it past
0 to 3 obstacles and places it on the green marker. Every command in this repository runs from `simulation/`.

```
env.py         PickPlaceEnv: reset / step / observation / render / contacts / height_map / obstacle_map
scene.py       EpisodeLayout (everything that changes per episode) and build_scene (MjSpec -> MjModel)
obstacles.py   Obstacle, the corridor sampler, add_obstacle
control.py     ArmController: 5 numbers (dx dy dz dyaw gripper) -> joint targets through damped least squares IK
rewards.py     Rewards: the staged, unweighted reward components
config.py      Config (dict with attribute access) and load_config: configs/default.yml + an override file + a dict
```

## Step and observation

`reset(seed, layout=None)` samples an `EpisodeLayout` (cube size, mass, friction, pick and place positions,
obstacles) or takes the one given, builds the scene, puts the arm at home with a small random posture and
lets the cube settle. `step(action)` runs one control step at 10 Hz: 25 IK updates, each followed by the
simulator steps of one IK period, contacts checked after every update. Actions are 5 numbers in [-1, 1]:
`dx dy dz` move the grasp-site target by up to 4 cm, `dyaw` turns the hand, the sign of the last number
opens or closes the gripper.

- `obs["proprio"]` (20 numbers, `p` everywhere else): 7 joint positions, 7 joint velocities, grasp-site xyz,
  finger-axis yaw, gripper width, gripper command (+1 open, -1 closed).
- `obs["state"]`: privileged, cube pose, grasp site, place position. Read only for `rl_true` and for labels.
- `info`: unweighted `reward_components`, `stage`, `grasped`, `obstacle_contact`, `table_contact`, `success`, `failed`.
- The episode ends when the cube has sat at the target for 15 steps, fell off the table, or after 300 steps.

Grasped = both finger pads touch the cube while the gripper is commanded closed. An obstacle contact = any
robot geom OR the cube touching an obstacle. `obstacle_distance()` = the closest distance between any arm
collision geom and any obstacle, capped at 6 cm.

## What the cameras give

| call | output | used by |
|---|---|---|
| `render("static")`, `render("wrist")` | 256x256 RGB; the static camera looks down at about 60 degrees, the wrist camera is tilted 30 degrees to the fingertips | V-JEPA |
| `height_map()` | the static depth image binned to 8x8 cells of the table, highest point per cell, clipped at 25 cm; arm, cube and obstacles all in it | inside `z`, the obstacle-aware SAC |
| `obstacle_map()` | the same depth image rendered together with a segmentation image, every pixel of a robot geom or of the cube dropped, 16x16 cells (4 x 5.75 cm), a cell needs 8 points (stray edge pixels made phantoms) | the obstacle model and the cost-to-go |

## Scene

Three obstacle kinds stand on the table between pick and place: `wall` (thin box across the corridor), `box`
(small, random yaw), `cylinder`. The first is always 16-19 cm tall (too tall to carry over), the rest 4-19 cm.
Pick and place are anywhere on the reachable table (`task.pick_end: anywhere`) at least 35 cm apart, the cube
is grey with a random size, mass, friction and yaw.

## Rewards

`Rewards.compute` returns the weighted total, the unweighted components and the success flag:

| component | value |
|---|---|
| reach | `1 - tanh(5 * hand to cube)` while not grasped |
| grasp, lift, transport | 1 while grasped; height / 4 cm; `1 - tanh(3 * cube to target)` |
| place, success, fail, drop | once: released within 7 cm of the target after a lift; first settled step; episode lost; released elsewhere |
| collision, table_hit | -1 per step while touching an obstacle / the table |
| proximity | 0 at 6 cm from any arm link, down to -1 at contact |
| empty_grasp | -1 per step while commanded closed with the fingers under 3 cm apart (the cube is 4.5 cm) |
| action | mean squared motion command |

The components are stored unweighted so training can pick any subset; the reward model `R` learns the
four penalties, the SAC in `rl/` builds its shaping on top (`GoalReward`).

## Config

`configs/default.yml` is the base (every knob, commented); another yml overrides only the keys it lists.
`full_mix.yml` and `hack_mix.yml` are collection mixes (`../data_collection/README.md`), `grade_e.yml` plus
`grade_e_layout.json` the fixed scene the SAC was trained on.
