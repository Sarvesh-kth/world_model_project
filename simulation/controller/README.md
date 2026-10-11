# Controller: the actor-critic and the latent MPC, side by side

```
planner.py            the latent MPC: Cartesian waypoint candidates, imagination with D, scoring with Q and R, CEM
control_pipeline.py   the four controllers on fresh scenes of one layout; WorldModels, Policy, episode(), summaries
control_clutter.py    the same controllers on the six held-out obstacle scenes (data/obstacles_v2 layouts)
```

| method | perception | action | what it tests |
|---|---|---|---|
| scripted | exact state | the scripted policy | the upper bound of the simulator |
| rl_true | exact state | the frozen SAC | what the actor-critic can do with perfect perception |
| rl_q | cameras through `Q(z, p)` (+ height map) | the frozen SAC on the estimate | is the JEPA perception good enough to drive the policy |
| jepa_mpc | cameras through `Q`, `D`, `R` | waypoint candidates imagined by `D`, best one executed | does planning in the latent find the task and the way around obstacles |

```bash
python -m controller.control_pipeline --methods jepa_mpc                   # 5 scenes, live window
python -m controller.control_pipeline --test-episodes 20 --headless        # all four, the benchmark
python -m controller.control_pipeline --methods jepa_mpc --obstacles 1 3   # fresh obstacle layouts per scene
python -m controller.control_pipeline --methods jepa_mpc --obstacles 1 3 --random-ends --test-episodes 20   # cube and goal at random table ends too
python -m controller.control_clutter --methods rl_q jepa_mpc               # six held-out obstacle scenes
python -m controller.control_clutter --methods jepa_mpc --only scene_0019 scene_0021
```
Add `--models-run data/episodes_v3 --tag v3` to point at the v3 models; `cluster_round.sh` runs the three
evaluations (20 random-end empty, 20 random-end obstacle, 6 clutter scenes) after training.

## Two controllers, two ways to use one world

Both controllers get the same perception, the JEPA latent read by `Q` (and the height map). They differ in
where the computation happens:

- **The actor-critic** (`rl_true` / `rl_q`) samples trajectories only while it trains, in the simulator.
  At test time it is one forward pass of a 128-128 network, about 1 ms per step, and it never imagines
  anything. It knows what its training distribution taught it: the fixed layout, and with `--obstacles`
  (`rl/README.md`) random corridor obstacles seen through the height map.
- **The MPC** (`jepa_mpc`) trains no policy at all. Every step it imagines candidate futures with `D`,
  scores them with `Q` and `R`, and executes the best first action; about 0.3 s per step on an RTX 4060,
  most of it the online encoder. It adapts to whatever the latent shows, including layouts no policy was
  trained on, at the price of the decision time and of trusting `D` over the horizon.

So the comparison is not "two samplers": it is an amortised reactive policy against online planning on
the same perception, judged on placements, contacts, decision time and held-out obstacle layouts. The
SAC is not part of the planner any more (the earlier CEM searched around the SAC's own action sequence,
which made "the planner beats the policy" circular); `--baseline-run` is only loaded for `rl_true` / `rl_q`.

## What runs, `control_pipeline.py`

`main()` loads everything once: the config and resting cube height of `--models-run` (`attempts/<tag>/info.json`),
the layout file, the SAC (`Policy`, only for `rl_true` / `rl_q`; it reads from `pipeline.json` whether the
policy expects the height map) and, for the camera methods, `WorldModels`. Then for every method and
every seed (`--seed + 20000 + i`, default 20494010 upward) it calls `episode()` and refreshes the
summary. `--obstacles LOW HIGH` draws fresh corridor obstacles per seed with the environment's sampler;
`--random-ends` also redraws the pick and place positions at random table ends (the planner's generalisation
test; the SAC only knows the file's layout).
Without `--resume` the output folders are wiped first.

**WorldModels(root, tag, args)** holds the online perception: the `Encoder` from `world_model/encode.py`
built with the settings in `features/meta.json` and the bases in `features/pca.npz` (so online `z` is
exactly the cached `z`), `Q` (`readout.pt`, or `readout_p.pt` with `--blind`), `D` (`dynamics.pt`) and `R`
(`reward.pt` if present and not `--no-penalties`).

- `encode(histories, heightmap)`: the latent of both frame histories (newest last) and the live height map
  (`PickPlaceEnv.height_map`, the static depth image).
- `read(z, p)`: `Q`, cube xyz and held probability. The **width gate** then zeroes held when the measured
  finger width is under 2 cm: fingers closed on nothing hold nothing (the cube is 4.5 cm). Without it `Q`
  kept reporting held after a missed grasp and the controller carried air to B in every failed episode.
  `--grasp-prior` adds the old stop-gap on top (held only with the cube estimate between the fingers).
- `predict(z, p, actions)`: `D` one step for a batch of candidates. `D`, `Q` and `R` sit on the GPU next
  to the encoder (5x faster than the CPU on the planner's batches of 64 candidates).
- `penalty(z, p, weights)`: `R`'s weighted penalty, `penalty_scale * (0.3 proximity - 0.5 P(collision) - 0.2 P(table) - 0.5 P(empty grasp))`, <= 0.
  The last term is what stops the planner from closing the fingers in the air: an imagined close away from
  the cube now costs score every step instead of being imagined as a hold.

**episode(root, method, seed, cfg, layout, rest_z, args, policy, models)** is one episode of one
controller, the same loop for all four:

1. `TaskSession.reset(seed)` (cube jittered by `--position-jitter`, 1 cm), the live window if there is a
   display, both first frames saved and (camera methods) encoded and read.
2. Each step picks the action: `scripted` from the state machine, `rl_true` from the SAC on the exact
   observation, `rl_q` from the SAC on `policy_observation(p, Q's xyz, Q's held, memory, ...)`, `jepa_mpc`
   from `planner.plan()`.
3. The environment steps, the new frames and height map are encoded, `Q` reads them, the task memory is
   advanced from the estimate (never from the true state), and the step is logged.
4. Outputs under `data/<out>/episodes/<method>/seed_<seed>/`: `frames/{static,wrist}_*.jpg`, `actual.gif`,
   `steps.csv` (per step: true and control reward, goal distance, lift, actions, the actual cube position,
   `Q_x Q_y Q_z Q_held_probability Q_xyz_mae_cm`, and for the planner the chosen proposal, its score,
   predicted potential and penalty, the waypoint, the score of every fixed proposal, one-step error of `D`),
   `candidates.csv` (every candidate's waypoint, switch step, gripper, score, terminal potential and penalty
   per step), `trajectory.npz`, `result.json` (the session result plus `Q_on_actual_observations`,
   `planner_fallback_steps`, `mean_decision_seconds`).

`summarize()` writes `results/<out>/summary.{json,csv,txt}` (placements, mean and median final distance,
contacts, decision time per method) and copies every episode's json and csv there; neither `results/`
nor `data/` is tracked.

## The planner, `planner.py`

Everything is in Cartesian space of the grasp site: the environment's actions already are `(dx, dy, dz,
dyaw, gripper)` deltas, so a plan is a place to go and a moment to close or open the hand, not a joint
trajectory. Every control step `plan()`:

1. **Candidates**: a waypoint `w` (xyz, clipped to the workspace), the step `k` at which the gripper
   command switches and the command after it (`+1` open / `-1` closed). The action at imagined step `t`
   is the go-to rule `clip((w - hand_t) / max_delta, -1, 1)` with `hand_t` taken from `D`'s imagined
   robot state, yaw 0, gripper per the switch. Eight fixed proposals are always in the population and
   scored like the rest: hold, the previous plan, go to the cube, go to the cube and close on arrival,
   above the cube, carry to the goal at `--carry-height`, lower at the goal, release at the goal. The
   other `--population - 8` candidates are sampled: `w ~ N(mean, std)` with the previous plan's waypoint as
   the first mean and `--waypoint-std` (15 cm in x / y, 6 cm in z), a switch somewhere in the horizon with
   probability `p_switch`, closed after it with probability `p_close`.
2. **Imagination** (`forecasts`): `D` rolls every candidate `--horizon` (8) steps, `Q` reads each imagined
   state (cube, held through the width gate), `R` its reward terms (reach, grasp, lift, transport,
   placed, proximity, collision, table hit, empty grasp), and the task potential of each imagined state is computed. A candidate is invalid when its imagination leaves the physical range (finger width
   outside [-0.2, 9] cm, hand or cube more than 5 m away, nonfinite numbers).
3. **Score** = potential at the end of the horizon + 0.1 x the mean potential along it + the summed
   penalties - `--action-cost` x the squared motion. Wide waypoints make these differ by whole units, so
   `D`'s noise (about 0.01) does not decide. The potential carries a linear pull toward the cube (0.1 over
   60 cm) while not held and toward the goal (0.3 over 80 cm) while held, because the task's tanh terms are
   flat beyond 30 cm: without the pull, 8 steps of carry 50 cm from the goal gained 0.05 while one imagined
   step near an obstacle cost 0.6, and the planner retreated instead of going around. The gripper closes
   when the imagined hand is within 2 cm of the waypoint, the measured grasp tolerance (closes within 1.5 cm
   hold 92-100 %, 1.5-2.5 cm about half).
4. **CEM**: the `--elites` (8) best refit the waypoint mean and std (floored at a fifth of the start),
   `p_switch` and `p_close`; `--iterations` (3) rounds; the best candidate over all rounds wins. Its first
   action is executed and its waypoint, switch step minus one and gripper are the warm start of the next
   step. If the best candidate is invalid the arm holds still (a fallback, counted).

**The potential** is what the latent is asked to deliver: a number that only goes up as the task
progresses. `--reward-source` picks where it comes from: `readout` builds it from `Q`'s cube and held
readings through the formula below (`planner.potential`), `head` from the reward model's predicted terms
(`planner.learned_potential`: `0.5 reach` while not grasped, `1 + 0.5 lift + transport` while grasped,
`3.5` when placed, the stages blended by the predicted grasp and placed probabilities), `both` (default)
averages the two. The penalties always come from `R`.

| stage | condition (from `Q` and the task memory) | potential |
|---|---|---|
| reaching | not held | 0.5 (1 - tanh 5 \|hand - cube\|) in 0 .. 0.5 |
| held | held | 1 + 0.5 lift / 4 cm + (1 - tanh 3 \|cube - goal\|) in 1 .. 2.5 |
| placed | lifted before, now released within the success radius at rest height | 3.5 |

The stages are strictly ordered (closing on the cube beats hovering at it, releasing at the goal beats
holding it there), so the planner never has to be told what to do next; it only has to predict it. The
penalties from `R` are the obstacle awareness: a wall across the corridor costs up to `penalty_scale x
(0.5 + 0.3)` per imagined step, more than any potential gain over the horizon, and the sampled waypoints
to the side are how the detour is found.

| flag | default | meaning |
|---|---|---|
| `--horizon` | 8 | steps `D` imagines; `D` trains on 8-step windows and beyond them it imagined a carried cube slipping away, which made every carry score worse than holding still |
| `--population` / `--elites` / `--iterations` | 64 / 8 / 3 | CEM size; the first 8 of the population are the fixed proposals |
| `--waypoint-std`, `--waypoint-std-z` | 0.15, 0.06 | initial spread of the sampled waypoints, metres |
| `--carry-height` | 0.08 | height of the carry proposal above the cube's rest height |
| `--action-cost` | 0.0002 | cost per unit squared motion over the horizon (keeps the arm still once placed; at 0.01 it outweighed the reach gain and the arm crept) |
| `--penalty-scale` | 2 | multiplier on `R`'s penalties; 1 = the recorded weights |
| `--reward-source` | readout | task potential from `Q` through the formula (default), from `R`'s terms (`head`), or their mean (`both`); `R`'s placed flag fires on imagined states with the hand high and the gripper opening (the post-placement retreat in the data), which pulled the planner toward the ceiling |
| `--plan-margin` | 0.05 | a fresh candidate replaces the current plan only if it scores this much higher (commitment; halves the direction flips) |
| `--waypoint-std-min` | 0.03 | the sampling spread shrinks with the distance to the cube (or the goal when holding it) down to this |
| `--no-penalties` | | ignore `R` entirely (potential from `Q` only) |
| `--no-width-gate` | | keep `Q`'s held reading when the fingers are closed on nothing |
| `--grasp-prior` | off | old stop-gap: an imagined hold counts only if the hand was at `Q`'s cube estimate when it closed; D and R trained on the miss data are meant to make it unnecessary |
| `--cube-anchor` | off | old stop-gap: aim at the last clear sighting of the cube while the hand is within 8 cm of it (Q used to drift under the hand) |
| `--blind` | | `readout_p` instead of `Q(z, p)`: the loop without the image |
| `--headless`, `--speed` | | no window / playback speed of the window |

## Obstacle scenes, `control_clutter.py`

Same `episode()`, but the layout comes from the `--scenes-run` (`data/obstacles_v2`) `scene_settings`:
the test scenes of that collection (`collection.json` `groups`), each with the obstacles of its
`<scene>_replay_obstacles` rollout (`--layout-suffix`). `--only scene_0019 scene_0021` picks scenes, the
cube is not jittered. The planner sees the obstacles through `R` and the height map; the SAC only if it
was trained with `--obstacles`. Writes `scenes.json` per scene next to the usual summary.

## Numbers

The closed-loop numbers of the current models are in `STATUS.txt` at the repository root (section
"v2"), with the commands that produced them. Outcomes vary between runs of the same seed: the encoder
runs in fp16 and the CEM amplifies small differences in `Q`, so quote rates over 20 seeds, not single
episodes.

## The live window

With a display and without `--headless`, `LiveViewer` opens a MuJoCo window in a separate process that
rebuilds the same scene and mirrors the joint positions it is sent (the episode process renders its
cameras through EGL and a window cannot share that context). `--speed` paces the playback but cannot beat
the planner's compute (about 0.3 s per step on an RTX 4060).
