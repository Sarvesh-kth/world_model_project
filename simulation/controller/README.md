# Controller: the latent MPC and the actor-critic, side by side

```
planner.py            the latent MPC: waypoint candidates, imagination with D, scoring with Q, R, O (and C), CEM
control_pipeline.py   the four controllers on fresh scenes; WorldModels (online perception), episode(), summaries
control_clutter.py    the same controllers on the six held-out obstacle scenes of data/obstacles_v2
```

| method | perception | action | what it tests |
|---|---|---|---|
| scripted | exact state | the scripted policy | the upper bound of the simulator |
| rl_true | exact state | the frozen SAC | the actor-critic with perfect perception |
| rl_q | cameras through `Q` | the frozen SAC on the estimate | is the perception good enough to drive a policy |
| jepa_mpc | cameras through `Q`, `D`, `R`; the obstacle-only depth map through `O` | waypoint candidates imagined by `D`, the best one executed | does planning in the latent find the task and the way around obstacles |

```bash
python -m controller.control_pipeline --methods jepa_mpc --random-ends                                   # 5 scenes, live window
python -m controller.control_pipeline --methods jepa_mpc --random-ends --obstacles 0 2 --test-episodes 20 --headless   # the obstacle table
python -m controller.control_pipeline --methods jepa_mpc --random-ends --obstacles 0 2 --test-episodes 20 --headless --cost-to-go
python -m controller.control_pipeline --test-episodes 20 --random-ends --headless                        # all four methods
python -m controller.control_clutter --methods jepa_mpc --only scene_0019 scene_0021                      # held-out obstacle scenes
```

Always `--random-ends` (cube and goal anywhere; the layout file alone is one fixed cube spot). The models
come from `models/v3c` (`--models-run`; a run folder works with `--tag`). Frames, gifs and a per-step csv
land in `data/<out>/episodes/<method>/seed_*/`, the summary in `results/<out>/summary.{txt,json,csv}`;
`--resume` continues an interrupted run. Seeds are `--seed + 20000 + i` (20494010 upward by default).

## One decision of the planner

Every control step (180-270 ms with the GPU to itself; the V-JEPA encode of the new frames is another 250 ms):

```
  frames (16 static + 16 wrist) --V-JEPA--> z          depth + segmentation --> obstacle map (running max)
  z, p --Q--> cube xyz, held
  64 candidates: a waypoint for the grasp site + when the gripper switches + open/closed after
                 (8 fixed proposals: hold, previous plan, to the cube, grasp, above the cube, carry, lower, release; the rest sampled)
  for t in 1..8:  go-to rule -> action_t;  D(z_t, p_t, action_t) -> z_t+1, p_t+1
                  Q -> cube, held;  R -> penalties;  O(map window around the imagined hand, gripper, held) -> proximity, P(contact)
  score = stage potential at t=8 (reach 0-0.5 < held 1-2.5 < placed 3.5) + 0.1 mean potential
          + pulls toward the cube / the goal (straight line, or C's path length with --cost-to-go) + penalties - action cost
  CEM: keep 8 elites, resample, 3 rounds; the first action of the best candidate goes to the arm; the plan is the warm start
```

The go-to rule turns a waypoint into actions: travel at the carry height, descend only when aligned within
2 cm, switch the gripper within 2 cm of the waypoint, stand still while closing, never below 2.2 cm above
the table. The potential comes from `Q`'s readings through a fixed formula (`--reward-source readout`); `R`
contributes the penalties (proximity, contact, table hit, empty grasp) at `--penalty-scale` 1.

## Obstacles

`--obstacle-source model` (default): the obstacle model `O` (`world_model/train_obstacle.py`) scores every
imagined hand position against the obstacle-only depth map, the carried cube included in its training
label; no clearance formula, no route. `--cost-to-go` replaces the straight-line pulls by the learned
cost-to-go `C` so a detour is credited within the 8-step horizon (without it the hand once parked holding
the cube beside a cylinder). `--obstacle-source map` is the old geometric variant (a height map with hand
and cube masks, a clearance penalty, an A* route), `--route-proposals` a hybrid, `none` leaves `R` alone.

## Rules on top of the models, and what retires them

| rule | flag to drop it | why it exists | status |
|---|---|---|---|
| gripper envelope: an imagined hold counts only if the hand was within 1.5 cm / 2 cm of `Q`'s cube when the fingers closed | `--no-grasp-prior` | `D` imagines a hold for fingers closing mid-travel, no such closes in the data | load-bearing (1/7 without); the `travel_close` data retires it (`NEXT.md`) |
| release rule: a held cube is released only with the real hand within half the success radius of the goal, else go there closed and open on arrival; imagined "placed" needs 0.4 of the radius | `--no-release-rule` | plans opened 4-9 cm short, `Q` reads a just-released cube toward the hand | the `short_release` data retires it |
| width rules: fingers under 2 cm hold nothing; fingers commanded shut at 3.5-6.5 cm hold something | `--no-width-gate` | `Q`'s held lags a real grasp by ~12 steps (16-frame time mean) | no measurable cost (14/20 vs 15/20), can go |

## Numbers

20 random-end scenes, 0-2 obstacles, same seeds, models `models/v3c`, zero obstacle contacts unless stated:

| variant | placed |
|---|---|
| learned obstacle model (default) | 16/20 |
| learned model + `--cost-to-go` | 15/20 |
| geometric map + route (`--obstacle-source map`) | 14/20, 233 contact steps |
| `--no-width-gate` | 14/20 |

The same seeds move by about 10 % between runs of the same code (fp16 encoder, the CEM amplifies small
differences): quote 20-scene rates and treat smaller gaps as noise. The full log is `STATUS.txt`.

## The live window

Without `--headless` and with a display, a MuJoCo window in a separate process mirrors the episode
(`window.py` finds a GLX setup; the episode itself renders through EGL). `--speed` paces it.
