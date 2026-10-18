# Results

Every controller run on the same fresh scenes, what each controller does, how it fails and what is next.
Commands run from `simulation/`; the raw summaries are in `results/<run>/`.

## The test

Cube and goal anywhere on the table at least 35 cm apart, the obstacles drawn between them, 0 / 1 / 2 / 3
obstacles, 10 scenes per count (seeds 20494010-19), the same scenes for every controller.
An episode ends with the cube resting within 7 cm of the goal for 15 steps, after 300 steps, or when the
cube falls off the table. "3 obstacles" means up to three: two of those scenes fitted only one or two.

- `mpc` (`controller/mpc.py`): every control step it encodes the last 16 frames of both cameras, reads the
  cube with `Q`, imagines 64 action sequences of 8 steps with `D`, scores them with the task reward on `Q`'s
  readings plus `R`'s and `O`'s penalties, refits three times (CEM) and executes the first action of the best.
  Its models saw only the 401 recorded episodes; it never got a reward from the simulator.
- `rl_true` and `rl_q` are one policy, `models/sac` (`rl/README.md`): a SAC trained in the simulator with
  the simulator's reward (placement +15, penalties for contact, proximity, table hit, drop and failure) from
  500 scripted demonstrations, the same 401 recorded episodes in its replay buffer, behaviour cloning and
  20,000 RL steps. `rl_true` gets the exact cube and contact flag, `rl_q` gets `Q`'s camera reading.

## Outcomes

| controller | 0 obstacles | 1 | 2 | 3 | all | contact steps | median error when placed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mpc` | 7/10 | 7/10 | 7/10 | 5/10 | 26/40 | 0 | 1.9 / 3.2 / 2.4 / 2.2 cm |
| `rl_q` | 5/10 | 4/10 | 4/10 | 1/10 | 14/40 | 262 | 3.6 / 3.3 / 3.2 / 1.6 cm |
| `rl_true` | 8/10 | 6/10 | 7/10 | 6/10 | 27/40 | 91 | 5.7 / 3.6 / 2.4 / 1.8 cm |

| | `mpc` | `rl_true` | `rl_q` |
|---|---|---|---|
| steps of a placed episode, median per count | 113 / 95 / 93 / 80 | 116 / 153 / 132 / 164 | 113 / 132 / 144 / 149 |
| decision time | 0.3-0.5 s | 1 ms | 1 ms plus the encoder |
| cube lost after a grasp, per episode | 0.0-0.2 (1.2 on three obstacles) | 1.6-3.2 with obstacles | 0.4-1.3 |

The MPC is the quickest in control steps and the most precise; the policy takes 30-80 more steps on obstacle
scenes because it slows down near obstacles and drops and regrasps the cube on the way. Per decision the
policy is hundreds of times faster.

## How they fail

| failure (of 40 scenes) | `mpc` | `rl_true` | `rl_q` |
|---|---:|---:|---:|
| grasp failed: fingers close beside the cube, or never close | 7 | 4 | 14 |
| carried, not placed: held and lifted, never arrives or lost far out | 6 | 4 | 7 |
| released 7-14 cm from the goal, just outside the radius | 1 | 5 | 5 |

- `mpc`: the grasp failures are two cubes 1 cm from the table's edge (seeds 15 and 17, lost at every count)
  where `Q` reads the cube 3 cm off and the planner closes on nothing again and again. The carried-not-placed
  ones are all obstacle scenes: it holds the cube correctly, stops 30-40 cm short in front of a tall obstacle
  on the line to the goal and oscillates there. No 8-step plan gains goal distance by going around, and its
  pull is the straight-line distance (the cost-to-go `C` that fixes this exists but is not in the `mpc` yet).
- `rl_true`: half its failures are placements that just missed, let go 8-11 cm out, where the shaped reward
  stops growing. The rest are scenes where it hovers next to the cube and never closes, or pushes through
  the obstacles instead of around: one scene with three boxes gave 85 of its 91 contact steps.
- `rl_q`: the same policy through `Q`. Trained on the exact cube and an immediate contact flag, it closes
  1-2 cm off, the cube slips, the width rule keeps saying "held", and it carries nothing to the goal.

## Three obstacles, `mpc` against `rl_true`

| seed | obstacles | `mpc` | `rl_true` |
|---|---|---|---|
| 10, 13, 14, 16, 19 | mixed, one scene with a single wall | placed, 76-111 steps, 1.3-3.1 cm | placed, 121-185 steps, 0.8-5.3 cm |
| 11 | wall 18, wall 6, box 5 | ten closes beside the cube, held it 15 steps, dropped it | placed, 160 steps, 6.5 cm |
| 12 | wall 17, cylinder 14, cylinder 18 | carried to 30 cm from the goal, lost it beside the cylinder | hovered 3 cm from the cube and never closed |
| 15 | wall 16, wall 7, box 8 (edge cube) | never held, `Q` 3 cm off | held and dropped three times near the low wall |
| 17 | box 18, box 17, box 10 (edge cube) | held 228 steps, parked 42 cm out between the boxes | pushed through: 85 contact steps, eight drops |
| 18 | wall 17, box 6, cylinder 17 | held 218 steps, parked 31 cm out in front of the wall | carried to 6 cm and let go at 9 cm |

Both solve the same five scenes; the policy also solves seed 11. The MPC's losses are a planning-horizon
problem: a correct grasp, zero contacts, no plan that sees past a tall obstacle. The policy's are a control
problem: no plan at all, so it pushes, drops and regrasps, and lets go a little early. The MPC is better at
not colliding, at placing precisely and at recovering from a bad grasp, and worse at long detours and at
edge cubes `Q` misreads; the policy is better at a detour, hundreds of times faster per decision, and
collapses on the cameras.

## Why the MPC does as well with no contacts

- Obstacles are checked at decision time: `O` scores every imagined hand position against the map it sees
  now, so a plan that comes near a wall loses before it is executed, whatever the layout.
- The policy has a penalty for obstacles too (0.5 per contact step, 0.3 near one), but only in training. Most
  of it is cloned from demonstrations that route around obstacles with a planner; the 20,000 RL steps changed
  little (the cloned actor had 612 contact steps on the validation scenes, the chosen checkpoint 423, later
  ones 380 to 672), the checkpoint was picked by placements, and against +15 for a placement a few contact
  steps are a cheap gamble where the demonstrations hold no detour.
- The MPC re-plans every step, so a `Q` error at the grasp costs a few steps, not the episode; the policy
  only knows the exact cube.
- The world model learns the physics once from the recorded episodes and the planner searches at run time;
  the policy has to have met every situation in training. The world model was trained on 401 episodes. The
  packaged policy needed 500 scripted demonstrations plus those same 401 episodes in its replay buffer, and
  reaching its validation score from cloning alone took 1,200 demonstrations: about three times the data for
  a controller that then still loses half of it on the cameras. Demonstrations were the only thing that
  moved it (200 -> 500 -> 1,200: 11 -> 18 -> 20 of 30 validation placements); RL added one or two and then
  drifted.

On the exact state the policy matches the MPC in placements (27 against 26 of 40); the gap is perception
and contacts. The same seeds move by a scene or two between runs, so differences that small are noise.

## One tall obstacle in the middle

A fixed scene: the cube 60 cm from the goal and an 18 cm obstacle exactly on the line between them
(`configs/middle_wall_layout.json`, `middle_cylinder_layout.json`), `mpc` only, two episodes each.

| obstacle | result | what happened |
|---|---|---|
| a 16 cm wide wall across the line | 0/2 (41 and 46 cm short) | grasped at step 58, then hovered 9 cm before the wall for 240 steps; every plan moved at most 2 cm, the score flat |
| a 6 cm cylinder | 1/2 (9.1 cm short, then 1.1 cm) | went round it; the miss let go 2 cm outside the radius |

The wall is the horizon limit in its purest form: going round means 10 cm sideways and 20 cm forward, more
than eight steps, and the first eight of them look worse than standing still.

A longer horizon helps only sometimes (`controller/mpc_variants/horizon_16.py`, `horizon_24.py`): with 16
steps one of two episodes went round the right end of the wall after 170 steps of hovering and placed at
2.4 cm, the other parked; with 24 steps the one episode parked again. The planner then imagines 192 plans
of 16 or 24 steps per decision, and `D` was trained on 8-step windows, so the
far end of a long plan is guesswork; the detour has to pay within what `D` can still predict. The cost-to-go
is the better lever.

## The SACs trained in imagination

Two SACs with the old 41 inputs, trained only on transitions imagined by `D` (`controller/imagined_sac.py`,
run by `controller/sac_scratch.py`), on the cameras on 30 empty scenes and 30 with 1-2 obstacles, then the
`mpc` on the same scenes (export in `results/control_release_v1/`):

| controller | empty | obstacles | contact steps |
|---|---:|---:|---:|
| imagined SAC, empty-table starts | 12/30 | 11/30 | 746 |
| imagined SAC, mixed starts | 9/30 | 8/30 | 1,126 |
| `mpc` | 30/30 | 24/30 | 0 |

RL in imagination never improved on its cloned start: the updates move the policy off the recorded actions
where the imagination is faithful, and the critic barely fits a reward where 0.5 % of the steps carry the
placement bonus. On one obstacle scene the `mpc` still placed when made to execute its 5th, 25th or even
64th best plan at every step, because every step re-plans from the new image.

## The waypoint planner `jepa_mpc`

20 scenes with 0-2 obstacles, same seeds: the learned obstacle model 16/20 with zero contacts, with the
cost-to-go 15/20, the old geometric map with an A* route 14/20 with 233 contact steps, without the
finger-width rules 14/20. On 20 scenes with 1-2 obstacles `mpc` 13/20 and `jepa_mpc` 15/20, both with zero
contacts; the old fixed-layout SAC through `Q` 0/20 with 833.

## What is next

1. Train the SAC on the perception it will get: a lagged, noisy "held", 2 cm of cube noise or `Q`'s own
   readings, and a retry after a slip. More demonstrations on obstacle scenes first, since they are the lever.
2. Put the cost-to-go `C` into the `mpc`'s pull, so a detour round a tall obstacle pays within the horizon.
3. Retire the waypoint planner's two rules with `data/episodes_rules` (encode with the same bases, merge, retrain,
   run `jepa_mpc` with `--no-grasp-prior --no-release-rule`).
4. The edge cubes: more `lateral` and `miss` episodes near the table's edge, or a readout trained on the
   planner's own episodes.
5. Twenty scenes per count (`bash evaluate.sh "mpc rl_q rl_true" 20`) to settle the rates.

## Reproduce

```bash
bash evaluate.sh "mpc rl_q rl_true" 10                 # from the repository root
python simulation/results_table.py
python -m controller.control_pipeline --methods mpc --layout configs/middle_wall_layout.json --test-episodes 2 --headless --out data/wall
python -m controller.sac_scratch --steps 40000 --episodes 30
```
