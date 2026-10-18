# results

What the controllers write: `results/<run>/summary.{txt,json,csv}` and a copy of every episode's
`result.json`, `steps.csv` and `candidates.csv`. Frames and gifs stay in `simulation/data/<run>/episodes/`.
Everything here is local except this file and `control_release_v1/`, the export of the imagined-SAC runs.

## The test (18 October 2026)

Three controllers on the same fresh scenes: cube and goal anywhere on the table at least 35 cm apart, the
obstacles drawn between them, 0 / 1 / 2 / 3 obstacles, 10 scenes per count (seeds 20494010-19), 120
episodes in all, on one laptop with an RTX 4060 (runs `sac_priv_eval_<n>`, `bash sac_priv_eval.sh`). An
episode ends with the cube resting within 7 cm of the goal for 15 steps, after 300 steps, or when the cube
falls off the table. "3 obstacles" means up to three: the sampler fits as many as the corridor allows, so
two of those ten scenes have one and two obstacles (seeds 16 and 19), the same two scenes as in the
one- and two-obstacle runs.

- `mpc`, the latent MPC on the cameras (`controller/mpc.py`): every control step it encodes the last 16
  frames of both cameras with V-JEPA, reads the cube with `Q`, samples 64 action sequences of 8 steps
  around the previous plan, imagines each with `D`, scores them with the task reward on `Q`'s readings plus
  `R`'s penalties plus the obstacle model `O` at the imagined hand, refits three times (CEM) and executes
  the first action of the best plan. Its models were trained on the 401 recorded episodes of `episodes_v3`
  (58,000 states) and nothing else; it never received a reward from the simulator.
- `rl_true` and `rl_q` are one policy, `models/sac_priv_v1` (`rl/README.md`): a SAC with 188 inputs (robot
  state, cube, goal, held, the reward's memory, a 9x9 window of the obstacle map around the hand and the map
  pooled to 8x8), trained in the simulator with the simulator's own reward (placement +15, penalties for
  collision, proximity, table hit, drop and a failed episode): 500 scripted demonstrations and the same 401
  recorded episodes in the replay buffer, behaviour cloning, then 20,000 SAC steps with 1 cm of noise on the
  cube it sees. `rl_true` gets the exact cube position and contact flag; `rl_q` gets `Q`'s reading from the
  cameras with the controllers' finger-width rules for "held", the perception the MPC has.

## Outcomes

| controller | 0 obstacles | 1 | 2 | 3 | all | obstacle contact steps | median distance to the goal when placed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mpc` | 7/10 | 7/10 | 7/10 | 5/10 | 26/40 | 0 | 1.9 / 3.2 / 2.4 / 2.2 cm |
| `rl_q` | 5/10 | 4/10 | 4/10 | 1/10 | 14/40 | 262 (5 scenes) | 3.6 / 3.3 / 3.2 / 1.6 cm |
| `rl_true` | 8/10 | 6/10 | 7/10 | 6/10 | 27/40 | 91 (3 scenes) | 5.7 / 3.6 / 2.4 / 1.8 cm |

Scenes placed by the MPC only 15, by `rl_q` only 3, by both 11, by neither 11. Two scenes fail for the MPC
at every obstacle count (seeds 15 and 17): their cube starts 1 cm from the table's edge at y = -0.39, where
`Q` reads it 3 cm off, and the hand closes beside it.

## Speed and precision

| | `mpc` | `rl_true` | `rl_q` |
|---|---|---|---|
| control steps of a placed episode, median per count (0 / 1 / 2 / 3) | 113 / 95 / 93 / 80 | 116 / 153 / 132 / 164 | 113 / 132 / 144 / 149 |
| fastest and slowest placement | 67 and 282 steps | 100 and 297 | 100 and 245 |
| decision time | 340-540 ms (GPU shared with two other runs; 240 ms alone) | 1 ms | 1 ms, plus the encoder |
| wall clock of a placed episode | about a minute (the V-JEPA encode is another 250 ms per step) | seconds | tens of seconds |
| hand travel per episode | 150-180 cm | 120-150 cm | 120-150 cm |
| fingers closed per episode (mean) | 1.5-3.2 | 1.0-2.1 | 2.3-3.6 |
| cube lost after a grasp, per episode (mean) | 0.0-0.2, 1.2 on three obstacles | 1.6-3.2 with obstacles | 0.4-1.3 |

In control steps the MPC is the quickest, and it gets quicker with more obstacles because the scenes it
still solves are the easy ones; its placements are the most precise (1-3 cm). The policy takes 30-80 more
steps on the obstacle scenes: it slows down near obstacles, lifts higher, and lets go of the cube and
regrasps it on the way (1.6-3.2 losses per episode on obstacle scenes). In wall clock the policy wins: 1 ms
against half a second per decision, which the MPC spends imagining 192 plans of 8 steps (through the
cameras both also wait 250 ms for the encoder, so there the gap is about twofold).

## How they fail

Every failed episode sorted by what the saved trajectory shows (40 scenes each):

| failure | `mpc` | `rl_true` | `rl_q` | what it looks like |
|---|---:|---:|---:|---|
| grasp failed (never held, or held under 10 steps) | 7 | 4 | 14 | fingers close beside the cube, or never close |
| carried, not placed | 6 | 4 | 7 | the cube is held and lifted but never arrives, or is lost far from the goal |
| released near the goal (7-14 cm out) | 1 | 5 | 5 | let go just outside the 7 cm success radius |
| placed | 26 | 27 | 14 | |

`mpc`: the grasp failures are the two edge cubes (seeds 15 and 17, four and two times) and one more, where
`Q`'s reading is 2-3 cm off and the planner closes on nothing again and again, each time imagining a hold
(`D` has seen few closes beside the cube). The carried-not-placed failures are all obstacle scenes: with the
cube in hand it stops 30-40 cm short of the goal in front of a tall obstacle on the straight line to the
goal and oscillates there (the hand moves 20-70 cm over the last 80 steps but gets nowhere): no 8-step plan
gains goal distance by going around, so standing in front of the wall scores as well as anything. The pull
in the score is the straight-line distance; the waypoint planner's cost-to-go `C` was built for exactly
this and the `mpc` does not use it yet (`../NEXT.md`).

`rl_true`: five of its ten failures are placements that just missed, let go 8-11 cm from the goal (the
success radius is 7); the shaped reward stops growing there and the policy opens. The grasp failures are
scenes where it hovers next to the cube and never closes, or closes and loses it. The carried-not-placed
ones are obstacle scenes where it pushes through instead of around: one scene with three boxes gave 85 of
its 91 contact steps and eight drops.

`rl_q`: the same policy through `Q`. Fourteen grasp failures: it was trained on the exact cube and an
immediate contact flag, so with `Q` 1-2 cm off it closes off-centre, the cube slips after a few steps, and
the finger-width rule keeps reporting "held", so it carries nothing to the goal (the "carried, not placed"
rows). The MPC runs on the same `Q` and loses far less to it because it re-plans every step: a missed close
costs a few steps (the empty-grip cost and the "a grasp needs open fingers arriving at the cube" check make
the next plan open and try again).

## Three obstacles: `mpc` against `rl_true`, scene by scene

| seed | obstacles (kind, height) | `mpc` | `rl_true` |
|---|---|---|---|
| 10 | box 17, box 6, cylinder 16 | placed, 80 steps, 1.3 cm | placed, 171 steps, 1.8 cm |
| 11 | wall 18, wall 6, box 5 | lost: ten closes beside the cube, held it for 15 steps at step 146 and dropped it | placed, 160 steps, 6.5 cm |
| 12 | wall 17, cylinder 14, cylinder 18 | lost: carried the cube 18 cm high to 30 cm from the goal, lost it beside the 14 cm cylinder at step 150 | lost: hovered 3 cm from the cube, 5 cm up, and never closed; moved 70 cm in 300 steps |
| 13 | wall 18, cylinder 7, wall 14 | placed, 103 steps, 2.4 cm | placed, 167 steps, 0.8 cm |
| 14 | cylinder 19, box 11, cylinder 9 | placed, 111 steps, 2.2 cm | placed, 185 steps, 5.3 cm |
| 15 | wall 16, wall 7, box 8 (edge cube) | lost: never held, `Q` 2.9 cm off, seven closes on nothing | lost: held it three times and dropped it three times near the low wall, one contact step |
| 16 | wall 18 (only one fitted) | placed, 77 steps, 3.1 cm | placed, 121 steps, 1.9 cm |
| 17 | box 18, box 17, box 10 (edge cube) | lost: held it for 228 steps, got to 42 cm from the goal between the low box and the two tall ones, dropped and regrasped it five times, never found a way past | lost: pushed through the boxes: 85 contact steps, ten closes, eight drops |
| 18 | wall 17, box 6, cylinder 17 | lost: held it for 218 steps, stopped 31 cm short, 10 cm in front of the 17 cm wall, oscillating (70 cm of hand travel in the last 80 steps, 4 cm net) | lost: carried it to 6 cm from the goal and let go at 9 cm, outside the radius |
| 19 | box 18, wall 8 (two fitted) | placed, 76 steps, 1.3 cm | placed, 127 steps, 1.2 cm |

Both solve the same five scenes; `rl_true` also solves seed 11 and both lose 12, 15, 17 and 18. On the five
shared placements the MPC needs 76-111 steps against 121-185, and ends 1.3-3.1 cm from the goal against
0.8-5.3 cm. Zero contacts for the MPC on every episode; the policy touched an obstacle on 86 steps, 85 of
them in one scene where it drove the cube through three boxes.

What is different: the MPC's failures with three obstacles are a planning-horizon problem. It holds the
cube correctly and keeps away from everything (the obstacle model keeps it 10-17 cm from the obstacle edges
while carrying), but with a tall obstacle on the line to the goal no 8-step plan sees the payoff of a
detour, so it waits. The policy's failures are a control problem: it does not plan at all and has no
obstacle in its score at run time, so it goes straight, pushes, drops and regrasps, and when it does arrive
it lets go a little too early. Each is bad at what the other is good at: the policy does find a way past
(seed 11) and arrives (seed 18), the MPC never touches anything and places exactly.

What is better or worse, in short: the MPC is better at not colliding (0 against 86 contact steps), at
placing precisely (2 cm against 2-6 cm) and at recovering from a bad grasp; it is worse at long detours
and at edge cubes `Q` misreads, and it costs half a second per step. The policy is better at a detour and
at the raw placement count with three obstacles (6 against 5, one scene, within noise), and hundreds of times
faster per decision on the exact state; it is worse at obstacles it was never penalised enough for, at the final centimetres,
and it collapses on the cameras (1/10).

## Why the MPC, with less experience, does as well with no contacts

- Obstacles are checked at decision time, not remembered. `O` scores every imagined hand position against
  the obstacle map it sees now, so a plan that comes near a wall loses before it is executed. The SAC only
  met obstacles as a training penalty and has to keep the avoidance in its weights; nothing stops it at test
  time.
- It re-plans from what it sees, so a `Q` error at the grasp costs steps, not the episode (above).
- The world model needed only the recorded episodes because it learns the physics once and the planner does
  the search at run time; the policy has to have met every situation in training. More demonstrations were
  the only thing that moved its validation (200 -> 500 -> 1200 demonstrations: 11 -> 18 -> 20 of 30
  placements from cloning alone); the RL phase added one or two placements in its first 20-40k steps and
  then drifted.

Read honestly: on the exact state the SAC matches the MPC in placements (27 against 26 of 40). The gap in
the table is the perception gap and the contacts. With 10 scenes per count and a run-to-run spread of about
10 % on the same seeds, differences of a scene or two are noise. The MPC placed 30/30 on empty tables on the
cluster's seeds (20495010 upward) and 7/10 here (20494010 upward): different scenes, and in two of the ten
local ones `Q` reads the cube 4-5 cm off along y for the whole episode.

To reproduce, from `simulation/` (20 scenes per count take about 5 hours on a free GPU):

```bash
bash sac_priv_eval.sh mpc 10
bash sac_priv_eval.sh rl_q 10
bash sac_priv_eval.sh rl_true 10
python sac_priv_table.py
```

## The SACs trained in imagination against the MPC (17 October, cluster)

Two SACs with the old 41 inputs (no obstacle input), trained only on transitions imagined by `D` with the
reward computed on `Q`'s readings (`controller/imagined_sac.py`, run by `controller/sac_scratch.py`): one
started its imagined episodes from empty-table states, the other from empty and obstacle states. Both ran on
the cameras on the same 30 empty scenes (seeds 20495010-39) and 30 scenes with 1-2 obstacles (20496010-39),
then the `mpc` on the same scenes. Episodes and figures are in `control_release_v1/`.

| controller | scenes | placed | mean final distance | obstacle contact steps |
|---|---|---:|---:|---:|
| imagined SAC, empty starts | empty | 12/30 | 23.7 cm | 0 |
| imagined SAC, empty starts | obstacles | 11/30 | 24.2 cm | 746 |
| imagined SAC, mixed starts | empty | 9/30 | 26.5 cm | 0 |
| imagined SAC, mixed starts | obstacles | 8/30 | 25.0 cm | 1,126 |
| `mpc` | empty | 30/30 | 2.2 cm | 0 |
| `mpc` | obstacles | 24/30 | 9.6 cm | 0 |

RL in imagination never improved on its cloned start: the imagination is faithful along the recorded
actions but the updates move the policy off them, and the critic barely fits a reward where 0.5 % of the
steps carry the placement bonus (`controller/README.md`). The four rank-selection episodes for the slides are in
`control_release_v1/runs/mpc_counterfactual_seed20496039/`: on one obstacle scene the MPC still placed when
made to execute its 5th, 25th or even 64th best plan at every step (93, 153 and 171 steps against 125 for
the best), because every step re-plans from the new image; executing the best for 60 steps and the 64th
after that did not place.

## The waypoint planner `jepa_mpc` (13-15 October)

20 fresh scenes with 0-2 obstacles, same seeds, models `models/v3c`:

| variant | placed | contact steps |
|---|---:|---:|
| learned obstacle model `O` (the default) | 16/20 | 0 |
| `O` + the cost-to-go `C` (`--cost-to-go`) | 15/20 | 0 |
| geometric height map + A* route (`--obstacle-source map`) | 14/20 | 233 |
| `O`, no finger-width rules (`--no-width-gate`) | 14/20 | 0 |

20 scenes with 1-2 obstacles on the cluster: `mpc` 13/20, `jepa_mpc` 15/20, the old fixed-layout SAC
through `Q` 0/20 with 833 contact steps, all with 0 contacts for the planners; on the six held-out clutter
scenes (`controller.control_clutter`) `mpc` 4/6, `jepa_mpc` 3/6, that SAC 3/6 with 288 contact steps.

## Reading the numbers

The same 20 seeds move by about 10 % between runs of the same code and models (the encoder runs in fp16
and the CEM amplifies small differences): quote 20-scene rates and treat smaller gaps as noise. Contact steps
are control steps in contact, not separate collisions. The full history is in `../STATUS.txt`.
