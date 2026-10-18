# results

What the controllers write: `results/<run>/summary.{txt,json,csv}` and a copy of every episode's
`result.json`, `steps.csv` and `candidates.csv`. Frames and gifs stay in `simulation/data/<run>/episodes/`.
Everything here is local except this file and `control_release_v1/`, the export of the imagined-SAC runs.

## Status: the three controllers on the same scenes (18 October 2026)

Same fresh scenes for every controller: cube and goal anywhere on the table, 0 / 1 / 2 / 3 obstacles in
between, 10 scenes per count (seeds 20494010-19), one laptop with an RTX 4060, runs `sac_priv_eval_<n>`.

| controller | 0 obstacles | 1 | 2 | 3 | all | obstacle contact steps | median distance to the goal when placed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mpc`, the latent MPC on the cameras | 7/10 | 7/10 | 7/10 | 5/10 | 26/40 | 0 | 2.4-3.5 cm |
| `rl_q`, the SAC on the cameras through `Q` | 5/10 | 4/10 | 4/10 | 1/10 | 14/40 | 262 (5 scenes) | 6.6-7.4 cm |
| `rl_true`, the same SAC on the exact state | 8/10 | 6/10 | 7/10 | 6/10 | 27/40 | 91 (3 scenes) | 3.7-6.3 cm |

Decision time: MPC 340-540 ms with the GPU shared, SAC 1 ms. Scenes placed by the MPC only 15, by `rl_q`
only 3, by both 11, by neither 11.

### How the three differ

`mpc` (`controller/mpc.py`, knobs in `mpc_config.py`): every control step it encodes the last 16 frames of
both cameras with V-JEPA, reads the cube with `Q`, samples 64 action sequences of 8 steps around the previous
plan, imagines each with `D`, scores them with the task reward on `Q`'s readings plus `R`'s penalties plus
the obstacle model `O` at the imagined hand, refits three times (CEM) and executes the first action of the
best plan. Its models were trained on the 401 recorded episodes of `episodes_v3` (58,000 states) and nothing
else; it never received a reward from the simulator.

`rl_q` and `rl_true` are one policy, `models/sac_priv_v1` (`rl/README.md`): a SAC with 188 inputs (robot
state, cube, goal, held, the reward's memory, a 9x9 window of the obstacle map around the hand and the map
pooled to 8x8), trained in the simulator with the simulator's own reward (placement +15, penalties for
collision, proximity, table hit, drop and a failed episode): 500 scripted demonstrations and the same 401
recorded episodes in the replay buffer, behaviour cloning, then 20,000 SAC steps with 1 cm of noise on the
cube it sees. `rl_true` feeds it the exact cube position and contact flag; `rl_q` feeds it `Q`'s reading
from the cameras with the controllers' finger-width rules for "held", the perception the MPC has.

### Why the MPC, with less experience, has no contacts and does as well or better

- Obstacles are checked at decision time, not remembered. `O` scores every imagined hand position against
  the obstacle map it sees now, so a plan that comes near a wall loses before it is executed. The SAC only
  met obstacles as a training penalty and has to keep the avoidance in its weights; nothing stops it at test
  time, hence 91 contact steps on the exact state and 262 through `Q`.
- It re-plans from what it sees. `Q` is 1-2 cm off at the grasp for both controllers. For the MPC a missed
  close costs a few steps: the empty-grip cost and the "a grasp needs open fingers arriving at the cube" check
  make the next plan open and try again. The policy was trained on the exact cube and an immediate contact
  flag: it closes off-centre, the cube slips after 3-4 steps, the width rule keeps reporting "held", and it
  carries nothing to the goal. Every `rl_q` failure on 0-1 obstacles is one of these.
- It places more precisely: median 2.4-3.5 cm from the goal against 6.6-7.4 cm for the SAC (the success
  radius is 7 cm); the policy lets go where its shaped reward stops growing.
- The world model needed only the recorded episodes because it learns the physics once and the planner does
  the search at run time; the policy has to have met every situation in training. More demonstrations were
  the only thing that moved its validation (200 -> 500 -> 1200 demonstrations: 11 -> 18 -> 20 of 30
  placements from cloning alone); the RL phase added one or two placements in its first 20-40k steps and
  then drifted.

Read honestly: on the exact state the SAC matches the MPC in placements (27 against 26 of 40) and is better
on three obstacles (6 against 5 of 10). The gap in the table is the perception gap and the contacts. With 10
scenes per count and a run-to-run spread of about 10 % on the same seeds, differences of a scene or two are
noise. The MPC placed 30/30 on empty tables on the cluster's seeds (20495010 upward) and 7/10 here
(20494010 upward): different scenes, and in two of the ten local ones `Q` reads the cube 4-5 cm off along
y for the whole episode.

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
