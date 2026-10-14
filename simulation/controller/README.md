# Controller: the latent MPC and the actor-critic, side by side

```
planner.py            the latent MPC: waypoint candidates, imagination with D, scoring with Q, R, O (and C), CEM
control_pipeline.py   the four controllers on fresh scenes; WorldModels (online perception), episode(), summaries
control_clutter.py    the same controllers on the six held-out obstacle scenes of data/obstacles_v2
imagined_sac.py       M3: a SAC trained only on transitions imagined by D (the actor-critic in imagination)
mpc.py                M3: the rule-free MPC, CEM over raw action sequences imagined by D; knobs in mpc_config.py
mpc_variants/         labelled comparison configs for mpc (rule gripper, bare task reward, ...)
compare.py            placed / failure kinds per run folder;  diagnose.py  one episode as text + diagnosis.png
replay.py             an episode redrawn in a MuJoCo window or as gif/mp4, with mpc's imagined plans
```

| method | perception | action | what it tests |
|---|---|---|---|
| scripted | exact state | the scripted policy | the upper bound of the simulator |
| rl_true | exact state | the frozen SAC | the actor-critic with perfect perception |
| rl_q | cameras through `Q` | the frozen SAC on the estimate | is the perception good enough to drive a policy |
| jepa_mpc | cameras through `Q`, `D`, `R`; the obstacle-only depth map through `O` | waypoint candidates imagined by `D`, the best one executed | does planning in the latent find the task and the way around obstacles |
| sac_imagined | cameras through `Q` (as rl_q) | a SAC trained in imagination (`imagined_sac.py`, `--imagined-run`) | can an actor-critic learn the task from the world model instead of the simulator |
| mpc | cameras through `Q`, `D`, `R`; the obstacle-only depth map through `O` | raw action sequences imagined by `D`, no rules on the motion or the gripper (`mpc.py`) | can a plain search in the latent do the task without hand-coded behaviour |

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

## The actor-critic trained in imagination, `imagined_sac.py` (M3)

```bash
python -m controller.imagined_sac --out data/sac_imagined_v1      # needs data/episodes_v3 (LFS, latents.npy assembled)
python -m controller.control_pipeline --methods sac_imagined --imagined-run data/sac_imagined_v1 --random-ends --test-episodes 20 --headless
```

A stable-baselines3 SAC with the exact-state SAC's recipe (behaviour cloning, critic warmup, a cloning term in
every actor update, `rl/rl_baseline.py`), but every training transition is imagined: an episode starts at a
recorded state of `data/episodes_v3` (obstacle-free scenes; each recorded episode is replayed through `Q` so the
start carries the task memory the controller would have had), `D` imagines the next (z, p), `Q` reads the cube
and held, and the reward is the SAC's `GoalReward` on `Q`'s readings plus `R`'s penalties. An imagined hold only
counts if the fingers closed at `Q`'s cube (the planner's gripper envelope). The policy sees the same 41 inputs as
rl_q, so it runs on the cameras through the rl_q code path. The checkpoint is picked by imagined placements on
held-out starts; a 20-episode exact-state check is logged as a diagnostic only.

Findings (Mac, 2026-10-13, obstacle-free random-end scenes):

| policy | exact state, 20 scenes | cameras, 10 scenes |
|---|---|---|
| cloned actor only (240 epochs on the data's clean episodes) | 14-18/20 | 7/10 (the 256-256 actor, 18/20) |
| + 40k imagined transitions, cloning weight 100 (the recipe) | 10/20 | 6/10 |
| + cloning weight 10 | 4-8/20 | |
| pure SAC (weight 0) | 0/20 | |
| reference on the same camera scenes: jepa_mpc 10/10, rl_q (SAC trained on one fixed layout) 1/10 | | |

RL in imagination never improved on its cloned start. Imagination is faithful along the data's actions (recorded
actions replayed: cube 1.0 -> 1.5 cm over 10 steps, held right 98.8 %, 10-step return correlation 0.98 with the
truth), but the updates move the policy off them: the failures shift to grasping, then reaching (weight 10: 7 of
20 scenes reach the cube and never hold it), while the imagined placements stay the same, and longer imagined
episodes (25 steps) make it worse (1/20). The critic barely fits (loss 0.7-1.0 against per-step rewards of about
0.03; 0.5 % of steps carry the +15 placement bonus, 57 % of the reward), a longer critic warmup and TD3+BC's
normalisation did not help. `R`'s penalties cost -0.023 per step on empty tables (phantom proximity).

## The rule-free MPC, `mpc.py` (M3)

```bash
python -m controller.mpc --self-check                                                     # planner logic on the CPU
python -m controller.control_pipeline --methods mpc --random-ends --test-episodes 20 --headless --out data/mpc_eval
python -m controller.control_pipeline --methods mpc --random-ends --obstacles 0 2 --test-episodes 20 --headless --out data/mpc_obst
python -m controller.control_pipeline --methods mpc --mpc-config controller/mpc_variants/rule_gripper.py ...   # a comparison row
python -m controller.compare data/mpc_eval data/mpc_obst --seeds                         # placed / failure kinds per run
python -m controller.diagnose data/mpc_eval/episodes/mpc/seed_20494010                   # text + diagnosis.png
python -m controller.replay data/mpc_eval/episodes/mpc/seed_20494010 --video             # replay with the imagined plans
```

The proposal's CEM/MPC in its plainest form: every candidate is a sequence of 8 raw actions (dx, dy, dz, dyaw,
gripper), sampled around the previous plan, with no waypoints, no go-to rule and no release rule; the gripper is
one sampled switch time per plan. `D` imagines each plan, `Q` reads every imagined state, and a plan is scored
with the SAC's own task reward (`GoalReward` on `Q`'s readings, discounted) plus `R`'s penalties and `O`'s at the
imagined hand on the obstacle-only map (`imagine()`). What is hand-made is only in the score and in what may be
imagined, all of it in `mpc_config.py` with the reason and the measurement behind each knob: a pull towards the
cube and then B that does not fade with distance (the task reward is flat beyond 30 cm), a glide slope and a
carry height, a grasp only counts if open fingers reached the cube before closing, a placement counts after 3
settled steps inside 3 cm in imagination. `mpc_variants/` holds labelled comparison rows (a rule gripper, the
bare task reward, no steadiness cost, penalty scale 1). On the older models (world_model_testing) this planner
placed 14/20 on fresh empty-table scenes and 4/20 in obstacle scenes; ported to `models/v3c` on 2026-10-14
(the v3c `Q` reads a carried cube's height without the old 3 cm bias, so `HELD_Z_OFFSET` is 0).

v3c findings (Mac, random-end empty-table scenes, 2026-10-14):

| run | placed | notes |
|---|---|---|
| 10 comparison scenes (seeds 20494010-019), port only + `PLACED_KEEPS_POTENTIAL` | 6/10 | placed cubes 4.9-5.5 cm from B; 3 cubes pushed from the side near the -y edge, 1 let go 9.8 cm short |
| the same 10 with `RELEASED_STAYS_PUT` + `APPROACH_SLOPE` (now development scenes) | 7/10 | placed 0.7-3.3 cm from B; 1 dropped (Q's held flickered off), 2 cubes near the -y edge never grasped |
| **10 fresh scenes (seeds 20495010-019), never tuned on** | **10/10** | mean 1.7 cm from B, 66-121 steps, 0.24 s per decision; no fresh cube started beyond y = -0.32 |

References on the 10 comparison scenes: jepa_mpc 10/10, the cloned actor 7/10, sac_imagined 6/10 (those seeds
were later tuned on for mpc, so compare on fresh seeds). What the three v3c knobs fix, measured on recorded
states: without `PLACED_KEEPS_POTENTIAL` the "progress" score made the release keep slipping to the last step
that still fit in the horizon (the planner held the cube at B to the end); without `RELEASED_STAYS_PUT` a cube
resting near B is read about 2 cm from B wherever it is, so the planner let go short; without `APPROACH_SLOPE`
the hand came down to cube height 5-10 cm away and pushed the cube. Open: near the -y table edge `Q`'s cube
reading slides 3-7 cm toward a nearby hand and the grasp misses; `Q`'s held reading can flicker off on a held
cube (the gripper envelope then cancels the width rule) and the planner opens. Not yet run on obstacle scenes.

## The live window

Without `--headless` and with a display, a MuJoCo window in a separate process mirrors the episode
(`window.py` finds a GLX setup; the episode itself renders through EGL). `--speed` paces it.
