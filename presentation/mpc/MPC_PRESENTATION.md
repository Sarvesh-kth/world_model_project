# MPC presentation: which examples to show, and what to say

For whoever (person or AI) builds the presentation slides for **controller 1, the CEM/MPC planner** (M3, Calle) of
the DD2600 project "JEPA World Model for Vision-Based Pick and Place". Every clip below was checked against
its episode's recorded data. Don't swap in other clips without checking their `result.json`, and don't claim
more than the "What to say" column.

Code: branch `calle-v3c` of https://github.com/Sarvesh-kth/world_model_project, `simulation/controller/mpc.py`
and `mpc_config.py` (frozen at commit `39d7a77c`). Full findings: `simulation/controller/README.md`, section
"The rule-free MPC".

## The method in four sentences (for the method slide)

1. Every 0.1 s the planner samples 64 candidate sequences of 8 raw actions (move dx, dy, dz, turn, gripper open
   or closed), and refines them with CEM over 3 rounds (keep the best 8, resample around them).
2. It does not try them in the simulator. It imagines each one with the JEPA world model: the camera images
   (static and wrist camera) are encoded to a latent, the learned dynamics model D predicts the next latents,
   and the readout Q reads the cube's position and "held" from every imagined latent.
3. Each imagined plan is scored with the task reward on Q's readings, plus the learned penalty model R and the
   learned obstacle model O (collisions, table, empty grasps, obstacles). The first action of the best plan is
   executed, and the whole thing repeats (model predictive control).
4. Honest caveat: the score also has hand-made parts (a pull toward the cube and then B, a glide slope and
   a carry height, a few checks on what counts as a grasp or a placement in imagination). There are no waypoints,
   no "go to" rule and no release rule; the planner decides when to close and open the gripper. The ablation
   `mpc_variants/plain_scoring.py` (only the learned reward and penalties) shows what the hand-made parts add.

## Headline numbers (for the results slide)

| Test | Placed | Obstacle contact steps | Source |
|---|---|---|---|
| 10 fresh empty-table scenes, random cube and target positions | **10/10**, 1.7 cm mean from B | 0 | Mac, `results/mac_v3c_mpc_fresh10` (penalty scale 2 then) |
| 20 fresh scenes with 1-2 obstacles | **13/20** (current default, penalty scale 1); 11/20 at scale 2 | **0** | cluster, `results/v3c_obs_p1`, `results/v3c_obs` |
| Sarvesh's 6 held-out obstacle scenes | **4/6** | 0 | cluster, `results/v3c_clutter` (scale 2) |
| References, same 20 obstacle scenes: jepa_mpc (Sarvesh's rule-based planner) / rl_q (team SAC) | 15/20 / 0/20 | 0 / 833 | cluster, `results/v3c_obs` |

About 0.25-0.3 s per decision. The empty-table and held-out rows were run at penalty scale 2 (the default
then), the 20 obstacle scenes at the current scale 1. Controller 2 is compared on the same seeds:
empty tables `--seed 20475010 --test-episodes 10`, obstacles `--seed 20476010 --test-episodes 20 --obstacles 1 2`.

## Clips that exist now (folder `media/`)

Each example has: `*_replay.mp4` / `.gif` (overview camera with what the planner imagined drawn on top),
`*_replay.png` (last frame) and `*_camera.gif` (the static camera the world model actually sees). The `.mp4`
and `.png` files are in the repo (`presentation/mpc/media/`); the `.gif` files (57 MB) are only in
`mpc_presentation.zip`, which Calle shares separately.

How to read a replay (put this legend on the slide):

| Colour | Meaning |
|---|---|
| orange, thick | the plan chosen at this step: the imagined gripper path |
| orange, faint | runner-up plans |
| purple dots | where the planner imagined the cube along its chosen plan |
| blue dots | where the gripper really went |
| green disc | the target B |

| File prefix | Shows | What to say | Use on |
|---|---|---|---|
| `S1_empty_success` | seed 20495010, fresh scene: grasp, carry, place 0.5 cm from B in 81 steps | the standard success; the imagined path (orange) leads the real one (blue) | method or results slide (best opener) |
| `S2_empty_long_carry` | seed 20495017: the cube starts 66 cm from B, placed 1.7 cm from B in 101 steps | long transport; planning works over the whole table | results |
| `P1a_before_side_push` then `P1b_after_approach_slope` | same scene, seed 20494012. Before: the hand came down beside the cube and pushed it sideways (never placed). After the glide-slope fix (`APPROACH_SLOPE`): grasped from above, placed 2.6 cm from B | a failure found, explained and fixed; show the two side by side | "failures and fixes" |
| `P2a_before_let_go_short` then `P2b_after_released_stays_put` | same scene, seed 20494018. Before: let go 7.5 cm from B, because Q reads a cube resting near B as being at B. After the fix (`RELEASED_STAYS_PUT`: in imagination a released cube stays where it was let go): placed 2.5 cm from B | the planner is only as good as the world model's readings; fixed in the planner, not in Q | "failures and fixes" |
| `F1_old_cube_pushed_off_table` | seed 20494015, old version (before the glide-slope fix): the hand pushed the cube off the table near the edge in 39 steps | dramatic failure of the first port; after the fix the same scene leaves the cube on the table, but still never grasps it (the edge problem of F3) | "failures and fixes" (optional) |
| `F2_open_held_flicker_drop` | seed 20494010, current version: grasped, then Q's "held" reading flickered off, the planner believed it had dropped the cube, opened the gripper and kept chasing | **open failure**, a world-model reading error, not fixed | "limitations" |
| `F3_open_edge_never_grasped` | seed 20494017, current version: the cube starts at the table's -y edge; there Q's cube estimate slides 3-7 cm toward the nearby hand, so the grasp keeps missing (the cube now stays on the table) | **open failure**, a world-model reading error near the table edge | "limitations" |

All Mac clips are empty-table scenes recorded on 2026-10-14. P1a, P2a and F1 are from the version before the
two fixes; all the others are the current planner.

## Obstacle clips (still on the KTH cluster, fetch them)

These are the best evidence for O. On the cluster, from `~/world_model_project/simulation`:

```bash
tar czf ~/mpc_examples.tgz --exclude=frames data/v3c_obs/settings.json \
  data/v3c_obs/episodes/mpc/seed_20496016 data/v3c_obs/episodes/mpc/seed_20496022 \
  data/v3c_obs/episodes/rl_q/seed_20496022 data/v3c_obs/episodes/mpc/seed_20496029 \
  data/v3c_obs/episodes/mpc/seed_20496017 data/v3c_clutter/episodes/mpc/seed_40464023
```

Download `~/mpc_examples.tgz` (about 100 MB). Each episode folder already contains `actual.gif` (camera view).
For the replay with imagined plans, unpack it into `simulation/` on a machine with the repo and run
`python -m controller.replay data/v3c_obs/episodes/mpc/seed_20496016 --video` (Mac: `../.venv/bin/python`).
It writes `replay.gif`, `.mp4` and `.png` into the episode folder.

| Episode | Shows (penalty scale 2 run) | What to say | Use on |
|---|---|---|---|
| `v3c_obs/.../mpc/seed_20496016` | an 18 cm tall wall lies right across the path from the cube to B; the cube is lifted to 18 cm and placed 2.0 cm from B, 0 contacts | obstacle avoidance learned through O, no hand-coded route | results (obstacles), best obstacle success |
| `v3c_obs/.../mpc/seed_20496022` next to `v3c_obs/.../rl_q/seed_20496022` | same scene: mpc places 2.1 cm from B with 0 contacts; the team SAC (rl_q, no obstacle model) collides on 283 steps and fails | why planning with a learned obstacle model matters | results (obstacles), side by side |
| `v3c_obs/.../mpc/seed_20496029` | holds the cube to the end, but hovers about 18 cm high, within 8 cm of a 17 cm wall, 39 cm from B | **main open failure with obstacles**: an 8-step plan around or over the wall scores worse than waiting (straight-line progress score); jepa_mpc failed this scene too | "limitations" |
| `v3c_obs/.../mpc/seed_20496017` | never grasps: Q reads the cube within about 1-2 cm the whole time, but the hand never comes within 5 cm of it; an 18 cm box stands 21 cm from the cube | open failure, cause not investigated, so say exactly that | "limitations" (optional) |
| `v3c_clutter/.../mpc/seed_40464023` | one of Sarvesh's 6 held-out obstacle scenes, placed | generalization to scenes the models never trained on | results (generalization) |

**For the current default (penalty scale 1),** the same seeds are in `data/v3c_obs_p1` on the cluster. To pick
other examples, run `python -m controller.compare data/v3c_obs_p1 --seeds` and check each pick's `result.json`
(`task_success`, `final_goal_distance_cm`, `obstacle_contact_steps`) before using it.

## Rules for the slides

- Label every clip with its seed and whether it is the current version or a "before the fix" example.
- Say "fails" plainly for F2, F3, seed 20496029 and seed 20496017; they are part of the evaluation (when and
  why it works or fails), not something to hide.
- Don't call the planner "learned end to end": the dynamics, readout, reward penalties and obstacle model are
  learned, but the score has hand-made parts (see the method slide's caveat).
- Single runs are noisy: about ±2 placements of 20 between runs. Don't present a 1-2 point difference as a
  result.
- The comparison with controller 2 (the actor-critic trained in imagination) belongs on the same seeds (see
  Headline numbers), not on these clips.
