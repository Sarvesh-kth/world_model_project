# Controller: four controllers and the CEM planner

```
control_pipeline.py   the four controllers on fresh empty-table scenes; WorldModels, episode(), the planner
control_clutter.py    the same controllers on the held-out obstacle scenes of a collect_obstacles run
run_final.sh          the whole evaluation: 20 empty scenes, 6 obstacle scenes with and without the penalty head
mpc.py                the standalone planner (method mpc): CEM in imagination with no SAC; mpc_config.py holds its knobs
replay.py             replay a recorded episode in a MuJoCo window or as gif/mp4, with the planner's imagined plans
diagnose.py           text diagnosis and one summary picture (diagnosis.png) of any recorded episode, no GPU
```

| method | perception | action | what it tests |
|---|---|---|---|
| scripted | exact state | the scripted policy | the upper bound of the simulator |
| rl_true | exact state | the frozen SAC | what the policy can do with perfect perception |
| rl_q | camera through `Q(z, p)` | the frozen SAC on the estimate | is the JEPA perception good enough to drive the policy |
| jepa_mpc | camera through `Q`, `D`, `R` | CEM around the SAC's plan, scored in imagination | does planning in the latent add anything (it is the only one that can avoid obstacles) |
| mpc | camera through `Q`, `D`, `R` | CEM in imagination starting from its own previous plan, no SAC | can planning in the world model do the task on its own (the proposal's CEM/MPC controller) |

```bash
python -m controller.control_pipeline --methods jepa_mpc                 # 5 scenes, live window
python -m controller.control_pipeline --test-episodes 20 --headless      # all four, the benchmark
python -m controller.control_clutter --methods rl_q jepa_mpc             # six held-out obstacle scenes
python -m controller.control_clutter --methods jepa_mpc --only scene_0019 scene_0021
bash controller/run_final.sh 20
python -m controller.control_pipeline --methods mpc --test-episodes 1 --headless --out data/mpc_smoke
python -m controller.mpc --self-check                                    # planner logic on the CPU, no camera
python -m controller.replay data/mpc_smoke/episodes/mpc/seed_20494010 --video   # gif + mp4 with the plans drawn
python -m controller.diagnose data/mpc_smoke/episodes/mpc/seed_20494010         # what went wrong, diagnosis.png
```

## The standalone planner, `mpc.py`, and the replay, `replay.py`

`mpc` searches like `jepa_mpc` (the same `forecasts()`: D imagines, Q reads, GoalReward plus R's penalties) but
without the SAC: no guide, no `--guide-margin`, the gripper is sampled by the CEM, and the search starts from its
own previous plan shifted by one step (the first plan holds still). Its knobs are in `mpc_config.py`, one
commented Python file to edit by hand (horizon, population, elites, iterations, noise, gripper `sampled` or
`rule`, score `reward` or `progress`, penalty scale); `--mpc-config other.py` uses a copy. Every mpc episode
saves `mpc_settings.json` (the values used) and `plans.npz` (per step the chosen plan's imagined gripper and
cube paths and the runner-ups' gripper paths). Every controller now also saves the simulator state per step
(`trajectory.npz` `qpos`), so `replay.py` can redraw any episode without a GPU: a MuJoCo window (on macOS
through `mjpython`; space pause, arrows step, up/down speed, R restart) or `--video` for a gif and an mp4.
`diagnose.py` reads the same files and prints what happened (arm motion, grasp, Q's and D's errors, the
candidates' scores, whether the plans expected a grasp that did not happen) and writes `diagnosis.png`
(six camera frames, top view with the imagined plans, distances, gripper, scores, model errors); in JupyterLab
double-click it, or `actual.gif` for the camera view. Without CUDA, `WorldModels` uses Apple's GPU (MPS), so
camera controllers also run on a Mac (about 3 min for a 300-step mpc episode).

## What runs, `control_pipeline.py`

`main()` loads everything once: the manifest of `--models-run` (its `config` is the environment config;
`known_rest_z` is the median resting cube height of its training states), the layout file, the SAC
(`--baseline-run/models/sac_best.zip`) and, if a camera method is requested, `WorldModels`. Then for every
method and every seed (`--seed + 20000 + i`, default 20494010 upward) it calls `episode()` and refreshes the
summary. Without `--resume` the output folders are wiped first.

**WorldModels(root, tag, args)** holds the online perception: the `Encoder` from `world_model/encode.py`
built with the settings in `features/meta.json` and the PCA basis in `features/pca.npz` (so online `z` is
exactly the cached `z`), `Q` (`readout.pt`, or `readout_p.pt` with `--blind`), `D` (`dynamics.pt`) and `R`
(`reward.pt` if present and not `--no-penalties`). Four methods:

- `encode(frames)`: the latent of the frame history (newest last, the encoder picks the last 16).
- `read(z, p)`: `Q`, cube xyz and held probability. The **width gate** then zeroes held when the measured
  finger width is under 2 cm: fingers closed on nothing hold nothing (the cube is 4.5 cm). Without it `Q`
  kept reporting held after a missed grasp and the policy carried air to B in every failed episode.
- `predict(z, p, actions)`: `D` one step for a batch of candidates.
- `penalty(z, p, weights)`: `R`'s weighted penalty, `penalty_scale * (0.3 proximity - 0.5 P(collision) - 0.2 P(table))`, <= 0.

**episode(root, method, seed, cfg, layout, rest_z, args, policy, models)** is one episode of one
controller, the same loop for all four:

1. `TaskSession.reset(seed)` (cube jittered by `--position-jitter`, 1 cm), the live window if there is a
   display, the first frame saved and (camera methods) encoded and read.
2. Each step picks the action: `scripted` from the state machine, `rl_true` from the SAC on the exact
   observation, `rl_q` from the SAC on `policy_observation(p, Q's xyz, Q's held, memory, ...)`, `jepa_mpc`
   from `plan()`.
3. The environment steps, the new frame is saved and encoded, `Q` reads it, the reward memory is advanced
   from the estimate, and the step is logged.
4. Outputs under `data/<out>/episodes/<method>/seed_<seed>/`: `frames/static_*.jpg`, `actual.gif`,
   `steps.csv` (per step: true and control reward, goal distance, lift, actions, the actual cube position,
   `Q_x Q_y Q_z Q_held_probability Q_xyz_mae_cm`, and for the planner `guide_score`,
   `best_candidate_score`, `kept_guide`, `predicted_penalty_sum`, `valid_candidates`, one-step errors of
   `D`), `candidates.csv` (every CEM candidate's score per step), `trajectory.npz`, `result.json` (the
   session result plus `Q_on_actual_observations`, `planner_fallback_steps`, `guide_kept_steps`).

`summarize()` writes `results/<out>/summary.{json,csv,txt}` (placements, mean and median final distance,
contacts, fallbacks per method) and copies every episode's json and csv there; `results/` is tracked,
`data/` is not.

## The planner

`plan()` is a cross-entropy method around the SAC. Every control step:

1. **Guide** (`actor_sequence`): roll the SAC through `D` for `--horizon` (8) steps, reading each imagined
   state with `Q`, to get the action sequence the policy would execute if the imagination were true.
2. **Candidates**: `--population` (64) sequences `guide + std * noise`, `std = --cem-std` (0.3). Half of the
   noise variance (`--cem-smooth` 0.5) is one draw shared over the whole horizon:
   `noise = sqrt(s) * eps_shared + sqrt(1 - s) * eps_step`. Independent per-step noise averages out to under
   1 cm of lateral spread over 8 steps, so no candidate could ever go around anything. The gripper is not
   sampled, it follows the guide (`--free-gripper` restores sampling): `D` and `Q` never saw fingers closed
   on nothing, so a candidate that closes early looks like a grasp in imagination and gets rewarded for
   it. Candidate 0 is always the guide itself, candidate 1 the previous step's plan shifted by one.
3. **Forecast** (`forecasts`): `D` rolls all candidates forward, `Q` reads every imagined state, the
   reward is `predicted_reward` on the estimate plus `R`'s penalty, discounted with gamma 0.99 and summed
   over the horizon; `--terminal-weight` (0) times the SAC critic's value of the final state can be added
   (1 in the original: the critic was exploited by `D`'s errors, candidates beat the guide by 0.5 to 1.5
   from imagination alone). A candidate is invalid when its imagination leaves the physical range
   (finger width outside [-0.2, 9] cm, hand or cube more than 5 m away, nonfinite numbers); invalid
   candidates score -1e9.
4. **Elites and refit**: the best `--elites` (8) give the new mean and std (floored at min(0.1, std)),
   repeat for `--iterations` (3); the best candidate over all iterations is kept.
5. **Decision**: the guide's own forecast is candidate 0, scored with the same `D`, `Q` and `R`. A candidate
   replaces the guide only when it beats that forecast by more than `--guide-margin` (0.25); otherwise the
   first guide action is executed. If the best candidate is invalid the guide is executed (a fallback).

The original planner (M2) is `--cem-std 0.6 --cem-smooth 0 --free-gripper --no-width-gate --guide-margin 0
--terminal-weight 1`. Each default above fixed a measured failure, the story is `CHANGES_FROM_M2.md`
sections P4 and C7: with the critic off and the margin at 0 the 8-step score differed by about 0.01
between candidates (pure `D` noise) and the guide was selected in 0 to 2 of 300 steps; with the gripper
sampled every failed episode closed beside the cube.

| flag | default | meaning |
|---|---|---|
| `--horizon` | 8 | steps `D` imagines |
| `--population` / `--elites` / `--iterations` | 64 / 8 / 3 | CEM size |
| `--cem-std` | 0.3 | noise around the guide (original 0.6) |
| `--cem-smooth` | 0.5 | share of the noise variance constant over the horizon |
| `--guide-margin` | 0.25 | a candidate must beat the guide's forecast by this much |
| `--terminal-weight` | 0 | SAC critic as terminal value (original 1) |
| `--penalty-scale` | 2 | multiplier on `R`'s penalties; 1 = the recorded weights, higher leaves the guide earlier near obstacles, 3 refused to approach a cube beside a wall |
| `--no-penalties` | | ignore `R` |
| `--free-gripper` | | let CEM sample the gripper |
| `--no-width-gate` | | keep `Q`'s held reading when the fingers are closed on nothing |
| `--blind` | | `readout_p` instead of `Q(z, p)`: the loop without the image (0/5 empty, 0/4 obstacles) |
| `--headless`, `--speed` | | no window / playback speed of the window |

## Obstacle scenes, `control_clutter.py`

Same `episode()`, but the layout comes from the `--scenes-run` (`data/obstacles_test1`) `scene_settings`:
the test scenes of that collection (`collection.json` `groups`), each with the obstacles of its
`<scene>_replay_obstacles` rollout (`--layout-suffix`). `--only scene_0019 scene_0021` picks scenes, the
cube is not jittered. `rl_true` and `rl_q` cannot see obstacles by construction; only the planner can,
through `R`. Writes `scenes.json` per scene next to the usual summary.

## Numbers and failure modes

Empty table, 20 fresh seeds (`results/control_final_20`): scripted 20/20, rl_true 20/20, rl_q 20/20,
jepa_mpc 19/20 (median 1.0 cm, max 2.2 cm); the one failure is a near-miss grasp after which `Q` says held
with the fingers closed on air. Six held-out obstacle scenes: rl_true 1/6 (923 contact steps), rl_q 2/6
(794), jepa_mpc 3/6 at penalty scale 1 (465), 2/6 without `R` (829), **4/6** at the default scale 2
(contacts 0 / 5 / 5 / 20 on the solved scenes). Scenes 19 and 21 are solved only by the planner and only
with the penalty head; scenes 22 and 23 (walls right beside the cube) fail before the grasp because `Q` is
1 to 2 cm off there.

Outcomes vary between runs of the same seed: the encoder runs in fp16 and the CEM amplifies small
differences in `Q`, so a scene that was solved once can be a near miss the next time. Quote rates over
20 seeds, not single episodes.

## The live window

With a display and without `--headless`, `LiveViewer` opens a MuJoCo window in a separate process that
rebuilds the same scene and mirrors the joint positions it is sent (the episode process renders its
camera through EGL and a window cannot share that context). `--speed` paces the playback but cannot beat
the planner's compute (about 0.3 s per step on an RTX 4060).
