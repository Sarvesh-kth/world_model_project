# NOTES

Decisions, results and open questions for M3, feeding the report. Newest entries at the bottom
of each section.

## Decisions

- **2026-09-26 PyTorch** for all M3 code, because V-JEPA 2 is PyTorch. *Not yet confirmed by the
  team* (the course labs use JAX).
- **2026-09-26 Build against M1's real env, no stand-in.** M1's repo arrived before the Fetch stand-in
  was built, so that was dropped. M1's env runs on the Mac from the M3 venv (see Results).
- **2026-09-26 Work lives in the shared repo** `world_model_project`, on branch `m3-control`, in
  `controller/` (the folder M1's README reserves for the planner). Never committed straight to
  `main`; changes reach it through pull requests. The first three commits were made in a personal
  prework repo (`m3-control-prework`, kept locally as an archive) and copied over.
- **2026-09-26 `controller` runs from the repo root** (`python -m pytest controller/tests`,
  `python -m controller.experiments.<name>`) and isn't pip-installed. M1's code runs from inside
  `simulation/`; `adapters/m1_adapter.py` bridges that by putting `simulation/` on the import path.
  Dependencies: `controller/requirements.txt` (M1's `requirements.txt` + torch + pytest).
- **2026-09-26 Outputs go under `data/`** (`data/runs`, `data/checkpoints`), which the repo's
  `.gitignore` already ignores, so no root file had to change.
- **2026-09-26 `EnvAdapter.step` returns Gymnasium's 5-tuple** `(obs, reward, terminated, truncated,
  info)`, not a single `done`: the actor-critic needs truncation vs termination to bootstrap
  correctly. M1's env already does this.
- **2026-09-26 `planner.py` separate from `cem.py`.** `cem.py` only holds Calle's code, which keeps
  the AI-use split clear. The planner loop runs `execute_steps` actions per plan; Level E uses a
  few, Grade C MPC is `execute_steps = 1`.
- **2026-09-26 Starting CEM hyperparameters** (`config.CEMConfig`): H = 10, N = 300, K = 30,
  4 iterations, init std 0.5, std floor 0.05, 3 executed steps per plan.

## Results

- **2026-09-26 M1 env smoke test** (Apple M5, M1 `bb60e1a`, seed 0): reset 0.27 s (rebuilds the
  MuJoCo model per episode), `env.step` 1.5 ms mean / 2.7 ms max, scripted "success" policy
  places the object in 95 steps. Headless rendering works on macOS with `MUJOCO_GL` unset,
  256×256 RGB for both cameras.
  Implication: the oracle (task 4) at N = 100, H = 10, 4 iterations is ≈ 4000 steps ≈ 6 s per
  decision single-threaded.
- **2026-09-26 Final-state-only cost makes the closed loop wander near the goal** (toy 2D point,
  obstacle between start and goal, H = 10, N = 300, K = 30, 5 iterations, 2 executed steps per plan,
  30 seeds). Cost = distance of the *final* imagined state: reached the goal in 5/30 seeds, the rest
  hover about 1.2 away until the 30-step limit. Cost = *mean* distance over all imagined states
  (`costs.mean_goal_distance_cost`): 30/30, in 7–8 steps; closest approach to the obstacle 1.39
  (radius 1.0). Why: with a final-only cost every plan arrives "at step H", so the executed first
  actions are almost unconstrained and their noise outweighs the pull toward the goal.
  **Relevance for Level E:** "final latent vs goal latent" is the same final-only shape, so expect
  the same wandering on the real task; compare both costs there. (V-JEPA 2-AC's exact energy still to
  be checked in the paper.)
- **2026-09-26 The CEM tests were checked before `cem_plan` existed.** Against a throwaway textbook
  CEM (in Claude's scratchpad, never in the repo): all tests pass on 30 of 30 seeds on CPU and MPS,
  so a failure means a bug, not a flaky test. Six deliberately broken versions each fail the
  intended test: no clipping, init_mean ignored, highest-cost elites, no std floor, trajectory
  without z0, one dynamics call per sample.
- **2026-10-01 M1 state save/restore is exact** (`M1Adapter.get_state/set_state`, 188 values on
  the Grade E scene): MuJoCo's full integration state + M1's controller targets / yaw / gripper +
  reward stage flags + step counters. Replaying 15 random actions after a restore is bit-identical,
  in the same env and in a clone. Needed one fix: after `mj_step`, MuJoCo's positions (site_xpos,
  contacts) still describe the previous substep; `get_state()` refreshes them (`mj_forward`) so the
  live and restored runs start from identical data.
- **2026-10-01 V-JEPA 2 ViT-L (64 frames, 256 px, 8192 tokens) encode time on the Mac** (Apple M5):
  1.0 s per clip on MPS in fp16, 3.9 s in fp32, 12.8 s on CPU. So closed-loop JEPA control is
  feasible locally if the clip is re-encoded once per plan, not per imagined step.
- **2026-10-01 Oracle CEM (task 4), first full run**, Grade E scene, 20 seeds, CEM H=8 N=64 K=8
  4 iterations, 2 executed steps per plan, yaw pinned to 0 (`data/runs/oracle/`):
  - reach: **20/20** in 26 steps on average (scripted expert: 15/20 within the 40-step limit; random: 0/20)
  - lift (old grasp test "within 2 cm of centre"): **11/20**; all 20 grasped, the 9 failures held the cube and
    never lifted it (`lift_oracle_cem_v1_centred_only.csv`). Scripted expert 20/20 in 55 steps.
  - place: stopped early (being rerun with the fixes below). Scripted 20/20 in 104 steps, random 0/20.
    One trial episode (seed 0) placed the cube in 72 steps; it dropped it from carry height onto the target.
  - Planning time: 2-6 s per plan (64 x 8 x 4 = 2048 real MuJoCo steps), up to ~35 s with the machine loaded.
- **2026-10-01 Grasp failures the oracle uncovered, and fixes** (the planner exploits any loophole in the cost):
  1. M1's `grasped` flag is also true for a pinch on one edge -> require a centred grasp. Too weak:
  2. fingers nearly shut on an edge within 2 cm of centre still counted -> a grasp now needs the fingers
     > 3 cm apart (cube body: 4.46-4.6 cm in M1's expert data; closed on nothing ~0.3 cm). New task feature
     `gripper_width`.
  3. with yaw pinned, some cube yaws gave a corner-to-corner grasp (fingers 5.8 cm apart) that couldn't lift
     -> the planner now uses the yaw action, and the costs penalize the angle to the nearest cube face
     (new task feature `grasp_yaw_error`, M1's own wrap).
  After 2+3, the failing seeds 1 and 3 lift (73 and 53 steps). The 20-seed lift/place rerun is still to do.
  Costs that only score the *final* state of M1's flags can't see these failures: every plan scores the
  same (best 1.93 vs mean 2.05), and CEM's mean then drifts in an arbitrary direction.
- **2026-10-01 State MLP (task 5)**, 60 Grade E episodes (7919 transitions), 12 held out, run with the
  layout *before* the two new features (`data/runs/state_mlp/`):
  - one-step validation error 0.113-0.127 (normalized) vs 0.168 for "nothing changes"
  - **checkpoint choice matters:** the epoch with the best one-step error (14) predicts 10 steps ahead
    worse (gripper 5.5 vs 4.1 cm) and planned worse (reach 5/20 vs 11-12/20) than late epochs. Now chosen by
    held-out 8-step rollout error (epoch 57: 2.7 cm). Kuba's `best.pt` is chosen by one-step error.
  - error by horizon (epoch 57): gripper 0.5 / 1.8 / 3.4 / 4.1 cm and object 0.4 / 1.2 / 2.0 / 2.4 cm
    after 1 / 4 / 8 / 10 steps
  - CEM with the MLP (same CEM settings as the oracle): reach 12/20, lift 2/20, place 4/20
  - **model exploitation:** 44 of 267 audited plans counted on a grasp that didn't happen in the real
    sim; contact events are what the MLP predicts worst, and the planner aims for imagined grasps.
- **2026-10-01 Rough JEPA dynamics model trained locally with M2's own pipeline** (`experiments/jepa_pipeline.py`;
  M2's scripts unchanged except encoding, which runs on MPS):
  60 episodes -> 600 transitions + 720 action branches -> 1920 V-JEPA clips (1.2 s/clip on MPS fp16) ->
  M2's `train_dynamics --architecture split` -> `data/jepa/grade_e/dynamics_split/best.pt`.
  M2's probe: object position from the pooled latent within **0.3 / 0.7 / 0.35 cm** (x/y/z, held out), vs
  3.6 / 15.9 / 4.0 cm for a constant guess, so the latent does carry where the cube is.
  The closed-loop JEPA agent runs end to end (smoke-tested with a random checkpoint; ~1-3 s per replan).
- **2026-10-02 First closed-loop look at the JEPA agent** (visualizer, seed 0, 40 steps, cost z + p):
  the arm heads for the goal *gripper position* (above the target, where the expert's goal clip ends)
  and ignores the cube. The goal state includes proprio p, and moving the arm there is the easiest way
  to look like the goal; the latent alone would have to pull it to the cube first. One seed, so
  preliminary: the `jepa_cem` run (both variants, z+p and z-only) is still to do.
- **2026-10-02 Oracle CEM (task 4), final run with the grasp fixes** (finger width, yaw planned +
  misalignment cost; Grade E scene, 20 seeds, CEM H=8 N=64 K=8 4 iterations, 2 executed steps per plan):
  - reach **19/20** in 27 steps (scripted 15/20 within the 40-step limit, random 0/20)
  - lift **20/20** in 44 steps (was 11/20 before the fixes; scripted 20/20 in 55 steps)
  - place **18/20** in 75 steps (scripted 20/20 in 104 steps, random 0/20). Failures: seed 2 grasped,
    lifted and was still carrying the cube 31 cm from the target at the 200-step limit; seed 13 grasped but
    never lifted (the residual stuck-grasp case, 1 of 20 now).
  - 3.7 s per step on average (a plan every 2 steps), up to ~30-70 s per plan with the machine fully loaded.

- **2026-10-02 Speed benchmark (task 6)** (`data/runs/bench_speed/`, Apple M5 16 GB, random weights,
  time for one `cem_plan` call):

  | model / latent | N=300 H=10 3 iters, CPU / MPS | N=1000 H=20 5 iters, CPU / MPS | imagined trajectory |
  |---|---|---|---|
  | pooled [1024] | 0.035 / 0.020 s | 0.31 / 0.18 s | 14 MB |
  | M2's SplitDynamics on [z \| p] (1044) | 0.033 / 0.022 s | 0.32 / 0.16 s | 14 MB |
  | patch tokens [256, 1024] (per-token MLP) | 15 s / **fails** (out of GPU memory) | skipped (> 4 GB) | 3.5 GB |

  With M2's pooled choice planning is nearly free: the JEPA agent's ~0.8 s per step is almost all
  V-JEPA encoding. Tokens would make planning 60-500x slower and don't fit at N=300, H=10 on a 16 GB Mac,
  even with a mock model cheaper than a real (attention) token predictor. **For M2:** pooled is the right
  call for compute; the cost is spatial detail (see the JEPA results).

- **2026-10-02 Grasp-angle feature made smooth.** The plain angle to the nearest face wraps from
  +45 to -45 degrees between faces; it's now `(1 - cos 4*theta) / 2` (0 square, 1 at 45 degrees).
  The state MLP can't predict a wrap. (It wasn't the main cause of the result below, though.)
- **2026-10-02 State MLP retrained on the final 33-value state**: 8-step error 2.67 cm (epoch 58),
  one-step 0.126, about as good as before. **Planning got much worse, though:** reach 2/20 (was 12/20),
  lift 0/20, place 0/20. 57 of 300 audited plans counted on a grasp that never happened. Pinning
  yaw didn't change it (10 seeds: reach 0/10 pinned vs 1/10 planned). Cause: the model's **bias**. One
  step's action changes the gripper's motion by only ~1 cm (the IK controller slides towards each target),
  and the model gets that difference right (+x vs -x: 1.15 cm predicted, 1.04 cm real). But it is
  offset by ~0.4 cm per step, ~3 cm over the 8-step horizon, the same size as the 3 cm reach radius. Near
  the cube the planner believes it has arrived and stops short (min distances 3-5 cm in most seeds).
  Two models with the same average error planned 12/20 vs 2/20: what matters is *which way* the bias
  points, which average error doesn't measure. Remedy for Grade C: a multi-step (rollout) training loss.
- **2026-10-02 JEPA model, offline checks** (`experiments/jepa_cem.py`, local rough D, split architecture):
  - latent distance to the goal along M1's successful episodes, by fraction of the episode (0-20% ... 80-100%):
    4.27, 2.40, 2.18, 1.72, 0.67. It falls steadily, so the latent goal cost does measure progress here.
  - D's predicted gripper move per unit action: 0.44 / 0.50 / 0.35 cm (x/y/z), real ~0.5 cm. D responds
    to actions with about the right size; its latent changes with the action too (+ vs - differ by
    40-64% of a one-step change).
  - gripper error by horizon: 0.34 / 1.36 / 2.92 / 3.75 cm after 1 / 4 / 8 / 10 steps (about the state MLP's).
- **2026-10-02 Level E deliverable: CEM on the JEPA world model, closed loop in M1's env** (Grade E
  scene, local rough D, 5 seeds x 150 steps per variant, CEM H=10 N=300 K=30 4 iterations, 2 executed
  steps per plan; ~0.8 s per step on the Mac, nearly all of it V-JEPA): **the arm attempts the task, 0/5
  picks in both variants.**

  | goal cost | closest gripper-cube | where the arm ends | latent MSE to goal | joint MSE to goal | cost gap mean-best |
  |---|---|---|---|---|---|
  | z + p (M2's loss terms) | 31-33 cm | over the target, every seed | 6.39 -> 3.48 | 1.45 -> 0.96 | 0.27 |
  | z only (pure visual goal) | 15-25 cm | scattered | 6.39 -> 3.04 | 1.45 -> 2.34 | 0.04 |

  Expert episodes go from 4.27 to 0.67 in latent MSE. Why it fails:
  1. The goal is reachable only through a grasp, and a 10-step (1 s) imagined horizon can't see one, so
     the cube never moves in imagination.
  2. With p in the cost, the shortest way to "look like the goal" is to move the arm to the goal pose.
  3. Without p, D's predicted latent changes over 1 s are small next to the distance to the goal; plans
     barely differ in cost (gap 0.04) and CEM's choice is close to random, so the arm wanders.

  Fixes to try at Grade C: intermediate goal clips (above the cube / grasped / lifted / over the target,
  as V-JEPA 2-AC does for pick-and-place, to check in the paper); M2's reward model R as the cost; a
  longer horizon or more steps per action; patch tokens instead of the pooled latent (where things are).
  The offline checks above show D itself is usable (it reacts to actions, and the latent goal distance
  falls along successful episodes), so the bottleneck is the goal specification and horizon, not D.

## Level E status (2026-10-02)

All M3 Level E tasks are done on branch `M3_level_E`. Tests: `.venv/bin/python -m pytest controller/tests`.
Reproduce: the commands in `controller/README.md`; outputs go to `data/runs/`.

| Task | Result |
|---|---|
| 1 CEM planner | `cem.py` + `planner.py` (warm start, k executed steps, MPC with k = 1) |
| 2 toy tests | reach / obstacle / token-shaped latents + contract tests; checked against a reference and 6 bug variants |
| 3 env adapter | M1 env with exact state save/restore, goal clip, task features |
| 4 oracle | reach 19/20, lift 20/20, place 18/20 (upper bound) |
| 5 state MLP | 8-step error 2.7 cm; CEM reach 2/20 (12/20 with an earlier model of equal error): model bias + exploited grasps |
| 6 speed | pooled / M2's model: 0.02-0.03 s per plan; patch tokens: 15 s or out of memory |
| 7 evaluation | `eval.py`, CSV + plots per experiment, `experiments/visualize.py` |
| 8 actor-critic | skeleton, stable on the toy problem |
| Deliverable | CEM on the JEPA world model runs closed loop and attempts the task; 0/5 picks; analysis above |

Next (Grade C): intermediate goal clips or M2's reward model as the JEPA cost, MPC (`execute_steps=1`),
a multi-step training loss for learned models, the actor-critic, and M2's own `best.pt` when Kuba shares it.

## Questions for M1

1. **State save/restore.** Could `PickPlaceEnv` get `get_state()` / `set_state()`? MjData alone
   isn't enough: the `ArmController` keeps `q_des`, `yaw`, `gripper_open`, and `Rewards` keeps its
   stage flags (`placed`, `was_grasped`, ...), plus `step_count` and `_success_steps`. I'll
   implement it in `m1_adapter.py` for now; it breaks silently if those internals change.
2. **Goal image.** There's no goal-image function (`obs["goal"]` is the place position). What should
   the arm be doing in the goal image: at its reset pose, or holding / just released the object?
   It matters because the latent distance "sees" the arm too.
3. **Action space for datasets.** Yaw is on by default, so actions are 5-dim
   `[dx, dy, dz, dyaw, gripper]`. Will datasets keep yaw? M2's model and my planner have to match.
   Is the gripper staying binary (`> 0` = open)?
4. **Importability.** The code only imports when run from inside `simulation/`
   (`from environment import ...`). Could `simulation/` become a package, or get a `pyproject.toml`?
   The top-level names `environment` / `data_collection` are generic enough to clash.
5. **Frames.** `obs["goal"]` is in the robot base frame, `privileged_state()` is in the world frame.
   Is that intended to stay?
6. **`.gitignore` ignores every `*.md`.** Was that meant to cover package READMEs and notes too?
   `controller/README.md` and `controller/NOTES.md` are force-added (`git add -f`) so they are
   versioned. Say if NOTES.md should stay local instead.
7. **Python version.** `python3 -m venv` gives Python 3.9 on a stock Mac, where pip falls back to
   old MuJoCo / torch versions. Pin 3.12 in the README (e.g. `python3.12 -m venv .venv`)?

## Questions for M2

1. **Latent shape:** pooled `[D]` or patch tokens `[T, D]`? (Task 6 will measure what each costs
   the planner in time and memory.)
2. **Predictor:** Markov `(z_t, a_t) → z_{t+1}`, or conditioned on several past frames like
   V-JEPA 2-AC (to verify in the paper)? Either works; with history, the frame window gets packed into `z`.
3. **Framework:** PyTorch?
4. **Encoder input:** which camera(s) at Level E, and what does `encode()` expect (uint8 HWC as M1
   renders, or preprocessed)? My assumption: raw uint8, preprocessing inside `encode()`.
5. **Actions:** does the dynamics model take M1's raw 5-dim action in `[-1, 1]`?

## AI-use log

Who wrote what, for the course's AI-use declaration.

| Date | Item | Written by |
|---|---|---|
| 2026-09-26 | Repo scaffold: README, CLAUDE.md, `.gitignore`, `pyproject.toml`, M1 submodule | Claude, from Calle's spec |
| 2026-09-26 | `MERGE.md`, `NOTES.md`, package `README.md` | Claude, from Calle's spec |
| 2026-09-26 | `interfaces.py`, `config.py`, `tests/conftest.py` | Claude, from Calle's spec |
| 2026-09-26 | `cem.py`: `cem_plan()` signature and docstring (body raises `NotImplementedError`) | Claude |
| 2026-10-01 | `cem.py`: `cem_plan()` body (Calle delegated all of M3 Level E to Claude on 2026-10-01) | Claude |
| 2026-09-26 | `planner.py`: `CEMPlanner` loop (queue, warm start, `execute_steps`) | Claude |
| 2026-09-26 | `tests/test_cem.py`, `tests/test_planner.py`, `tests/test_costs.py` | Claude |
| 2026-09-26 | `dynamics/toy.py`, `costs.py` | Claude |
| 2026-09-26 | Throwaway reference CEM used only to calibrate the tests (scratchpad, not in the repo, not shown to Calle) | Claude |
| 2026-09-26 | Moved the package into the shared repo as `controller/`, adapted config paths and docs | Claude |
| 2026-10-01 | `adapters/m1_adapter.py`, `tests/test_m1_adapter.py`, task-feature layout in `interfaces.py` | Claude |
| 2026-10-01 | `dynamics/oracle.py`, `agents.py`, `eval.py`, staged costs in `costs.py`, `experiments/oracle_cem.py` | Claude |
| 2026-10-01 | `dynamics/state_mlp.py`, `experiments/state_mlp.py` | Claude |
| 2026-10-01 | `adapters/m2_adapter.py`, `experiments/encode_features.py`, `experiments/jepa_pipeline.py` (runs M2's scripts unchanged) | Claude |
| 2026-10-01 | `tests/test_oracle_and_eval.py`, `tests/test_learned_models.py`, more `tests/test_costs.py` | Claude |
| 2026-10-01 | `actor_critic.py` + `tests/test_actor_critic.py` (task 8 skeleton) | Claude |
| 2026-10-01 | `experiments/bench_speed.py`, `experiments/jepa_cem.py`, `experiments/demo.py` | Claude |
| 2026-10-02 | Grasp fixes (finger width, yaw planning + alignment cost), 13th/14th task features | Claude |
| 2026-10-02 | Visualizer: `visual.py`, `experiments/visualize.py` (replaces `demo.py`), imagined-path replay in `agents.py`, `tests/test_visual.py` | Claude |
| 2026-10-02 | Final Level E runs (oracle, state MLP, JEPA closed loop, speed benchmark) and their analysis | Claude |
