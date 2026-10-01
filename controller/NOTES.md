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
