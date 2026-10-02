# controller (M3 – Control)

The controllers that act on top of the world model: a CEM planner (Level E), with MPC and an
actor-critic policy to follow (Grade C). Owner: Calle (M3). Level E was built on the
`M3_level_E` branch, which also merges `main` (M1) and `M2_Kuba` (M2).

## Level E results (Grade E scene, details and analysis in `NOTES.md`)

| Planner + world model | Reach | Lift | Pick-and-place | Notes |
|---|---|---|---|---|
| M1's scripted expert (reference) | 15/20* | 20/20 | 20/20, 104 steps | reads the true sim state |
| CEM + the simulator itself (oracle) | 19/20 | 20/20 | **18/20, 75 steps** | upper bound; 2-6 s per plan |
| CEM + learned state MLP | 2/20 | 0/20 | 0/20 | model bias ~0.4 cm/step; plans aim for imagined grasps |
| **CEM + JEPA world model (Level E)** | - | - | 0/5 (both goal costs) | attempts the task; z+p cost parks the arm at the goal pose, z-only wanders |
| random | 0/20 | 0/20 | 0/20 | |

\* within the 40-step reach limit; the expert hovers before descending.

The JEPA model itself checks out (it reacts to actions, the latent goal distance falls along successful
episodes, the latent locates the cube within ~0.5 cm). What fails is the goal specification over a 1 s
horizon. Planning with M2's pooled latent costs ~0.02 s per plan; patch tokens would not fit in 16 GB.

## How it fits together

```
 M1  simulation/environment     M3  controller/                          M2  simulation/world_model
 ───────────────────────        ──────────────────────────────           ────────────────────────
 PickPlaceEnv (MuJoCo)  ◄──────  adapters/m1_adapter.py                   V-JEPA 2 encoder (frozen)
   reset / step / render           exact state save/restore, goal clip   D: (z, p, a) -> (z', p')
        │                          task features (sim ground truth)            ▲
        │ obs, frames                     │                                     │
        ▼                                 ▼                                     │
 agents.py: obs -> planner state z ->  planner.py (CEMPlanner)  ->  action  adapters/m2_adapter.py
                                         └─ cem.py: cem_plan(z0, goal, dynamics_fn, cost_fn)
                                                       ▲              ▲
                       dynamics/ : toy | oracle (the sim itself) | state MLP | JEPA D (M2)
                       costs.py  : goal distance | staged state costs | latent goal cost
```

The planner never sees the environment or a model directly: only `dynamics_fn(z, a) -> z_next`
and `cost_fn(trajectory, goal) -> cost` (`interfaces.py` §2). The same `cem_plan` drives a 2D toy
point, MuJoCo itself, a learned state model, and M2's JEPA model. Each world model comes with a
planner state:

| World model | Planner state z | Cost | Agent |
|---|---|---|---|
| toy point (`dynamics/toy.py`) | 2D point | goal distance | tests only |
| oracle (`dynamics/oracle.py`) | 12 task features + full sim state (200) | staged state costs | `OracleCEMAgent` |
| state MLP (`dynamics/state_mlp.py`) | 12 task features + robot/object state (32) | staged state costs | `StateMLPAgent` |
| M2's JEPA D (`adapters/m2_adapter.py`) | normalized [V-JEPA z (1024) \| proprio (20)] | latent goal cost | `JEPACEMAgent` |

## Files

| File | What |
|---|---|
| `interfaces.py` | §1 what M3 needs from M1 / M2 (observed values + assumptions), §2 M3's contracts, task-feature layout |
| `config.py` | `CEMConfig`, output paths (all under `data/`), device choice, where M1's code is |
| `cem.py` | `cem_plan()`: one CEM search |
| `planner.py` | `CEMPlanner`: execute k steps, warm-start, replan (`execute_steps=1` = MPC) |
| `costs.py` | goal distance, staged pick-and-place costs, latent goal cost |
| `agents.py` | random, scripted (M1's expert), oracle-CEM, state-MLP-CEM, JEPA-CEM agents |
| `eval.py` | run any agent on any env for many seeds (in parallel), CSV rows, summaries, plots |
| `actor_critic.py` | Grade C skeleton: actor, critic, lambda-returns, one imagined-rollout update |
| `dynamics/toy.py`, `oracle.py`, `state_mlp.py` | world models to plan with |
| `adapters/m1_adapter.py` | M1's env behind `EnvAdapter`; the only file importing M1 code |
| `adapters/m2_adapter.py` | V-JEPA 2 encoder, 64-frame clip buffer, M2's D as `dynamics_fn`; the only file importing M2 code |
| `experiments/oracle_cem.py` | task 4: CEM with the simulator as model, 3 stages × 20 seeds |
| `experiments/state_mlp.py` | task 5: train the state MLP, multi-step error, CEM, exploitation audit |
| `experiments/bench_speed.py` | task 6: planning time / memory vs latent shape, N, H, iterations |
| `experiments/jepa_pipeline.py` | M2's own Grade E recipe run locally → a rough JEPA dynamics model |
| `experiments/encode_features.py` | V-JEPA features on any device, in M2's file format (M2's script needs CUDA) |
| `experiments/jepa_cem.py` | the Level E deliverable: offline checks + closed-loop CEM on the JEPA model |
| `experiments/visualize.py` | watch any agent in MuJoCo (live viewer or MP4), with the planner's imagined paths drawn |
| `visual.py` | draws plans, executed path and goal into a MuJoCo scene (viewer and videos) |
| `tests/` | pytest suite (fast by default; `-m slow` / M1-dependent ones are marked) |
| `NOTES.md` | decisions, results, questions for M1/M2, AI-use log |

## Setup

From the repo root. Needs Python 3.10+, developed on 3.12. On a stock Mac `python3` is 3.9, so name the version:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r controller/requirements.txt
# or:  python3.12 -m venv .venv && .venv/bin/python -m pip install -r controller/requirements.txt
```

`controller/requirements.txt` = M1's `requirements.txt` + M2's `requirements-model.txt` +
torch, torchvision, pytest. The first JEPA run downloads V-JEPA 2 ViT-L (~1.3 GB) from Hugging Face.

## Running (from the repo root)

```bash
.venv/bin/python -m pytest controller/tests                  # everything (~15 s)
.venv/bin/python -m pytest controller/tests -m "not slow"    # skip the slower ones

.venv/bin/mjpython -m controller.experiments.visualize --agent oracle --viewer  # watch it live (macOS: mjpython)
.venv/bin/python -m controller.experiments.visualize --agent jepa              # or as an MP4
.venv/bin/python -m controller.experiments.oracle_cem --seeds 20 --workers 6     # task 4, ~1-1.5 h on an M5
.venv/bin/python -m controller.experiments.state_mlp                             # task 5, ~2 min
.venv/bin/python -m controller.experiments.bench_speed                           # task 6, ~5 min
.venv/bin/python -m controller.experiments.jepa_pipeline                         # rough JEPA D, ~45 min
.venv/bin/python -m controller.experiments.jepa_cem --seeds 5                    # Level E, ~25 min
```

Results land in `data/runs/<experiment>/` (CSV, `summary.json`, PNG plots); checkpoints in
`data/checkpoints/` and `data/jepa/grade_e/`. All of `data/` is gitignored.

## Visualizer

```bash
.venv/bin/mjpython -m controller.experiments.visualize --agent oracle --viewer     # live, like M1's play.py
.venv/bin/python   -m controller.experiments.visualize --agent jepa --max-steps 120  # MP4 + last frame PNG
```

On top of M1's scene it draws what the planner imagines (all world models predict the gripper):
**orange thick** = the plan CEM chose (imagined gripper path over the horizon), **orange faint** =
runner-up elite plans, **blue dots** = where the gripper actually went, **green ball** = the goal
gripper position (JEPA agent). A text panel shows step, stage, grasp and the last planning time.
Options: `--agent scripted|oracle|state_mlp|jepa|random`, `--task reach|lift|place`, `--seed`,
`--speed` (live pace), `--view overview|static` and `--size` (video), `--no-overlay`. The viewer stays
open after the episode until you close it. While the oracle plans (2-6 s) the arm pauses: that's
the planner thinking, not a hang.

## Manual testing checklist (Level E)

1. **Tests:** `.venv/bin/python -m pytest controller/tests`. All should pass.
2. **Watch the reference:** `.venv/bin/mjpython -m controller.experiments.visualize --agent scripted --viewer`.
   M1's expert places the cube in ~100 steps.
3. **Watch the planner with a perfect model:** `--agent oracle --viewer` (a few minutes). Expect the
   short orange plan to lead the gripper to the cube, a grasp, a carry and a drop on the target.
4. **Learned state model:** `--agent state_mlp` (after `experiments.state_mlp`). Expect a
   noticeably worse arm: it reaches, sometimes grasps, rarely places.
5. **Level E on the JEPA model:** `--agent jepa --viewer` (after `experiments.jepa_pipeline`; ~1 s per
   replan). This is the "arm attempts pick-and-place with CEM on the JEPA world model" deliverable.
   Watch where the orange plan goes compared to the cube and the green goal ball.
6. **Compare with M2's own model:** once Kuba shares `best.pt`, run
   `.venv/bin/python -m controller.experiments.jepa_cem --checkpoint path/to/best.pt`.

## Git workflow

- Never commit to `main`. M3 branches: `m3-control` (scaffold), `M3_level_E` (Level E, includes
  `main` and `M2_Kuba`).
- Teammates' updates: `git fetch origin && git merge origin/main` (and `origin/M2_Kuba`).
- Push a branch: `git push -u origin M3_level_E`, then open a pull request into `main`.
- The repo's `.gitignore` ignores all `*.md`: `README.md` and `NOTES.md` here are tracked; a new
  `.md` file needs `git add -f`.
