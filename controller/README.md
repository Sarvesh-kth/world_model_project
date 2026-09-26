# controller (M3 – Control)

The controllers that act on top of the JEPA world model: a CEM planner now, MPC and an
actor-critic policy later. Owner: Calle (M3).

The planner never sees the environment or the world model directly. It only gets
`dynamics_fn(z, a) -> z_next` and `cost_fn(trajectory, z_goal) -> cost` (see `interfaces.py` §2).
So the same code plans with a toy function, the MuJoCo simulator itself, a learned model, or
M2's JEPA model from `jepa_model/`.

| File | What |
|---|---|
| `interfaces.py` | §1 what M3 needs from M1 (`simulation/`) and M2 (`jepa_model/`), §2 contracts M3 owns |
| `config.py` | CEM hyperparameters, output paths, device choice, where M1's code is |
| `cem.py` | `cem_plan()`: one CEM search over action sequences |
| `planner.py` | `CEMPlanner`: the loop around `cem_plan` (warm start, execute k steps, replan) |
| `costs.py` | cost functions over imagined trajectories |
| `dynamics/` | models to plan with: `toy.py` (2D point); later the simulator oracle, a state MLP, a mock latent model |
| `adapters/` | `m1_adapter.py` (next): M1's `PickPlaceEnv` behind the `EnvAdapter` interface, the only file importing `simulation/` |
| `experiments/` | runnable experiments, `python -m controller.experiments.<name>` |
| `tests/` | pytest suite |
| `NOTES.md` | decisions, results, questions for M1/M2, AI-use log |

## Setup

From the repo root. Needs Python 3.10+; developed on 3.12. On a stock Mac `python3` is 3.9, so name the version:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r controller/requirements.txt
# or without uv:  python3.12 -m venv .venv && .venv/bin/python -m pip install -r controller/requirements.txt
```

`controller/requirements.txt` pulls in the repo's `requirements.txt` (M1's simulation deps) and
adds torch and pytest, so one venv runs both `simulation/` and `controller/`.

## Running

Unlike `simulation/` (run from inside that folder), everything here runs **from the repo root**,
so `controller` (and later `jepa_model`) import as packages:

```bash
.venv/bin/python -m pytest controller/tests            # all tests
.venv/bin/python -m pytest controller/tests -m "not slow"
.venv/bin/python -m controller.experiments.<name>
```

## Git workflow

- Work on the `m3-control` branch, never directly on `main`.
- Get teammates' latest work: `git fetch origin && git merge origin/main`.
- When a task is done: `git push`, then open a pull request `m3-control` → `main`.
- The repo's `.gitignore` ignores all `*.md` files. This README and `NOTES.md` were added with
  `git add -f` and are tracked now. A **new** `.md` file needs `git add -f` too.
