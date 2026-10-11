# The v3 chain on the cluster, step by step

Everything runs from `simulation/` with the venv active. `bash cluster_round.sh v3 400` does all of the
steps below in one go (restartable); this is the same chain by hand, for stopping and continuing.

```bash
cd simulation
source ../.venv/bin/activate              # or: pip install -r ../requirements.txt into a Python 3.12 venv
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa    # only on a machine without EGL; otherwise leave unset
export HF_HOME=/path/with/space           # V-JEPA weights (1.2 GB) download here on the first encode
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

## 1. Collect (400 episodes, about 1.5 h with 4 workers, 5 GB)

```bash
python -u -m data_collection.collect --config configs/full_mix.yml --episodes 400 --out data/episodes_v3/episodes_raw --workers 4 --seed 1000
```

Cube and goal anywhere on the table, 0-3 obstacles, random cube yaw, the kinds of `full_mix.yml`.
Stopped early? The collector appends: run it again with the number still missing and another `--seed`
(e.g. `--episodes 150 --seed 1250`); episode folders keep counting up.

## 2. Run folder

```bash
python -m world_model.prepare_episodes --episodes data/episodes_v3/episodes_raw --run data/episodes_v3
```

Rerun it after every collection; earlier episodes keep their split, new ones are appended.

## 3. Encode (the slow step; ctrl-c and continue any time)

```bash
python -u -m world_model.encode --run data/episodes_v3                 # first time: fits the PCA bases, then encodes
python -u -m world_model.encode --run data/episodes_v3 --resume        # after ctrl-c, or after more episodes were added
```

Progress is saved every 50 states in `features/progress.json`; `--resume` continues from there. About
0.09 s per state on the cluster GPU (0.27 s on the RTX 4060); 400 episodes are about 52k states. When it
is complete `features/meta.json` exists and `--resume` says so and stops.

## 4. Train Q, R, D

```bash
python -u -m world_model.train readout  --run data/episodes_v3 --tag v3 --epochs 60
python -u -m world_model.train reward   --run data/episodes_v3 --tag v3 --epochs 60
python -u -m world_model.train dynamics --run data/episodes_v3 --tag v3 --epochs 60 --rollout-steps 8 --task-weight 1.0 --robot-sees-z
```

`readout` first (`dynamics` needs it). Q and R can run side by side. Retrain after the data grew, or
to try other settings, with `--overwrite` on each (same tag), or use a new `--tag`. Watch the printed
`train=` / `val=` per epoch: they should fall together now; the best validation epoch is kept and printed
at the end (`reports/<stage>_training.json` has the curve, `reports/readout_validation.json` the cube MAE).

## 5. Checks and the closed loop

```bash
python -u -m world_model.eval_reward --run data/episodes_v3 --tag v3 --split test
python -u -m world_model.eval_grasp  --run data/episodes_v3 --tag v3 --split test
M="--models-run data/episodes_v3 --tag v3 --headless"
python -u -m controller.control_pipeline --methods jepa_mpc --test-episodes 20 --random-ends $M --out data/control_v3_empty
python -u -m controller.control_pipeline --methods jepa_mpc --test-episodes 20 --random-ends --obstacles 1 3 $M --out data/control_v3_obstacles
python -u -m controller.control_clutter  --methods jepa_mpc $M --out data/control_v3_clutter
```

Always `--random-ends`; the default layout is one fixed cube spot and says nothing. Summaries land in
`results/control_v3_*/summary.txt`.

## 6. What to copy back

`data/episodes_v3/attempts/v3/` (models, reports, info.json) and `data/episodes_v3/features/{pca.npz,meta.json}`
are all the controllers need (about 60 MB); plus `results/control_v3_*`. The frames and `latents.npy` stay
on the cluster.
