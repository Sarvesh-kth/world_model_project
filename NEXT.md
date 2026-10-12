# What could be done next

State on 2026-10-13: the latent MPC avoids obstacles with learned models only (the obstacle model `O` on the
obstacle-only depth map, optionally the cost-to-go `C`), 16/20 placements on 0-2 obstacle scenes with zero
contacts against 14/20 and 233 contact steps for the old geometric map (STATUS.txt, top section). Four
planner rules remain. Everything below runs from `simulation/` with the venv active.

## 1. Retire the gripper envelope and the release rule (data collected, chain half done)

`data/episodes_hack` has 150 episodes (20,455 states) of `configs/hack_mix.yml`: fingers shut mid-travel
toward the cube (`travel_close`), the cube let go 5-14 cm short of the goal and then placed properly
(`short_release`), and misses. The encoding was started with the v3 bases and stopped. On another machine
(the cluster) first pull the data through LFS and unpack it (root `README.md`, Install), then:

```bash
python -u -m world_model.prepare_episodes --episodes data/episodes_hack/episodes_raw --run data/episodes_hack   # only on a fresh clone
python -u -m world_model.encode --run data/episodes_hack --pca data/episodes_v3/features/pca.npz --resume   # ~1.5 h on the RTX 4060, ~30 min on the cluster
python -u -m world_model.merge_runs --runs data/episodes_v3 data/episodes_hack --out data/episodes_v4 --overwrite
bash world_model/run_combined.sh data/episodes_v4 v4 40 retrain          # Q, R, D, O, C -> attempts/v4 (D is the long stage)
# package (what cluster_round.sh does): models/v4/{models/*.pt, info.json, features/{pca.npz, meta.json without keys}}
B="--models-run models/v4 --headless --test-episodes 20 --random-ends --obstacles 0 2 --methods jepa_mpc"
python -u -m controller.control_pipeline $B --out data/control_v6c_02
python -u -m controller.control_pipeline $B --no-grasp-prior --out data/control_v6n_02
python -u -m controller.control_pipeline $B --no-release-rule --out data/control_v6r_02
python -u -m controller.control_pipeline $B --no-grasp-prior --no-release-rule --no-width-gate --out data/control_v6x_02
```

If `--no-grasp-prior` holds its rate with the v4 models, make it the default (planner.py `--no-grasp-prior`
becomes `--grasp-prior`); same for the release rule. The width rules cost nothing in the ablation (14/20 vs
15/20, noise) and can be switched off now; the second one covers Q's 12-step held lag, which comes from the
16-frame time-mean latent and has no data fix.

## 2. Grasping, the failure every variant shares

Seeds 17, 20, 23 fail in every variant: the hand never grasps beside a box or at the table edge, Q 1-2 cm off.
Options: more `lateral` and `miss` episodes at those spots; a readout trained with the planner's own
episodes (the controller saves frames, online latents and the true state per episode, a script that turned them into a run
existed before and was removed as unused); or a finer static grid for Q only.

## 3. Variance

The same 20 seeds move by about 10 % between runs of the same code and models (the CEM amplifies float
differences). Quote 20-scene rates, and for a decision between two variants run both twice or use 40 scenes.
A deterministic torch setting (`torch.use_deterministic_algorithms`, fp32 encoder) would make runs
repeatable at a cost in speed; worth it for ablations.

## 4. Speed (now ~1.8 Hz with the GPU to itself)

Per step: render + maps 36 ms, V-JEPA encode of two 16-frame clips ~250 ms, planner 180-270 ms, C 12 ms.
Cheapest cuts, each a flag or a small change, none measured for success rate yet: 8 wrist frames
(`encode.py --wrist-clip-frames 8`, needs a re-encode and retrain), 2 CEM rounds (`--iterations 2`),
population 48, encode every second step and plan on the last latent in between, R retrained as penalties
only (its task heads are unused since `--reward-source readout`). 10 Hz needs a different encoder.

## 5. The cost-to-go, honestly

`C` is learned, but its labels are a grid planner on synthetic scenes: a distilled planner. On 0-2 obstacles
it is within noise of the plain obstacle model (15/20 vs 16/20) and it fixed the one failure the plain model
has that the map variant does not (parking beside an obstacle, 1-3 obstacle scenes). Keep it optional unless
3-obstacle scenes come back; then measure both on 40 scenes.

## 6. Three obstacles and the clutter scenes

The 1-3 obstacle runs were stopped at the user's request (partial numbers in STATUS.txt). `control_clutter`
(the six held-out layouts) has not been run with the learned obstacle model yet:
`python -m controller.control_clutter --methods jepa_mpc --headless --out data/control_v5_clutter`.

## 7. Disk

`simulation/data` is 13 GB. Safe to delete: the superseded controller runs and their summaries
(`data/control_v3*`, `data/control_v4*`, `data/control_100*`, `data/control_dbg14`, `data/control_smk1` and
`results/` of the same names) once STATUS.txt has what is needed from them; the archive
`../world_model_project_data_archive/2026-10-12/` (11 GB). Keep `data/episodes_v3` (the training run) and
`data/episodes_hack` (step 1).

```bash
rm -rf data/control_v3* data/control_v4* data/control_100* data/control_dbg14 data/control_smk1 ../results/control_v3* ../results/control_v4* ../results/control_100*
```

## 8. Commit

Nothing is committed. Through Git LFS go `simulation/models/v3c` (45 MB) and the training data: the v3
run folder (`manifest.json` 457 MB, `features/latents.npy.part00/01` 2.15 GB, bases and maps) and
`data/episodes_hack/episodes_raw.tar` (581 MB), about 3.3 GB in all. `git lfs install` once, then
`git add -A && git commit && git push`. GitHub's free LFS quota is 1 GB of storage and 1 GB of bandwidth a
month, so the push needs a data pack (50 GB for 5 USD a month) or another LFS host; without it the push is
refused. `STATUS.txt` has the day's log, the READMEs describe the current code.
