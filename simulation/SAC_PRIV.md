# Simulator-trained SAC against the MPC (branch `sac_priv_testing`, 18 October 2026)

The comparison row the MPC was missing: a SAC that learns from the simulator's own reward and penalties
(not from R, D or imagination), sees the obstacles through the same obstacle-only depth map the MPC's
obstacle model reads, and runs on Q's cube estimate through the cameras at test time (`rl_q`). The MPC
is untouched; it is only run. Nothing here needs the old `rl_baseline_v1` or the imagined SACs.

Inputs of the policy (`rl/task_control.py`): the 41 numbers of the old SAC (robot state, cube from Q,
goal, held from Q with the controllers' width gates, the reward memory) plus the 9x9 window of the 16x16
obstacle-only map around the hand and the map pooled to 8x8 (147 numbers), 188 in all. No image.
Training: 500 scripted demonstrations and the 401 recorded `episodes_v3` episodes in the replay buffer
(their rewards replayed from the recorded contacts and proximity), behaviour cloning, then SAC with the
simulator reward (placement +15, collision 0.5, proximity 0.3, table 0.2, drop 1, fail 10 per step/event),
1 cm noise on the cube it sees, validation on 10 empty + 20 obstacle scenes every 20k steps.
The packaged policy is `models/sac_priv_v1` (the 20k checkpoint, 20/30 validation placements: 9 empty,
11 obstacles). `rl/README.md` has the training details and flags.

## Evaluate on the cluster

From the repository root after the usual setup (`README.md`, Install; the world models through LFS):

```bash
git fetch origin && git checkout sac_priv_testing
git lfs pull --include="simulation/models/**" --exclude=""
cd simulation && source ../.venv/bin/activate
```

The planner first, then the policy on the same scenes (`--resume` adds a method to a run folder), 20
scenes per obstacle count (0, 1, 2, 3). Each MPC episode takes 1-3 minutes on a free GPU, each `rl_q`
episode about one (V-JEPA every step), so 20 scenes x 4 counts x 2 controllers is about 5 hours:

```bash
bash sac_priv_eval.sh mpc 20
bash sac_priv_eval.sh rl_q 20 models/sac_priv_v1
python sac_priv_table.py          # the markdown table from results/sac_priv_eval_<n>/
```

Scenes are seeds `20494010 + i` (the pipeline default), the first ten are the local run's. For the
cluster's earlier matched seeds (`20495010...`) add `--seed 20475010` to both control_pipeline calls in
the script. Per-episode results, steps and candidates land in `../results/sac_priv_eval_<n>/`, frames and
gifs in `data/sac_priv_eval_<n>/episodes/<method>/seed_*/` (`actual.gif`, `trajectory.npz` for
`controller.replay`). To add the exact-state diagnostic row, run the same script with `rl_true`.

## Retrain (optional, CPU only, about 1.5 h)

```bash
python -u -m rl.rl_baseline --out data/sac_priv_v2 --random-ends --obstacles 0 3 --obstacle-map both \
  --recorded-run data/episodes_v3 --cube-noise 0.01 --net 256 256 --demos 500 --rl-steps 300000 \
  --eval-every 20000 --validation-episodes 30 --validation-empty 10 --validation-obstacles 1 3 --test-episodes 30
```

`data/episodes_v3/manifest.json` is enough for `--recorded-run` (no frames or latents needed). More
demonstrations are the lever that moved validation most (200 -> 500 -> 1200 demos: 11 -> 18 -> 20 of 30
from cloning alone); the RL phase adds a few placements in the first 20-40k steps and then drifts.
`--pull` (the planner's linear distance pull as a potential term) exists but was not used in the runs here.

## What the local 10-scene run showed

See `CONTROL_RESULTS.md` (section of 18 October) for the table. In short: the MPC places 7/10 on 0-2
obstacles and 5/10 on 3 with zero contacts; the SAC places 8/10, 6/10, 7/10, 6/10 on the exact state but
only about half of that through Q, because a policy trained on the exact cube and contact flag does not
tolerate Q's centimetre of error at the grasp and its late held reading, while the MPC re-plans every
step with the same Q and recovers.
