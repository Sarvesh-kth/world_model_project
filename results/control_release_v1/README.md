# control_release_v1

The export of the imagined-SAC experiment, kept in Git so the results can be checked without rerunning it:

```
policies/train_empty/, policies/train_mixed/   the two imagined SACs: models/sac_best.zip (LFS), pipeline.json, imagined_training.json
reports/calle_sac_scratch_v1/                  per-episode results, the summary and the figures (with_mpc/ has the SAC + MPC ones)
reports/calle_mpc_matched_empty30/             the mpc on the same 30 empty scenes
reports/calle_mpc_matched_obstacles30/         and the same 30 obstacle scenes
reports/mpc_allpaths_seed20496039/             one mpc episode recorded with every CEM candidate of every round
runs/<name>/                                   settings, result.json and steps.csv per episode, the replay arrays (trajectory.npz, plans.npz)
runs/mpc_counterfactual_seed20496039/          the four rank-selection episodes (controller.mpc_counterfactual), summary.json has their outcomes
inventory.json                                 every file with its hash and where it came from
```

The gifs were not exported; a replay is rendered from the saved simulator states without any model, from
`simulation/` (first `git lfs pull --include="results/control_release_v1/**/*.npz" --exclude=""`):

```bash
python -m controller.replay ../results/control_release_v1/runs/calle_mpc_matched_obstacles30/episodes/mpc/seed_20496039 --video --models-run models/world_model --tag '' --speed 0.5
python -m controller.replay ../results/control_release_v1/runs/mpc_allpaths_seed20496039/episodes/mpc/seed_20496039 --video --all-paths --models-run models/world_model --tag ''
```

To run one of the two policies again: `python -m controller.control_pipeline --methods sac_imagined
--imagined-run ../results/control_release_v1/policies/train_empty --random-ends --headless --out data/...`
(the zip comes through LFS: `--include="results/control_release_v1/policies/**"`).
