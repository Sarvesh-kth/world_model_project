# data

Run folders. Tracked through Git LFS: the two inputs below, the v3 run folder (manifest, latents in 1.5 GB
parts, bases, maps) and the hack episodes (`episodes_hack/episodes_raw.tar`), about 3.3 GB, so a clone can
retrain (the root README's Install section pulls and unpacks them). Everything else is made by the collector,
`encode.py`, `train.py`, `rl_baseline.py` and the controllers and stays local. The models the controllers
run with live in `../models/` (tracked too).

Tracked:

```
episodes_v3/          the 401 full-mix episodes: the run folder (manifest.json, scene_settings.json), the latents
                      (features/latents.npy in parts, pca.npz, meta.json, obstacle_maps.npz, cost_to_go_*.npz);
                      the frames (episodes_raw/) and the trained attempts/ are local. models/v3c = attempts/v3c packaged
episodes_hack/        150 episodes of configs/hack_mix.yml (closes mid-travel, releases short of the goal, misses), as one tar;
                      the data that retires the gripper envelope and the release rule, not encoded yet (../../NEXT.md)
rl_baseline_v1/       the old fixed-layout SAC (models/sac_best.zip, pipeline.json, demos, evaluations)
obstacles_v2/         collection.json + scene_settings.json: the six held-out obstacle layouts control_clutter replays
```

Local, made by the commands in the READMEs:

```
sac_priv_*/           runs of rl.rl_baseline for the MPC's scenes (sac_priv_wg1 is the packaged models/sac_priv_v1)
sac_priv_eval_<n>/    the three controllers on <n> obstacles, 10 scenes (sac_priv_eval.sh); summaries in ../../results/
control_*/            other controller runs (frames, gifs, per-step csv per episode); summaries in ../../results/
```
