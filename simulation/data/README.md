# simulation/data

Run folders. Nothing here is tracked except the two inputs below; everything else is made by the
collector, `encode.py`, `train.py` and the controllers, on this machine or on the cluster
(`../CLUSTER.md`, `../cluster_round.sh`).

Tracked:

  rl_baseline_v1/       the frozen exact-state SAC (models/sac_best.zip, pipeline.json with its layout, demos,
                        evaluations), for the rl_true / rl_q controllers only
  obstacles_v2/         collection.json + scene_settings.json: the six held-out obstacle layouts control_clutter replays

Made by the v3 chain (local):

  episodes_v3/          the full-mix episodes (episodes_raw/), the run folder (manifest.json), the latents
                        (features/latents.npy, pca.npz, meta.json) and the models (attempts/v3/). To run the
                        controllers on another machine, copy attempts/v3/ and features/{pca.npz,meta.json}.
  control_*/            controller runs (frames, gifs, per-step csv per episode); their summaries go to ../../results/

The runs made before 2026-10-12 (full_v2, obstacles_v2 frames, episodes_v2, planner_v1, combined_v2, the
control_v3..v9 runs, the old results/) used the 8x8 latent and a biased cube placement; they were moved to
../../../world_model_project_data_archive/2026-10-12/ and can be deleted.
