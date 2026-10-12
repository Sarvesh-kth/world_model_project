# data

Run folders. Tracked through Git LFS: the two inputs below, the v3 run folder (manifest, latents in 1.5 GB
parts, bases, maps) and the hack episodes (`episodes_hack/episodes_raw.tar`), about 3.3 GB, so a clone can
retrain (the root README's Install section pulls and unpacks them). Everything else is made by the collector,
`encode.py`, `train.py` and the controllers and stays local. The models the controllers run with live in
`../models/<tag>/` (tracked through Git LFS too), packaged from a run's `attempts/<tag>/` and `features/`.

Tracked:

  rl_baseline_v1/       the frozen exact-state SAC (models/sac_best.zip, pipeline.json with its layout, demos,
                        evaluations), for the rl_true / rl_q controllers only
  obstacles_v2/         collection.json + scene_settings.json: the six held-out obstacle layouts control_clutter replays

Made by the chain (local):

  episodes_v3/          the 401 full-mix episodes (episodes_raw/), the run folder (manifest.json), the latents
                        (features/latents.npy, pca.npz, meta.json, obstacle_maps.npz, cost_to_go_*.npz) and the
                        models of every tag (attempts/v3, v3b, v3c). models/v3c = attempts/v3c packaged.
  episodes_hack/        150 episodes of configs/hack_mix.yml (closes mid-travel, releases short of the goal, misses),
                        the data that retires the gripper envelope and the release rule; encoding not finished
                        (features/progress.json, continue with encode --resume), see ../../NEXT.md
  control_*/            controller runs (frames, gifs, per-step csv per episode); their summaries go to ../../results/.
                        control_v5*_02 are the 0-2 obstacle runs of the STATUS table; the v3*/v4* runs before them
                        can be deleted (NEXT.md lists the command)

The runs made before 2026-10-12 (the 8x8 latent, a biased cube placement) were moved to
../../../world_model_project_data_archive/2026-10-12/ and can be deleted.
