# simulation/data

Only the inputs the controllers and the training chain need live here. Everything else (old experiment
outputs, the pre-test1 vision_v2 encoder run, the obstacle-aware SAC runs) was moved to
../../../world_model_project_data_archive/simulation_data on 2026-10-10.

Used at run time by controller.control_pipeline / controller.control_clutter (scripted, rl_true, rl_q, jepa_mpc):

  combined_test1/       Q, R, D weights (attempts/combined/models/*.pt), PCA basis + encoder settings
                        (features/pca.npz, features/meta.json), manifest.json the models were trained on
  rl_baseline_v1/       frozen exact-state SAC (models/sac_best.zip), its demos, evaluation and pipeline.json
                        (config + layout, read by the collectors)
  obstacles_test1/      the six held-out obstacle layouts control_clutter replays (collection.json, scene_settings.json)

Training sources (combined_test1 symlinks into these; only needed to re-encode or re-merge):

  full_test1/               full task, cube jittered +-3 cm, frames + features (the PCA basis was fitted here)
  obstacles_test1/          paired empty / replay-with-obstacles / scripted-around scenes, frames + features
  episodes_test1/           data_collection.collect full mix (success, scripted, drop, collide, random)
  obstacles_test1_fullpca/  obstacles_test1 re-encoded with the full_test1 PCA basis (links to the frames)
  episodes_test1_fullpca/   episodes_test1 re-encoded with the full_test1 PCA basis

Shared through git (Git LFS for the binaries): combined_test1/attempts/combined/, combined_test1/features/{pca.npz,meta.json},
rl_baseline_v1/ and obstacles_test1/{collection.json,scene_settings.json}. `git lfs pull` fetches them after a clone;
everything else here is rebuilt by the collectors and encode.py (world_model/README.md, "Checkpoints").

Controller runs write to data/control (control_pipeline) and data/control_clutter (control_clutter);
both are overwritten on the next run, the shareable summaries go to results/<run name>.
The folder layout of a run is described in world_model/README.md.
