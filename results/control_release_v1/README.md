# Controller results release

Selected SAC policies: policies/train_empty and policies/train_mixed each contain
models/sac_best.zip, pipeline.json and imagined_training.json. These folders work
with controller.control_pipeline --methods sac_imagined --imagined-run <folder>.
Frozen D/Q/R/O/PCA/encoder metadata remain at simulation/models/v3c (already in Git LFS).

Reports include actual episode outcomes, settings, CSVs and any generated plots.
Runs may be partial: consult inventory.json and their recorded completion flags.
Replay copies contain evaluation states and imagined paths, not training examples.
trajectory.npz omits z; original files remain unchanged. Original source hashes in
metadata describe originals; inventory hashes describe these exported copies.

From simulation/, regenerate a replay, with no controller evaluation or training:

    ../.venv/bin/python -m controller.replay ../results/control_release_v1/runs/calle_mpc_matched_obstacles30/episodes/mpc/seed_20496039 --video --models-run models/v3c --tag '' --speed 0.5

For runs/mpc_allpaths_seed20496039, add --all-paths to draw every recorded proposal.
Counterfactual variant episodes are under runs/mpc_counterfactual_seed20496039/<variant>/episodes/mpc/seed_20496039.

No raw frames, GIF/MP4/video archives, training datasets, replay buffers, virtualenvs
or cached JEPA latents are included. Source scripts/notebooks are in the repository.
