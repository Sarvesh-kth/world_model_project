# models

The trained models the controllers load, tracked through Git LFS (`git lfs pull --include="simulation/models/**" --exclude=""`).

```
world_model/            the world model package: models/{readout,readout_p,reward,dynamics,obstacle,cost_to_go}.pt,
                info.json (the config and rest height of the data it was trained on), features/{pca.npz, meta.json}
                (the PCA bases and encoder settings, so the controller encodes frames the way the data was encoded)
sac/            the simulator-trained SAC for the MPC's scenes: models/sac_best.zip, pipeline.json (what it was
                trained with, read by the controller), models/validation_history.json
```

`world_model` is `data/episodes/attempts/world_model` packaged by `train.sh`: 401 full-mix episodes, V-JEPA 2
latents on a 16x16 grid, Q R D trained 60 epochs, the obstacle model on the obstacle-only maps and the
cost-to-go on synthetic scenes. `world_model/README.md` describes each model, `rl/README.md` the SAC.

The controllers take `--models-run models/world_model` (the default) and `--baseline-run models/sac` for
`rl_q` / `rl_true`. A run folder works too: `--models-run data/episodes --tag world_model`.
