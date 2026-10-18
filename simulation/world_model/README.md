# world_model

From recorded episodes to the models the planners run with: the run folder, the V-JEPA 2 encoding, the
readout `Q`, the reward model `R`, the dynamics `D`, the obstacle model `O`, the cost-to-go `C`, and two
checks of what they learned.

```
prepare_episodes.py   a collected dataset -> run folder (manifest.json, train / val / test split by episode)
encode.py             V-JEPA 2 -> one latent z per state, the PCA bases; --pca reuses another run's bases, --resume continues
merge_runs.py         several runs encoded with the same bases -> one run
train.py              readout (Q), reward (R), dynamics (D); load_model, readout(), reward_terms(), imagine()
train_obstacle.py     the obstacle model O on the obstacle-only depth map
train_cost_to_go.py   the cost-to-go C on synthetic scenes
eval_reward.py        R on real latents and on latents imagined by D over 1 / 4 / 8 steps
eval_grasp.py         does the world model tell a grasp from a miss in imagination
run_combined.sh       train everything on one run, then the two checks:  bash world_model/run_combined.sh data/episodes_v3 v3c 60 [retrain]
common.py             the run folder format, json / csv helpers, the metrics Q is judged by
```

## The chain

```bash
python -m data_collection.collect --config configs/full_mix.yml --episodes 400 --out data/episodes_v3/episodes_raw --workers 4 --seed 1000
python -m world_model.prepare_episodes --episodes data/episodes_v3/episodes_raw --run data/episodes_v3
python -m world_model.encode --run data/episodes_v3                                            # the slow step; --resume to continue
python -m world_model.train readout  --run data/episodes_v3 --tag v3c --epochs 60
python -m world_model.train reward   --run data/episodes_v3 --tag v3c --epochs 60
python -m world_model.train dynamics --run data/episodes_v3 --tag v3c --epochs 60 --rollout-steps 8 --task-weight 1.0 --robot-sees-q
python -m world_model.train_obstacle   --run data/episodes_v3 --tag v3c --epochs 30            # 3 min, renders each scene's map once
python -m world_model.train_cost_to_go --run data/episodes_v3 --tag v3c                        # 15 min, synthetic data cached
python -m world_model.eval_reward --run data/episodes_v3 --tag v3c --split test
python -m world_model.eval_grasp  --run data/episodes_v3 --tag v3c --split test
```

`readout` before `dynamics` (D needs Q). `bash cluster_round.sh <tag> 400` from `simulation/` runs the
whole chain including the collection, the package and three closed-loop runs; every step restarts where it
stopped. To add episodes: collect into another folder, `prepare_episodes`, `encode --pca
data/episodes_v3/features/pca.npz`, `merge_runs --runs data/episodes_v3 data/<new> --out data/<merged>`,
train with a new tag (`../../NEXT.md`, step 2).

## A run folder

```
data/<run>/
  manifest.json          every recorded state: key, scene, split, the last 64 frame paths per camera, height map,
                         p (20 robot numbers), cube xyz, held, contacts, reward components, the action that led here, kind
  scene_settings.json    the layout and seed of every episode
  features/latents.npy   one z per state, float32 (N, 9280); pca.npz and meta.json (bases, encoder settings, block layout)
  features/obstacle_maps.npz, cost_to_go_*.npz   the obstacle-only map per scene, the synthetic cost-to-go data
  attempts/<tag>/models/{readout,readout_p,reward,dynamics,obstacle,cost_to_go}.pt   + info.json, reports/*.json
```

Scenes are the unit of the split, so every state of an episode is in one split.

## The latent z

`encode.py` runs `facebook/vjepa2-vitl-fpc64-256` frozen in fp16 on the newest 16 frames of each camera,
averages the tokens over time, keeps the spatial grid (static 16x16 cells, wrist 8x8) and projects each cell
with a PCA (32 / 16 numbers per cell, fitted label-free on 300 training clips and reused online by the
controller). `z = [static 8192 | height map 64 | wrist 1024]`. The grid is what makes the cube readable: on
a mean-pooled or 8x8 latent no readout placed the 15 px cube better than 3-4 cm. 0.26 s per state on an RTX
4060, 0.09 s on the cluster, the same online.

## The models

| model | input -> output | notes |
|---|---|---|
| Q `readout` | z, gripper width and command -> cube xyz (absolute while free, relative to the hand while held), held | a `GridReadout`: convolutions on each camera grid and a spatial softmax on the static grid, no arm pose as input (an MLP on (z, p) learned "the cube is near the hand"). 0.6 / 0.7 cm on validation, held F1 0.99 |
| R `reward` | z, p -> reach, grasp, lift, transport, placed, proximity, P(collision), P(table hit), P(empty grasp) | the planners use the penalties; the task terms are unused since the potential comes from Q |
| D `dynamics` | z, p, action -> z', p' | `SplitDynamicsQ`: a robot head on (p, a, frozen Q's reading), a visual head on (z, p, a, p'); trained on 8-step windows with a task-consistency term through the frozen Q, otherwise the cube vanished from the imagined latent |
| O `obstacle` | a 9x9 window of the obstacle map around the hand, hand height, gripper, held -> proximity (the carried cube counted), P(contact) | one shared 64-wide MLP per cell and a max over the cells (4,930 parameters; an MLP over the window memorised scenes). Proximity MAE 0.18 when penalised, contact AUC 0.97 |
| C `cost_to_go` | obstacle map, target -> a 32x32 field of path length | a 6-layer convolution trained on 20k synthetic scenes, labels from a grid value iteration where cells taller than 12 cm block within 7 cm: a distilled planner, optional (`--cost-to-go`), runs on the CPU |

Every checkpoint carries its normalisation statistics; `train.fit` keeps the best validation epoch (AdamW,
cosine decay, weight decay 0.05, dropout, Gaussian noise on z for Q and R: without them every stage
memorised the scenes by epoch 3).

## The package the controllers run with

`models/<tag>/` = `attempts/<tag>/models/*.pt` + `info.json` (config, rest height) + `features/{pca.npz,
meta.json}` (meta without its `keys` list), about 45 MB, tracked through Git LFS. `train.model_folder(root,
tag)` resolves both forms: `--models-run models/v3c` (no tag) or `--models-run data/episodes_v3 --tag v3c`.
