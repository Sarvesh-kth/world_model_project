# World model: data, encoder, Q, R, D, O, C

```
prepare_episodes.py   a collected dataset -> run folder (manifest.json, train / val / test split by episode)
encode.py             V-JEPA 2 -> z for every state, the PCA bases; --pca reuses another run's bases, --resume continues
merge_runs.py         several runs encoded with the same bases -> one run
train.py              readout (Q), reward (R), dynamics (D); load_model, model_folder, readout(), imagine()
train_obstacle.py     the obstacle model O on the obstacle-only depth map
train_cost_to_go.py   the cost-to-go C on synthetic scenes
eval_reward.py        R on real latents and on latents imagined by D
eval_grasp.py         does the world model tell a grasp from a miss in imagination
run_combined.sh       train everything on one run, then the two checks:  bash world_model/run_combined.sh data/episodes_v3 v3c 60 [retrain]
common.py             the run folder format, metrics, helpers
```

## A run folder

```
data/<run>/
  manifest.json          every recorded state: key, scene, split, the last 64 frame paths per camera, heightmap (64),
                         p (20), object_xyz, held, obstacle_contact, reward_components, action_from_previous, case
  scene_settings.json    the layout and seed of every episode
  features/latents.npy   one z per state, float32 (N, 9280); pca.npz, meta.json (encoder settings, block layout)
  features/obstacle_maps.npz, cost_to_go_*.npz   the obstacle-only map per scene, the synthetic cost-to-go data
  attempts/<tag>/models/{readout,readout_p,reward,dynamics,obstacle,cost_to_go}.pt   + info.json, reports/*.json
```

Scenes are the unit of the split, so every state of an episode is in one split.

## The latent z

`encode.py` runs `facebook/vjepa2-vitl-fpc64-256` frozen in fp16 on the newest 16 frames of each camera,
averages the tokens over time, keeps the spatial grid (static 16x16 cells, wrist 8x8) and projects each cell
with a PCA (32 / 16 values per cell, fitted label-free on 300 training clips, reused online by the
controller). `z = [static 8192 | height map 64 | wrist 1024]`. The grid is what makes the cube readable: on a
mean-pooled or 8x8 latent no readout placed the 15 px cube better than 3-4 cm. 0.26 s per state on an RTX
4060 (0.09 s on the cluster), the same online.

## The models

| model | input -> output | notes |
|---|---|---|
| Q `readout` | z, gripper width and command -> cube xyz (absolute while free, relative to the hand while held), held | a `GridReadout`: convolutions on each camera grid, a spatial softmax on the static grid, no arm pose as input (an MLP on (z, p) learned "the cube is near the hand"). 0.6/0.7 cm on validation, held F1 0.99 |
| R `reward` | z, p -> reach, grasp, lift, transport, placed, proximity, P(collision), P(table hit), P(empty grasp) | the planner uses the four penalties; its task terms are unused since `--reward-source readout` |
| D `dynamics` | z, p, action -> z', p' | `SplitDynamicsQ`: a robot head on (p, a, frozen Q's reading), a visual head on (z, p, a, p'); trained on 8-step windows with a task-consistency term through the frozen Q (`--task-weight 1`), otherwise the cube vanished from the imagined latent |
| O `obstacle` | a 9x9 window of the obstacle map around the hand, hand height, gripper, held -> proximity (carried cube included), P(contact) | one shared 64-wide MLP per cell and a max over cells (4,930 parameters); an MLP over the window memorised scenes. Proximity MAE 0.18 when penalised, contact AUC 0.97 |
| C `cost_to_go` | obstacle map, target -> a 32x32 field of path length | a 6-layer convolution trained on 20k synthetic scenes from the environment's samplers, labels from a grid value iteration where cells taller than 12 cm block within 7 cm; a distilled planner, optional in the controller (`--cost-to-go`), runs on the CPU |

All checkpoints carry their normalisation statistics; `train.fit` keeps the best validation epoch (AdamW,
cosine decay, weight decay 0.05, dropout, Gaussian noise on z for Q and R: without them every stage
memorised the scenes by epoch 3).

## The chain

```bash
python -m data_collection.collect --config configs/full_mix.yml --episodes 400 --out data/episodes_v3/episodes_raw --workers 4 --seed 1000
python -m world_model.prepare_episodes --episodes data/episodes_v3/episodes_raw --run data/episodes_v3
python -m world_model.encode --run data/episodes_v3                                            # --resume to continue
python -m world_model.train readout  --run data/episodes_v3 --tag v3c --epochs 60
python -m world_model.train reward   --run data/episodes_v3 --tag v3c --epochs 60
python -m world_model.train dynamics --run data/episodes_v3 --tag v3c --epochs 60 --rollout-steps 8 --task-weight 1.0 --robot-sees-q
python -m world_model.train_obstacle   --run data/episodes_v3 --tag v3c --epochs 30            # 3 min, renders each scene's map once
python -m world_model.train_cost_to_go --run data/episodes_v3 --tag v3c                        # 15 min, synthetic data cached
python -m world_model.eval_reward --run data/episodes_v3 --tag v3c --split test
python -m world_model.eval_grasp  --run data/episodes_v3 --tag v3c --split test
```

`readout` before `dynamics` (D needs Q). `bash cluster_round.sh <tag> 400` from `simulation/` runs the whole
chain including the collection, the package and the closed-loop runs; every step restarts where it stopped.
To add episodes: collect into another folder, `prepare_episodes`, `encode --pca data/episodes_v3/features/pca.npz`,
`merge_runs --runs data/episodes_v3 data/<new> --out data/<merged>`, train with a new tag (`NEXT.md`, step 1).

## The package the controllers run with

`models/<tag>/` = `attempts/<tag>/models/*.pt` + `info.json` (config, rest height) + `features/{pca.npz,
meta.json}` (meta without its `keys` list), about 45 MB, tracked through Git LFS. `train.model_folder(root,
tag)` resolves both forms: `--models-run models/v3c` (no tag) or `--models-run data/episodes_v3 --tag v3c`.
