# world_model: the flow, file by file

This is the plain walkthrough. README.md in this folder explains the ideas; this file follows the code
in the order it runs and says what every function does, what it reads and what it writes. Everything
runs from `simulation/`.

## The flow in one page

```
rl/rl_baseline.py  ->  data/rl_baseline_v1/        the frozen SAC: pipeline.json (config, layout), models/sac_best.zip

collect_full.py        ->  data/full_test1/            manifest.json + observations/<rollout>/static_NNNN.jpg
collect_obstacles.py   ->  data/obstacles_test1/       same format
data_collection.collect + prepare_episodes.py  ->  data/episodes_test1/   same format, frames linked from episodes_raw/

encode.py  --run data/full_test1                      ->  data/full_test1/features/{latents.npy, pca.npz, meta.json}
encode.py  --run data/obstacles_test1_fullpca --pca ...  (a folder of symlinks to obstacles_test1 plus its own features/)
encode.py  --run data/episodes_test1_fullpca  --pca ...

merge_runs.py  ->  data/combined_test1/     one manifest, one latents.npy, symlinks to the three source runs

train.py readout   --run data/combined_test1 --tag combined  ->  attempts/combined/models/readout.pt, readout_p.pt
train.py reward    ...                                       ->  attempts/combined/models/reward.pt
train.py dynamics  ... --rollout-steps 8 --task-weight 1 --robot-sees-z  ->  attempts/combined/models/dynamics.pt
eval_reward.py     ...                                       ->  attempts/combined/reports/reward_test.json

controller/control_pipeline.py   loads features/meta.json + pca.npz (the online encoder) and the three .pt files
```

`run_combined.sh` runs the merge, the three training stages and `eval_reward` in that order and skips what
already exists.

## common.py

Small helpers the other files share. No GPU, no MuJoCo.

- `SIMULATION`: the absolute path of `simulation/`, used to keep run folders under `data/`.
- `P_COLUMNS`, `A_COLUMNS`: the names of the 20 robot numbers and the 5 action numbers, in the order
  `data_collection/writer.py` writes them (`joint_pos_1..7`, `joint_vel_1..7`, `ee_x ee_y ee_z ee_yaw`,
  `gripper_width`, `gripper_cmd`; `action_dx dy dz dyaw gripper`). `prepare_episodes` reads csv columns
  by these names; everything else works on arrays in this order.
- `write_json(path, value)`: writes json through a temporary file and renames it, so a run that is
  killed while writing never leaves a half file. All manifests and reports go through this.
- `save_csv(path, rows)`: rows are dicts with possibly different keys; the header is the union.
- `load_run(root)`: reads `<root>/manifest.json` and checks it is one of ours (`schema`
  `vision_consequences_v1`).
- `load_features(root, manifest)`: reads `features/meta.json` and `features/latents.npy`, checks the
  state keys in meta match the manifest (same states, same order) and that the array is finite.
  Returns `(z, meta)`, `z` an `(N, 1024)` float32 array.
- `state_arrays(manifest)`: the `p` (N, 20), cube `xyz` (N, 3) and `held` (N,) arrays of all states.
- `check_manifest(manifest)`: the sanity checks run before encoding and training. Returns a list of
  problems: collection not complete, duplicate state keys, a scene in several splits, a rollout whose
  state count is not actions + 1, an action that does not match the next state's
  `action_from_previous`, a split with only held or only not-held states.
- `outcome_metrics(pred_xyz, probability, xyz, held)`: how good Q is: xyz MAE per axis in cm, height
  MAE, held accuracy / precision / recall / F1, label counts.
- `record(env, key, scene, split, history, action, info, reward)`: one state dict for the manifest,
  read straight from the environment: `p`, cube pose, `held` (both finger pads touching while closed),
  the two contact flags, the reward stage, the reward and its unweighted components, the action that
  led here, the time, and `frames` = the last 64 frame paths ending at this state (the encoder takes the
  newest 16).
- `new_manifest(campaign, cfg, settings)`: the empty manifest every collector starts from.

## collect_full.py

The full pick-and-place task on the SAC's layout, recorded for training Q and D.

- `CASES = normal, drop_early, drop_middle, drop_late`; `VIEWS = empty, clutter`.
- `layouts(base, seed, jitter)`: from the SAC's layout makes the scene's two layouts: `empty` (cube
  start moved by a uniform jitter of up to 3 cm per axis) and `clutter` (the same plus 1 to 3 boxes at
  the far table edge, never on the route). Same rng seed, so the cube start is identical in both.
- `RecoverySession(cfg, layout, jitter, case)`: a `TaskSession` (rl/task_control.py) that can force the
  gripper open. `reset` remembers the cube's start. `phase()` labels where the task is
  (approach, grasp_lift, carry, lower, release_settle, forced_release, recovery) from the exact
  state; it is stored with every state. `step(action)`: for a drop case, once the cube is held, lifted
  4 cm and the carry has progressed past 15 / 50 / 75 % of the way to B, the next 7 actions are
  replaced by "stand still, open"; the cube falls. After that the scripted policy is restarted
  (`restart_scripted` in info) and it picks the cube up again. `info` also carries
  `executed_action` (what really went to the simulator) and `forced_release`. `result()` adds whether
  the intervention fired and whether the cube was regrasped.
- `run_rollout(root, manifest, cfg, name, scene, split, layout, seed, case, view, sac, replay)`: runs one
  episode and records it. The controller is the SAC for `normal`, the scripted policy otherwise;
  with `replay` given (the clutter view) the executed actions of the empty twin are replayed instead,
  and it is an error if the forced release fired at another step. Every step saves
  `observations/<name>/static_NNNN.jpg`, appends a `record(...)` state with `view`, `case`, `phase`,
  `task_success` and `forced_release`, and at the end appends the rollout (state indices, actions,
  result) and its layout to the manifest. Returns the executed actions.
- `main()`: reads `config` and `layout` from `data/rl_baseline_v1/pipeline.json`, loads
  `models/sac_best.zip`, makes the scene groups (`scene_0000` ... with seeds `--seed + i`, the first
  `--train-scenes` train, then val, then test), and for every scene, case and view calls `run_rollout`
  (empty first, then clutter replaying its actions). The manifest is written after every rollout and
  an existing manifest is continued, so an interrupted collection resumes. At the end
  `complete = True`, `scene_settings.json` (layout per rollout) and `collection.json` (settings,
  config, layout, groups) are written.

Output per scene: 8 rollouts. Defaults 16 / 4 / 6 scenes, about an hour.

## collect_obstacles.py

The paired obstacle scenes the penalty head learns from.

- `layouts(base, cfg, seed, jitter, count)`: the jittered empty layout and the same with 1 to 3
  corridor obstacles from `environment.obstacles.sample_obstacles` (the environment's own sampler
  with `task.obstacles.count` overridden).
- `run_rollout(...)`: like the one in `collect_full` but the controller is `sac`, `scripted` or
  `replay`, nothing is forced, and every state also stores `obstacle_distance`. Uses
  `RecoverySession` with case `normal` only for its `phase()` and `executed_action`.
- `main()`: per scene three rollouts: `<scene>_normal_empty` (SAC, empty), `<scene>_replay_obstacles`
  (those actions replayed with the obstacles present, so contacts and penalties appear),
  `<scene>_scripted_obstacles` (the scripted route planner in the obstacle layout). Same resume and
  output files as `collect_full`. The 6 test scenes are the held-out obstacle benchmark
  (`controller/control_clutter.py` reads `collection.json` and `scene_settings.json` from here).

## prepare_episodes.py

Converts a `data_collection.collect` dataset into the manifest format, so the collector's full mix
(successful, noisy, drops, collisions, random) can be encoded and trained on.

- Reads every `episode_*/data.csv` and `meta.json`. Episodes are split 70 / 15 / 15 by index.
- Every csv row becomes one state: `p` from `P_COLUMNS`, cube pose from the `object_*` columns,
  `held` from `grasped`, the contact flags, the reward and its components from the `reward_*`
  columns, `action_from_previous` from the row's own action columns (the collector stores the action
  that led to the row), the frame path `episodes/<episode>/images/static_<serial>.jpg`.
- Every episode becomes one rollout; `view` is `obstacles` when the layout had obstacles, `case` is
  the policy kind.
- Writes `manifest.json` and `scene_settings.json` into `--run` and symlinks `<run>/episodes` to the
  episode folder so the frame paths resolve.

## encode.py

Turns frames into latents. Needs CUDA. Also provides the `Encoder` class the controller uses online.

- `MODEL`, `PINNED_REVISION`: the Hugging Face model and the weights revision everything was run with.
- `select_history(history, clip_frames, stride)`: the newest `clip_frames` entries of a list, `stride`
  apart, the oldest repeated when the list is short. Used offline on frame paths and online on frame
  arrays.
- `read_rgb(path)`: jpeg -> RGB array, cached (consecutive states share 15 of 16 frames).
- `load_frames(root, state, clip_frames, stride)`: the clip of one state as a uint8 tensor
  `(frames, 3, H, W)`.
- `Encoder(model, revision, pooling, pca, device, clip_frames, stride, dtype, spatial_pool)`:
  - `preprocess(video)`: the HF video processor's steps on the GPU: scale to 0..1, resize to 292,
    centre crop 256, normalise with the processor's mean and std.
  - `tokens(video)`: the encoder without its predictor, autocast to fp16; returns `(T/2 * 16 * 16, 1024)`.
  - `grid(video)`: the pooled grid before PCA. `mean_all`: the mean of all tokens (1024).
    `spatial_mean`: reshape to `(T/2, 16, 16, 1024)`, mean over time, then average `spatial_pool x
    spatial_pool` blocks: `(8, 8, 1024)` with the default 2. `spatial_last`: the last time slice
    instead of the mean.
  - `encode(video)`: `grid` then, unless `mean_all`, subtract the PCA mean and multiply by the
    components: `(8, 8, 16)` flattened to 1024.
  - `encode_frames(frames)`: online entry point, a list of RGB arrays newest last.
  - `name`: the pooling name written to meta.json, e.g. `spatial_mean_pca16_grid8`; the controller
    splits it at `_pca` to get the pooling back.
- `fit_pca(encoder, root, manifest, clips, dims, seed)`: takes `clips` random training states, stacks
  all their grid cells (300 clips x 64 cells = 19,200 vectors of 1024) and fits a `dims`-component PCA
  with `torch.pca_lowrank`. Prints the explained variance.
- `main()`: checks the manifest, creates `features/`, builds the encoder, gets the PCA basis (copied
  from `--pca`, or fitted and saved to `pca.npz`), then encodes every state in manifest order into a
  memmap `latents.partial.npy`, flushing and writing `progress.json` every 50 states so `--resume` can
  continue. At the end renames to `latents.npy`, writes `meta.json` (`keys`, model, revision, pooling
  name, grid, camera, clip settings, dtype, spatial_pool) and deletes `progress.json`.

## merge_runs.py

- Loads each run's manifest and features, requires the same pooling name, clip settings, dtype and the
  same PCA components as the first run.
- Prefixes every state key, scene and frame path with the run folder name, shifts every rollout's
  state indices by the number of states merged so far, concatenates states, rollouts and
  scene_settings.
- Writes the merged `manifest.json`, `scene_settings.json`, `features/latents.npy` (concatenated),
  copies `pca.npz` from the first run, writes `features/meta.json` with the merged keys, and symlinks
  each source run into the new folder so the frame paths `<run>/observations/...` resolve.

## train.py

The three networks and the helpers that load and run them.

- `mlp(inputs, outputs, width=128)`: Linear - LayerNorm - GELU - Linear - GELU - Linear.
- `SplitDynamics(z_dim)`: `robot` head `(p, a) -> delta p` (width 128), `visual` head
  `(z, p, a, p') -> delta z` (width 512). `forward` returns `(z + dz, p + dp)`; the `p` and `p'` fed to
  the visual head are detached.
- `SplitDynamicsZ(z_dim)`: same, but the robot head takes `(z.detach(), p, a)`.
- `normalizer(a)`: mean and std (floored at 1e-3) over the first axis.
- `normalized(a, stats, name)`: `(a - stats[name_mean]) / stats[name_std]` as a float tensor; `name` is
  `z`, `p` or `xyz`. Every network sees normalised inputs; the statistics live in the checkpoint.
- `build_model(ck)`: rebuilds a network from a checkpoint dict (`role`, `input_dim`, `architecture`)
  and loads its weights.
- `load_model(root, role, tag)`: `torch.load` of `<root>/attempts/<tag>/models/<role>.pt`, returns
  `(model, ck)`.
- `readout(model, ck, z, p)`: Q on raw inputs; returns cube xyz in metres and the held probability.
  Works for `readout` (z and p) and `readout_p` (p only).
- `imagine(model, ck, z, p, actions)`: D rolled forward from one raw `(z, p)` through a list of
  actions; returns raw `z` and `p` for every step (used by `eval_reward`; the controller has its own
  batched version).
- `fit(folder, role, model, train, val, loss_fn, meta, args)`: the training loop. Refuses to overwrite an
  existing `models/<role>.pt`. AdamW at `--lr`, batches of `--batch-size`, gradient norm clipped to 5,
  `--epochs` epochs. After each epoch the validation loss is computed in batches; the epoch with the
  lowest one is saved as the checkpoint (`meta` + `role`, `epoch`, `val_loss`, `model` state dict).
  Writes `reports/<role>_training.json` with the loss curve and settings.
- `main()`:
  1. loads the manifest, checks it, loads `z`, `p`, `xyz`, `held`, splits the state indices into train
     and val, computes the normalisation statistics from the training states and normalises
     everything once.
  2. `readout`: for `readout` (input `[z, p]`, 1044) and `readout_p` (input `p`, 20) trains an
     `mlp(..., 4)` with `outcome_loss` = MSE of the first 3 outputs against normalised xyz + BCE of the
     4th against held. Then evaluates both on the validation states with `outcome_metrics` and writes
     `reports/readout_validation.json`.
  3. `reward`: targets per state are `[proximity, collision != 0, table_hit != 0]` from
     `reward_components`; trains `mlp(1044, 4)` with `penalty_loss` = MSE + BCE + BCE on the first
     three outputs. Writes `reports/reward_validation.json` (proximity MAE, precision and recall of the
     two contacts).
  4. `dynamics`: builds the training windows: for every rollout and every start, the `w + 1` states
     and `w` actions (`w = --rollout-steps`). With `--task-weight > 0` loads the frozen `readout` of
     the same tag, checks it was trained with the same statistics. `dynamics_loss` rolls the model
     through the window and sums, per step, the normalised MSE of `z` and of `p`, plus the task term:
     the frozen Q applied to the imagined `(z, p)` must give the recorded xyz (MSE) and held (BCE).
     Trains `SplitDynamicsZ` with `--robot-sees-z`, `SplitDynamics` otherwise.

Checkpoint keys: `model`, `role`, `z_mean z_std p_mean p_std xyz_mean xyz_std`, `z_dim`, `encoder`,
`seed`, `input_dim` (Q, R), `architecture`, `rollout_steps`, `task_weight` (D), `epoch`, `val_loss`.

## eval_reward.py

- `scores(r, rc, z, p)`: R on raw inputs: proximity clipped to [-1, 0], sigmoid of the two logits.
- `metrics(prox, col, tab, truth)`: proximity MAE (overall and on penalised states), and for collision
  and table hit the positives, precision, recall and AUC (fraction of contact / clear pairs ranked
  correctly).
- `main()`: loads R and D of `--tag`, the truth per state from `reward_components`. Real latents:
  metrics on the `--split` states of the empty scenes and of the obstacle scenes separately.
  Imagined latents: for every obstacle rollout of the split, 20 evenly spaced start points; from each,
  D is rolled 1, 4 and 8 steps with the recorded actions and R is scored on the imagined final state
  against the truth of the real final state. Writes `attempts/<tag>/reports/reward_<split>.json`.

## run_combined.sh

`bash world_model/run_combined.sh [run] [tag] [epochs]`, defaults `data/combined_test1 combined 30`.
Merges if there is no manifest yet, trains `readout`, `reward`, `dynamics` (the last with
`--rollout-steps 8 --task-weight 1.0 --robot-sees-z`) if their checkpoint is missing, logging to
`<run>/logs/`, then runs `eval_reward` on val and test.

## What the controller takes from here

`controller/control_pipeline.py` builds `Encoder` with the settings in `features/meta.json` and the
basis in `features/pca.npz`, so the online `z` is the same as the cached one, and loads
`readout.pt`, `reward.pt` and `dynamics.pt` with `load_model`. It uses `readout()` and `normalized()`
from `train.py`, and the manifest's `config` as the environment config. Nothing else from this folder
runs at control time.
