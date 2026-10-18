# Current controller results, weights and GIFs

Handoff for people and agents, 17 October 2026, branch **`codex/calle-sac-obstacles`**.
Cluster results were published in commit **`5bbbbdb0`**. All paths below are
relative to the repository root unless stated otherwise. On Kuba's cluster:

```text
$HOME/Robots&EmbodiedAI/project/world_model_sac_obstacles
```

## 18 October: the simulator-trained SAC against the MPC, 0-3 obstacles (branch `sac_priv_testing`)

The imagined SACs above had no obstacle input and learned from imagined rewards, so the table above says
little about SAC versus MPC. This branch adds the row that was missing: a SAC trained in the simulator
with the simulator's own reward and penalties (placement +15, collision, proximity, table hit, drop, fail),
the obstacles seen through the same obstacle-only depth map the MPC's obstacle model reads (a 9x9 window
around the hand plus the map pooled to 8x8), and Q's cube estimate through the cameras at test time
(`rl_q`). The MPC code and settings were not touched. [simulation/SAC_PRIV.md](simulation/SAC_PRIV.md)
has the method, the cluster commands and the retraining command; the policy is `simulation/models/sac_priv_v1`.

Local run on this laptop (RTX 4060), 10 fresh random-end scenes per obstacle count, seeds `20494010-19`,
both controllers on the same scenes (`results/sac_priv_eval_<n>/`):

| Controller | 0 obstacles | 1 | 2 | 3 | All | Obstacle contact steps |
|---|---:|---:|---:|---:|---:|---:|
| Calle MPC (unchanged) | 7/10 | 7/10 | 7/10 | 5/10 | 26/40 | 0 |
| SAC on Q, `rl_q` | 5/10 | 4/10 | 4/10 | 1/10 | 14/40 | 262 (5 scenes) |
| SAC on the exact state, `rl_true` (diagnostic) | 8/10 | 6/10 | 7/10 | 6/10 | 27/40 | 91 (3 scenes) |

Median final B distance of the MPC's placements 2.4-3.5 cm (21 cm on 3 obstacles, where half fail), of the
SAC's 6.6-7.4 cm. Scenes solved by the MPC only: 15; by the SAC only: 3; by both: 11; by neither: 11.
Decision time 340-540 ms for the MPC with the GPU shared, 1 ms for the SAC.

What the numbers say:

- The SAC itself is as good as the MPC: on the exact state it places 27/40 against the MPC's 26/40, and
  it is the only one of the two that gets 6/10 on three obstacles. But it touches obstacles (91 contact
  steps in 3 scenes); the MPC never does, on any count.
- Through Q it loses half of that (14/40). Every one of its failures on 0-1 obstacles is "held but never
  lifted" or a carry that never arrives: the policy was trained on the exact cube and an immediate contact
  flag, and at test time Q reads the cube 1-2 cm off at the grasp (fingers close on the edge, the cube
  slips after 3-4 steps, no retry) or the width rule keeps saying held after a slip, so it carries air to
  B. The MPC runs on the same Q but re-plans every step and has the empty-grip cost and the gripper
  envelope, so a missed grasp costs it a few steps instead of the episode.
- Training (validation = 10 empty + 20 one-to-three obstacle scenes, every 20k steps): the whole 256-cell
  map plateaued at 12-15/30 with ~1,000 contact steps (the policy has to memorise layouts); the hand-centred
  window went to 17/30; window + coarse map with 500 demonstrations 18/30 from cloning and 20/30 after 20k
  SAC steps (the packaged policy); 1,200 demonstrations with a short cloning 20/30 from cloning alone
  (13/20 on obstacles) and 18/30 after 20k steps with the fewest contacts (264). Demonstrations are the
  lever; RL adds a few placements in the first 20-40k steps and then drifts (-2 to -3 by 100k).

Why the MPC scores 7/10 on empty tables here and 30/30 above: same code and settings, different scenes
(seed base 20475010 there, the pipeline default 20474010 here). In two of the ten local scenes Q reads the
cube 4-5 cm off along y for the whole episode and the planner never gets there; re-running the MPC here
on two of the cluster's own edge scenes (20495037, 20495038) placed both with Q's error at the cluster's
level, so the local renderer is not the cause. Twenty scenes per count on the cluster will settle the rates.

Gripper yaw (checked on request, nothing changed): the MPC samples `dyaw` like the three translations and
no score term mentions yaw, so the yaw random-walks 3-5 degrees per step and sweeps about 90 degrees per
episode, often pinned at the +-90 limit. It does not decide outcomes: the finger misalignment to the cube
face at the first grasp averages 22 degrees for successes and failures alike in the 60 matched episodes,
and grasps succeeded at up to 44 degrees.

## What was evaluated

Two fresh imagined-SAC policies use Calle's original trainer, recorded
`episodes_v3` starts/features and frozen `simulation/models/v3c` world models.
The original Calle actor was unavailable. `train_empty` uses empty-table policy
train/validation starts; `train_mixed` includes empty and obstacle starts.
Both use BC initialization followed by imagined SAC updates, with the same
40,000-transition budget. No world models were refitted for this comparison.

Each policy was tested on the same 30 empty and 30 obstacle scenes. Calle's
standalone `mpc` was then evaluated on those 60 scenes. Actual seeds are
`20495010–20495039` (empty) and `20496010–20496039` (one or two obstacles).
Recorded layouts/reset comparisons are in the exported figure provenance.

| Controller | Test scenes | Placements | Mean final B distance | Obstacle contact steps |
|---|---|---:|---:|---:|
| Empty-trained imagined SAC | Empty | 12/30 | 23.73 cm | 0 |
| Empty-trained imagined SAC | Obstacles | 11/30 | 24.18 cm | 746 |
| Mixed-trained imagined SAC | Empty | 9/30 | 26.54 cm | 0 |
| Mixed-trained imagined SAC | Obstacles | 8/30 | 25.01 cm | 1,126 |
| Calle MPC | Empty | 30/30 | 2.22 cm | 0 |
| Calle MPC | Obstacles | 24/30 | 9.59 cm | 0 |

Source: [full metrics CSV](results/control_release_v1/reports/calle_sac_scratch_v1/presentation/with_mpc/presentation_metrics.csv).
Placement uses the recorded held/lift/release/settling criteria, not final
distance alone. This is one training seed per SAC variant and a comparison of
complete systems: MPC receives a depth-based obstacle map and additional score
terms; the SAC actor has 41 inputs and no explicit obstacle map. Legacy MPC
records lack a historical code/model hash snapshot; see the
[provenance limits](results/control_release_v1/reports/calle_sac_scratch_v1/presentation/with_mpc/figure_provenance.json).

### Where JEPA is used

- **Imagined SAC during execution:** new static/wrist images → frozen JEPA
  encoder and spatial/PCA processing → `Q` cube/held estimate → SAC actor →
  one actual action. `D/Q/R` supply imagined transitions during training.
- **Current `mpc`:** new images → JEPA features; CEM proposes raw action
  sequences → `D` predicts eight successive future states → `Q`, learned
  `R/O` penalties and task/additional score terms rank plans → execute only
  the first action → acquire new images and run JEPA again.
- All MPC candidates use the same learned dynamics. No SAC guide is present
  in this standalone planner. Older Sarvesh `jepa_mpc` guide-retention numbers
  must not be applied to it.

## Shared files in Git

```text
results/control_release_v1/
├── README.md
├── inventory.json                source paths, hashes, run counts/completion
├── policies/
│   ├── train_empty/
│   └── train_mixed/
│       ├── models/sac_best.zip    selected actor/critic checkpoint, Git LFS
│       ├── pipeline.json         minimal loader metadata
│       └── imagined_training.json
├── reports/
│   ├── calle_sac_scratch_v1/      summaries, paired/closest results, figures
│   ├── calle_mpc_matched_empty30/
│   ├── calle_mpc_matched_obstacles30/
│   └── mpc_allpaths_seed20496039/
└── runs/
    ├── calle_sac_scratch_v1/      settings/scenes, evaluation/*/*/episodes/...
    ├── calle_mpc_matched_empty30/ episodes/mpc/seed_*/...
    ├── calle_mpc_matched_obstacles30/
    ├── mpc_allpaths_seed20496039/
    └── mpc_counterfactual_seed20496039/
        ├── summary.json
        ├── counterfactual.json
        └── <variant>/episodes/mpc/seed_20496039/...

simulation/models/v3c/             frozen D/Q/R/O, PCA, encoder/config metadata
simulation/controller/replay.py   saved-state video renderer
simulation/controller/mpc_counterfactual.py  script for the four interventions
presentation/SAC_MPC_Inspection.ipynb        galleries/plots/ZIP on original runs
presentation/sac_viewer.py
presentation/mpc/MPC_PRESENTATION.md         presentation narrative and clip choices
```

Each exported episode has `result.json`, `steps.csv` and compact
`trajectory.npz`; MPC additionally has `mpc_settings.json`, `candidates.csv`
and `plans.npz`. The trajectory retains physical states/actions, not cached
JEPA `z`. Only all-path/probe recordings contain all CEM rounds. Hashes in
`inventory.json` describe exported files; original run metadata describes
the unmodified source files. Keep that distinction when checking hashes.

### Which weights to load

`policies/train_empty/models/sac_best.zip` is the selected 20,032-transition
checkpoint; `policies/train_mixed/models/sac_best.zip` is the selected
40,000-transition checkpoint. Both are renamed copies of their actual selected
checkpoints. Read `imagined_training.json` → `best` and `settings` for identity
and recipe. In particular, `settings.obstacles` is false/true respectively;
the minimal loader `pipeline.json` has a placeholder `obstacles: null` for both.

For an explicitly requested new SAC evaluation, pass the corresponding
`policies/<label>` folder to `controller.control_pipeline --methods sac_imagined
--imagined-run <folder>`, together with the frozen v3c model bundle.
Do not substitute `simulation/data/rl_baseline_v1`: that is a different SAC.

Weights and replay arrays use Git LFS. After a code-only pull they may be small
text pointers. Retrieve only what the intended operation needs, from repo root:

```bash
# Saved-state video rendering: episode replay arrays only, no training data.
git lfs pull --include="results/control_release_v1/**/*.npz" --exclude=""

# Only if loading policies/world models for a new evaluation:
git lfs pull --include="results/control_release_v1/policies/**/*.zip,simulation/models/v3c/**" --exclude=""
```

## GIF catalogue: what we made and why

**Original GIF/MP4 files were intentionally excluded from this release.**
The following locations describe the cluster files produced by the workflow;
Git contains their results and replay data, not copies of the media. Media
pixels were not re-inspected while writing this handoff.

| Set | Cluster location | Purpose |
|---|---|---|
| 120 SAC camera episodes | `simulation/data/calle_sac_scratch_v1/evaluation/{train_empty,train_mixed}/{empty,obstacles}/episodes/sac_imagined/seed_*/actual.gif` | Inspect grasp, carry, placement and failure for both policy training pools and both test conditions. |
| 30 empty MPC camera episodes | `simulation/data/calle_mpc_matched_empty30/episodes/mpc/seed_*/actual.gif` | Matched empty-table comparison with both SACs. All 30 placed, so this group has no failure clips. |
| 30 obstacle MPC camera episodes | `simulation/data/calle_mpc_matched_obstacles30/episodes/mpc/seed_*/actual.gif` | Matched obstacle comparison: 24 successes and 6 failures. |
| Standard MPC imagined-plan replays | `<episode>/replay.gif`, when rendered | Explain chosen hand forecast, five stored alternatives, imagined cube and actual hand history; historical recordings do not contain every CEM proposal. |
| Successful full-path source episode | `simulation/data/mpc_allpaths_seed20496039/episodes/mpc/seed_20496039/actual.gif` | Separate one-scene capture retaining all proposals; 125 steps, 4.36 cm, placed. The plain `actual.gif` has no overlays. |
| Full-path overlay | `presentation/mpc_allpaths_seed20496039.gif` after the repaired replay command | Show all 64 proposals × 3 CEM rounds at each decision, chosen plan and actual movement, at half speed. |
| Four counterfactual camera episodes | `simulation/data/mpc_counterfactual_seed20496039/<variant>/episodes/mpc/seed_20496039/actual.gif` | Record what physically happened when rank selection changed, with fresh JEPA observations after every action. |
| Four counterfactual overlays | `simulation/data/mpc_counterfactual_seed20496039/gifs/{rank_05,rank_25,rank_64,best_60_then_64}.gif` | Compare choosing 5th/25th/64th, or best for 60 actions then 64th. See the presentation notes for exact outcomes and ranking semantics. |
| Selected SAC gallery | `presentation/calle_sac_scratch_v1_presentation.zip` | Up to 40 selected original camera GIFs, grouped by policy, test condition and outcome; figures/metrics included. |
| Selected SAC + MPC gallery | `presentation/sac_mpc_comparison_presentation.zip` | Saved final selection manifests contain **54 clips**: 40 SAC, 4 empty MPC successes, 10 obstacle MPC examples. Copies of evaluation GIFs, not additional experiments. |
| Calle's nine historical presentation examples | `presentation/mpc/media/`, and separately shared `mpc_presentation.zip` for GIFs | Earlier successes, before/after approach/release fixes, edge and held-flicker failures. Exact prefixes, seeds and historical settings are in `MPC_PRESENTATION.md`; do not merge these with the matched benchmark. |

Gallery rule per group: four reproducibly sampled successes, three failures
closest by final B distance and three other random failures, without duplicates.
Missing outcomes are disclosed. Each GIF must keep its seed, controller,
training label, condition and actual `task_success` label when used in slides.

### Earlier development GIFs in the original project/archive

These are separate historical assets, outside the new release. Their paths are
relative to the original `world_model_project` checkout or Sarvesh's run archive;
they are not guaranteed to exist in this fresh clone. Retrieve originals before
claiming a historical clip is available; a fresh rerun needs its original model
bundle and must be labelled as a new run.

| Earlier set | Historical path | Why it matters in the presentation |
|---|---|---|
| Frozen-SAC comparison | `simulation/data/control_frozen_v2/episodes/<method>/seed_*/actual.gif` | Earlier scripted/exact-state RL versus JEPA-readout RL and future-planning comparison. |
| V1 clutter/drop examples | `simulation/data/q_clutter_v1/control/<scene>/mixed/episodes/rl_q/seed_*/actual.gif` | The five previously downloaded Q-mixed clips: two placements, the farthest failure and two closest failures. These use JEPA perception feeding RL, without live future-plan selection. |
| Native LeWM comparison | `simulation/data/lewm_native_v1/control/` | Separate latent-goal planning attempt; keep its controller labels and results distinct from Calle MPC. |
| V1.5 MPC pilot | `simulation/data/control_test2/episodes/jepa_mpc/seed_*/actual.gif` | Intermediate development point before V2/V3. |
| Sarvesh V2 | `simulation/data/control_final_full20/episodes/{rl_q,jepa_mpc}/seed_*/actual.gif` | Spatial representation and two controller modes before the final data/model revision. |
| Sarvesh V3 | `simulation/data/control_final_20/episodes/{rl_q,jepa_mpc}/seed_*/actual.gif` | Later Sarvesh model/data version; its SAC-guided planner differs from current standalone Calle MPC. |

The saved V1 presentation favourites are scene0019 empty/normal seed20464024
(placement, 3.10 cm) and scene0016/scene0019 clutter/drop_late
seeds20464021/20464024 (failed placements, 7.84/8.73 cm). Their detailed folder
names are `scene_0019_empty_normal`, `scene_0016_clutter_drop_late` and
`scene_0019_clutter_drop_late`. Do not confuse these historical forced-drop
episodes with the new normal-obstacle tests, which insert no forced drops.

### Overlay legend

Current replay: thick green = chosen imagined hand path; faint green = up to
five valid runner-ups from the winning round; orange → red = increasing
spatial deviation from the chosen path; purple dots = chosen imagined cube;
blue = actual hand history. Orange/red is not a reward or risk heatmap.
Invalid/nonfinite/overlapping paths may not all appear as separate visible lines.
Older Calle clips use orange for the chosen path; label the clip's own legend.

The title of `presentation/visualise_imagined_paths.ipynb` is not proof that a
clip contains every proposal: its first cell points to a standard matched-run
replay. Use the separate full-path source and `--all-paths` for the 192 proposals.

## Regenerate videos from the Git release

Use the existing project environment. Replay restores saved MuJoCo states;
it needs no JEPA inference, GPU, new controller evaluation or training. These
are new renderings of recorded states, not pixel-identical original camera GIFs.
Explicit model/tag overrides avoid the original cluster's absolute paths.
On KTH, use the working headless backend (`MUJOCO_GL=osmesa
PYOPENGL_PLATFORM=osmesa` if necessary).

From `simulation/`:

```bash
# Standard MPC replay: chosen forecast + five stored alternatives.
../.venv/bin/python -m controller.replay \
  ../results/control_release_v1/runs/calle_mpc_matched_obstacles30/episodes/mpc/seed_20496039 \
  --video --models-run models/v3c --tag '' --every 1 --speed 0.5 --size 512 \
  --out ../presentation/generated/mpc_matched_seed20496039

# Full CEM history of the separately captured successful source.
../.venv/bin/python -m controller.replay \
  ../results/control_release_v1/runs/mpc_allpaths_seed20496039/episodes/mpc/seed_20496039 \
  --video --all-paths --models-run models/v3c --tag '' \
  --every 1 --speed 0.5 --size 512 \
  --out ../presentation/generated/mpc_allpaths_seed20496039

# The four already completed interventions: render only, do not rerun control.
for variant in rank_05 rank_25 rank_64 best_60_then_64; do
  ../.venv/bin/python -m controller.replay \
    "../results/control_release_v1/runs/mpc_counterfactual_seed20496039/$variant/episodes/mpc/seed_20496039" \
    --video --all-paths --models-run models/v3c --tag '' \
    --every 1 --speed 0.5 --size 512 \
    --out "../presentation/generated/$variant"
done
```

Each command writes GIF, MP4 and a final-frame PNG; half-speed is 5 fps for
10 Hz recordings. SAC episodes can use the same renderer with their exported
episode folder; they have no imagined-plan overlay. Never use
`controller.mpc_counterfactual` merely to obtain a video: that script performs
new live rollouts and enforces its original run fingerprint.

## Existing figures and presentation entry points

Already exported, without needing a plotting rerun:

- `results/control_release_v1/reports/calle_sac_scratch_v1/presentation/`:
  `sac_comparison` and `sac_outcome_stages`, PNG/SVG plus CSV/provenance.
- Its `with_mpc/` subfolder: `sac_mpc_comparison` and
  `sac_mpc_outcome_stages`, PNG/SVG, full metrics, six clip selection manifests
  and figure provenance. Figures use all 180 benchmark episodes.
- [Presentation notes](presentation/mpc/MPC_PRESENTATION.md): historical nine
  clips plus the latest matched comparison and rank-intervention story.
- [Presentation workflow](presentation/README.md): notebook, gallery/ZIP
  generation and original cluster paths. `SAC_MPC_Inspection.ipynb` is the
  current tracked notebook; the original standalone `SAC_Inspection.ipynb`
  was removed in the cluster results commit.

The notebook/viewer expects the original cluster run folders, saved absolute
paths, original arrays and actual GIFs. Do not point it at the compact export
and assume it is a relocated original run. On another machine, use the
already shared figures/metrics and the replay commands above. The release
does not include training datasets, raw frames, cached latents, replay buffers
or every training checkpoint; it is sufficient for result inspection and
saved-state video reconstruction, not restarting the completed training job.
