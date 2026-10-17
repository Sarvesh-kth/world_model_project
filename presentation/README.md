# Fresh SAC inspection and presentation figures

Open `SAC_Inspection.ipynb` in Jupyter after pulling this branch. It has seven separate code cells: setup, one per policy/table combination, a figure export, and a ZIP download. The viewer is outside `simulation/`, so this update does not change the running campaign's source signature. It imports no Torch/MuJoCo/JEPA and performs no new rollouts or training.

Each gallery refreshes completed episodes and displays:

- Four successes sampled with a fixed random seed.
- Three failed episodes with the smallest **final** cube-to-goal distance.
- Three random failures excluding those three closest cases.

If too few outcomes exist, it displays only what exists and says how many are missing. No duplicate clips are added. Selections and paths are saved separately per group. A near-goal failed episode remains a failure; the recorded placement flag includes lifting/release/settling requirements. Selection is reproducible for an unchanged set of completed episodes.

The figure cell waits for pipeline completion and all planned seeds. It validates run/scene/initial-state pairing and creates two presentation figures using **all episodes**, independently of gallery selections:

1. `sac_comparison`: success counts/rates, every final distance with mean, and contact timesteps per episode. Contact totals are annotated; contact timesteps are not separate collision events.
2. `sac_outcome_stages`: observed outcome categories from actual evaluation labels: placed; never held; held but never lifted4cm; lifted and still held at end; lifted and unheld at end without placement. These are observable endpoint/stage categories, not inferred causes such as "bad perception" or "premature commanded release".

Both figures are saved as300dpi PNG and editable SVG under `results/calle_sac_scratch_v1/presentation`, alongside CSV metrics, checkpoint-selection metadata, exact video selections and figure provenance. The selected checkpoint can be BC-only if SAC updates failed to improve validation; the exported CSV and printed output identify it. Only one training seed per variant is currently used. The figures describe these observed test sets; they do not establish a general SAC/JEPA advantage or identify a causal failure mechanism.

The notebook is committed with no outputs; GIFs and generated figures stay in ignored run/results folders. No new dependencies are added: galleries need IPython; figures need the already-used Matplotlib.

## Download package

Rerun the setup cell after pulling updated code, then run `viewer.export_zip()` in the last cell. It regenerates the final figures, applies the same reproducible gallery selection, and creates `calle_sac_scratch_v1_presentation.zip` under the project presentation/ directory with a direct KTH Jupyter download link. It waits for the complete campaign and reports missing selected GIFs instead of silently excluding them. Selections can change while a run is in progress; rerun galleries after completion to match the final archive.

Archive folders:

- `videos/{train_empty,train_mixed}/{empty,obstacles}/{successes,failures_closest,failures_random}/`: up to 40 selected GIFs and their result JSON files.
- `graphs/`: both figures as PNG and SVG.
- `metrics/`: full-campaign CSV, provenance, and the four selection manifests.
- `README.txt`: selection rules, actual clip counts, and interpretation limits.

No weights, raw frames or trajectory arrays are included. GIFs are copied without conversion. The export replaces a previously generated ZIP of the same name; raw run files are read only. If the notebook interface cannot follow the link, download the named ZIP from its file browser.

## SAC versus MPC (matched 30 + 30 scenes)

Open **SAC_MPC_Inspection.ipynb** for the combined comparison. Setup plus six separate gallery cells, then final plots and ZIP. Uses data/calle_sac_scratch_v1 plus data/calle_mpc_matched_empty30 and data/calle_mpc_matched_obstacles30. Same 4 successes + 3 closest failures + 3 other random failures per group, up to60GIFs total. All180episodes feed the figures and CSV (including failure-only mean distance and decision time). Plots wait if either MPC group is missing/incomplete. MPC galleries can show available completed episodes before that.

The reader checks planned seed sets, actual layouts and initial-state hashes against the saved SAC scenes/resets, and checks current world-model bundle files against original SAC input hashes. It rejects blind/disabled-penalty/different-source options and mixed MPC configurations. It records complete per-run CLI and per-episode MPC settings in provenance. **Legacy MPC runs did not record historical code/model hashes**, so current-file checks cannot prove original runtime bytes. This is a complete-system comparison: MPC uses different obstacle inputs/score terms from the41-input actor.

Combined outputs stay under results/calle_sac_scratch_v1/presentation/with_mpc. ZIP: presentation/sac_mpc_comparison_presentation.zip; videos split train_empty/train_mixed/mpc then empty/obstacles and outcome category. Actual recorded GIFs are copied, not regenerated. Direct download (after export on the cluster):

https://gpu1.eecs.kth.se/user/kuba/files/Robots%26EmbodiedAI/project/world_model_sac_obstacles/presentation/sac_mpc_comparison_presentation.zip?download=1

For a custom notebook, pass `mpc_runs={"empty": <empty-run>, "obstacles": <obstacle-run>}` to SACViewer, use `viewer.show("mpc", "empty")` / `viewer.show("mpc", "obstacles")`, then the usual presentation_figures/export_zip methods. Need IPython, NumPy and Matplotlib; if plots report missing Matplotlib, use `%pip install matplotlib` in the notebook kernel. No simulator/model imports or changes to experiment source signatures.

## Replay every MPC proposal with slow playback

Historical `plans.npz` contains only the chosen path plus5runner-ups from the winning CEM round. A replay cannot recover the discarded candidates. The optional **--save-all-plans** flag records all64candidates in each of3rounds during a fresh visualisation episode, including scores, validity, round/candidate IDs, chosen/runner-up indices and hand paths. Recording adds no sampling, scoring or action changes. It is off by default. Near the episode time limit paths are NaN-padded to the configured horizon; unrecorded slots carry round=-1. Invalid/nonfinite forecasts remain in the data; only finite geometry can be drawn, and captions state validity counts.

Run in a terminal from simulation/ after pulling this branch:

```bash
../.venv/bin/python -u -m controller.control_pipeline \
  --methods mpc --models-run models/v3c \
  --mpc-config controller/mpc_config.py \
  --random-ends --obstacles 1 2 \
  --seed 20476039 --test-episodes 1 \
  --position-jitter 0.01 --penalty-scale 1 \
  --threads 4 --headless --save-all-plans --resume \
  --out data/mpc_allpaths_seed20496039

../.venv/bin/python -m controller.replay \
  data/mpc_allpaths_seed20496039/episodes/mpc/seed_20496039 \
  --video --all-paths --every 1 --speed 0.5 --size 512 \
  --out ../presentation/mpc_allpaths_seed20496039
```

This reruns **one** episode with fresh live perception, not the30-scene benchmark or SAC training. Separate output keeps the original benchmark intact; numerical/rendering variation can change outcomes, so the new video must use its own result.json. These simulation source edits change the old training wrapper's source fingerprint; do not attempt to resume the finished sac_scratch campaign with changed code. The presentation readers still read its completed results.

Replay colours: thickgreen chosen imagined hand; faintgreen up to5valid high-scoring runner-ups from the winning round; other finite proposals orange→red with increasing **mean Euclidean distance from the chosen path across corresponding future timesteps**. Colours do not encode reward, collision probability or distance from goal. Invalid forecasts are fainter. Purpledots chosen imagined cube, blueactual hand history. Fallbacks show an explicit hold-still notice rather than label an invalid forecast as the executed path. No full candidate cube paths or rendered future camera images are recorded. Overlapping proposals can look like one line.

--every1 keeps all recorded steps; --speed0.5 gives half-speed GIF/MP4 playback (5frames/s for a10Hz recording). Speed changes display timing only. Replay always includes the final recorded state. No JEPA inference occurs during offline rendering. This renderer uses a larger geometry budget for all proposals.

Jupyter display after rendering:

```python
from pathlib import Path
from IPython.display import Image, display
gif = Path.home() / "Robots&EmbodiedAI/project/world_model_sac_obstacles/presentation/mpc_allpaths_seed20496039.gif"
display(Image(data=gif.read_bytes(), format="gif"))
```

Download on KTH after rendering: https://gpu1.eecs.kth.se/user/kuba/files/Robots%26EmbodiedAI/project/world_model_sac_obstacles/presentation/mpc_allpaths_seed20496039.gif?download=1

### Replay framebuffer repair

If512pxexportfailswith `Image height 512 > framebuffer height 480`, onlytheofflineexportfailed;thenewepisodesavedallpaths. Replay now enlarges model.vis.global_.offheight/offwidth before Renderer construction. Pull and rerun **only controller.replay**,usingthealreadyrecordedepisode. Withoutpulling, --size480alsofitsdefaultframebuffer. Open presentation/mpc_allpaths_seed20496039.gif aftersuccessfulrendering; episode/actual.gifalwaysshowstheplaincameraandhasnoimaginedoverlays. Successfulcaptureobservedfromuserlog: seed20496039placedat4.36cm,125steps,0contactsteps,231msdecision. Thisisaseparatevisualisationrun,notanew30-scenebenchmark.

## Four rank-selection counterfactual GIFs

From simulation/: `python -u -m controller.mpc_counterfactual`. Defaults to the successful data/mpc_allpaths_seed20496039/episodes/mpc/seed_20496039 reference. This runs four **actual closed-loop MuJoCo episodes**, with unchanged frozen world models and copied source layout/goal/reset seed/jitter/configuration. It does not just draw a discarded prediction as if it happened. No training or model changes.

| Case | Choice at each robot decision, after all CEM rounds |
|---|---|
| rank_05 | fifth highest score |
| rank_25 | 25th highest score |
| rank_64 | 64th highest score |
| best_60_then_64 | normal best for actions1–60;64th from action61 |

Ranks are **within the64-candidate round that contains the normal overall CEM winner**, not candidate IDs or ranks across192proposals. CEM's three sampling/refinement rounds still optimise normally; only the final chosen sequence changes. Descending score, ties retain candidate order. Invalid candidates rank at their original penalised scores; a selected invalid candidate invokes the original hold-still fallback and is logged, not silently replaced by another rank. The default no-retreat64-candidate source configuration is required. Current observations are re-encoded after every actual action; each variant's next proposals therefore diverge naturally. A candidate with lower model score need not perform worse in the simulator—that is what this experiment examines. One scene cannot establish general ranking quality or JEPA's advantage.

Original recording stays read-only. New runs preserve their own full proposals, executed commands, predicted scores, rank/round/validity/fallback fields, outcome and real trajectory. The script checks each initial qpos/p/cube/held against reference exactly. For the switch variant, summary.json reports whether first60actions and recorded states match the source within1e-6, plus per-array maximum differences. It does not assume a mathematically exact prefix under fresh rendering/inference. The original source ended after125steps; variants use the normal environment success/timeout rules (up to300steps), so their GIF lengths can differ.

Four half-speed5fps all-path overlay GIFs (and renderer MP4/PNG outputs) appear under simulation/data/mpc_counterfactual_seed20496039/gifs. summary.json contains actual outcomes, contact/fallback counts and prefix comparison. counterfactual.json records source/settings/model/code hashes and runtime versions. Each variant has a completion marker and episode hashes; interrupted episodes may be rerun, completed matching episodes are reused before re-rendering. A file lock prevents duplicate launches; changed inputs/render settings require a new --out. Source MPC settings are frozen as literal constants in source_mpc_config.py, without changing the shared configuration.

Cluster command:

```bash
nohup ../.venv/bin/python -u -m controller.mpc_counterfactual \
  > data/mpc_counterfactual_seed20496039_launch.log 2>&1 < /dev/null &
echo "PID: $!"
tail -f data/mpc_counterfactual_seed20496039_launch.log
```

After COMPLETE, Jupyter cell:

```python
from pathlib import Path
from IPython.display import Image, display
root = Path.home() / "Robots&EmbodiedAI/project/world_model_sac_obstacles/simulation/data/mpc_counterfactual_seed20496039"
for name in ("rank_05", "rank_25", "rank_64", "best_60_then_64"):
    print(name)
    display(Image(data=(root / "gifs" / f"{name}.gif").read_bytes(), format="gif"))
print((root / "summary.json").read_text())
```

## Publish code, selected policies and replayable results (no new datasets/media)

From the repository root: `.venv/bin/python scripts/share_control_results.py --stage`. This prepares results/control_release_v1 with compact reports/plots and evaluation replays from the SAC, twoMPC, all-path and optionalcounterfactualruns, plus policies/train_empty and policies/train_mixed (selectedsac_best.zip,pipeline.json,imagined_training.json). SourceactorhashesarecheckedagainstthecompletedSACcampaign. D/Q/R/O/PCAremainatthealreadytrackedmodels/v3cpath. inventory.jsonlistscopiedfilehashes,sourcepaths,completionswhereavailableandepisodecounts;partialrunsarelabelledbyrecordedstatus,notcalledcomplete.

Onlyevaluationtrajectoryqpos,p,cube/held/actionsarecopied;cachedzisdroppedandoriginalarraysremainunchanged. Allrecordedplansareincludedforoverlayregeneration. NoPNGframes,rawcameraimages,GIFs/MP4s,downloadZIPs,trainingdatasets/replaybuffers/latentcacheorvirtualenvsareadded. ExistingtrainingdatainGit/historyisnotdeleted. TheexporterconfiguresGitLFSfortherelease'smodelZIPs/replayNPZsandstagesonlyenumeratedfiles,thenstageschangesinsourcefolders. Notebooksarestagedwithoutputsclearedwhilelocaloutputsremainuntouched. Pre-existingstagedfilesstoptheoperationtoavoidmixingunrevieweddata. Itdoesnotcommit/push.

Use Gitfetch/rebase--autostashthencommit/pushontheexistingcodex/calle-sac-obstaclesbranch. --autostashpreserveslocaltrackedmodelsymlinks/notebookoutputs;onlyselectedstagedfilesentertherelease. Nomodelretraining/newrolloutisinvokedbythisexporter. ItrequiresNumPyfromtheexistingprojectvirtualenv.
