# Fresh SAC inspection and presentation figures

Open `SAC_Inspection.ipynb` in Jupyter after pulling this branch. It has six separate code cells: setup, one per policy/table combination, and a final figure export. The viewer is outside `simulation/`, so this update does not change the running campaign's source signature. It imports no Torch/MuJoCo/JEPA and performs no new rollouts or training.

Each gallery refreshes completed episodes and displays:

- Four successes sampled with a fixed random seed.
- Three failed episodes with the smallest **final** cube-to-goal distance.
- Three random failures excluding those three closest cases.

If too few outcomes exist, it displays only what exists and says how many are missing. No duplicate clips are added. Selections and paths are saved separately per group. A near-goal failed episode remains a failure; the recorded placement flag includes lifting/release/settling requirements. Selection is reproducible for an unchanged set of completed episodes.

The final cell waits for pipeline completion and all planned seeds. It validates run/scene/initial-state pairing and creates two presentation figures using **all episodes**, independently of gallery selections:

1. `sac_comparison`: success counts/rates, every final distance with mean, and contact timesteps per episode. Contact totals are annotated; contact timesteps are not separate collision events.
2. `sac_outcome_stages`: observed outcome categories from actual evaluation labels: placed; never held; held but never lifted4cm; lifted and still held at end; lifted and unheld at end without placement. These are observable endpoint/stage categories, not inferred causes such as "bad perception" or "premature commanded release".

Both figures are saved as300dpi PNG and editable SVG under `results/calle_sac_scratch_v1/presentation`, alongside CSV metrics, checkpoint-selection metadata, exact video selections and figure provenance. The selected checkpoint can be BC-only if SAC updates failed to improve validation; the exported CSV and printed output identify it. Only one training seed per variant is currently used. The figures describe these observed test sets; they do not establish a general SAC/JEPA advantage or identify a causal failure mechanism.

The notebook is committed with no outputs; GIFs and generated figures stay in ignored run/results folders. No new dependencies are added: galleries need IPython; figures need the already-used Matplotlib.
