# Controller experiment handoff

Read [CONTROL_RESULTS.md](CONTROL_RESULTS.md) first for the current experiment,
weights, results, GIF locations and replay commands. For slides, also read
[MPC_PRESENTATION.md](presentation/mpc/MPC_PRESENTATION.md), especially its
17 October update. Earlier sections describe historical runs and colours.

- This branch is `codex/calle-sac-obstacles`. `train_empty` and `train_mixed`
  are fresh fits using Calle's imagined-SAC recipe, not his unavailable original
  checkpoint. They share frozen v3c world models.
- Treat `results/control_release_v1` as an archived export. Read its inventory,
  run settings, per-episode results and training metadata before making claims.
  Do not rewrite its files or silently replace an episode with a new rollout.
- Original camera GIFs and generated overlay GIFs remain on the cluster; they
  were deliberately excluded from the new Git upload. Replay NPZ files allow
  new renderings of recorded states without training or controller evaluation.
- Current method `mpc` is Calle's standalone CEM planner, with no SAC guide.
  The old 97.38% SAC-guide statistic belongs to another implementation.
- Rank-5/25/64 and best-60-then-64 clips are four actual closed-loop experiments
  on one scene. All use the same learned dynamics; they are not rendered
  hypothetical predictions or a general ranking-quality benchmark.
- Contact steps are timesteps, not distinct collision events. Keep historical
  benchmarks, the matched 180 episodes and the single-scene probes separate.
- Preserve raw runs and shared model/data symlinks. Put new media under
  `presentation/generated/` and new experiments in separate output folders.
  Do not add datasets, raw frames or new video archives to Git unless requested.
- GPU work belongs on the teaching cluster. Do not start evaluations/training
  just to inspect results or regenerate a saved-state replay.
