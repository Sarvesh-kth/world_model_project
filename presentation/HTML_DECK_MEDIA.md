# Missing media for the three-stage HTML presentation

The 18 October HTML revision uses the existing final-system MP4s supplied in
`deck-v3-html.zip`. The older V1/V2 clips and rank-intervention GIFs were absent
from that ZIP. Slots display the controller, seed, outcome and final distance;
they never substitute another episode. The user's deck stays outside this repo
under the presentation artifacts in the parent workspace.

## Collect and publish only the selected GIFs

Run in the **cluster terminal**, after any active result push has finished:

```bash
set +e
(
  set -e
  cd "$HOME/Robots&EmbodiedAI/project/world_model_sac_obstacles"
  GIT_LFS_SKIP_SMUDGE=1 git pull --rebase --autostash origin codex/calle-sac-obstacles
  .venv/bin/python scripts/collect_presentation_media.py --stage
  git diff --cached --stat
  git commit -m "Share selected historical and rank-comparison presentation GIFs"
  git fetch origin codex/calle-sac-obstacles
  GIT_LFS_SKIP_SMUDGE=1 git rebase --autostash origin/codex/calle-sac-obstacles
  git push origin HEAD:codex/calle-sac-obstacles
)
```

The collector looks in this checkout and the sibling `world_model_project`.
It copies only up to eleven named GIFs into `presentation/history_media`, with
source paths, descriptions and SHA256 hashes in `selection.json`. Missing clips
are reported and left as placeholders. It rejects LFS pointer text pretending
to be a GIF, refuses to overwrite a different selected clip, and stops before
staging if unrelated changes are already staged. No simulation, model inference,
training data, weights, raw frames or new MP4s are involved.

If the Sarvesh runs are absent, get his original archive/project path and add it
to the collection command on its **first invocation**, replacing that line above:

```bash
.venv/bin/python scripts/collect_presentation_media.py \
  --source-root /absolute/path/to/Sarveshs/project-or-archive --stage
```

The extra root can contain `simulation/data/<run>`, `data/<run>` or `<run>`.
Do not replace a missing historical clip with a fresh evaluation under current
weights. Preserve its original controller/seed/version identity.

## Selected filenames

| File | Recording |
|---|---|
| `v1_success.gif` | q_clutter_v1, scene0019 empty/normal, rl_q seed20464024, placed3.10cm |
| `v1_failure.gif` | q_clutter_v1, scene0016 clutter/drop_late, rl_q seed20464021, failed7.84cm; deliberate drop |
| `v2_success.gif` | control_final_full20, jepa_mpc seed20494012, placed0.45cm |
| `v2_failure.gif` | control_final_full20, jepa_mpc seed20494022, failed7.49cm |
| `v3_success.gif` | historical control_final_20, jepa_mpc seed20494022, placed0.70cm; optional backup asset |
| `v3_failure.gif` | historical control_final_20, jepa_mpc seed20494011, failed53.34cm; optional backup asset |
| `rank_05.gif` | current rank5 intervention, placed2.39cm |
| `rank_25.gif` | current rank25 intervention, placed1.36cm |
| `rank_64.gif` | current rank64 intervention, placed6.38cm |
| `best_60_then_64.gif` | best through action60, rank64 from61, failed13.18cm |
| `mpc_allpaths.gif` | full192proposal overlay of source seed20496039, placed4.36cm |

## Bring them into the HTML package

On the presentation machine, pull the branch with smudging disabled, then:

```bash
git lfs pull --include="presentation/history_media/*.gif" --exclude=""
```

Copy the selected GIFs into the HTML package's `media/` folder with exactly the
names above and run `python3 build_deck.py` from the package. In the maintained
workspace the builder also imports these files directly from this repository.
It refreshes `index.html`, the standalone embedded HTML and the portable ZIP.

Alternatively, the HTML placeholder's **Add recorded clip** button displays a
local GIF/MP4 immediately. This temporary choice does not modify the saved HTML;
copy/rebuild to preserve the clip for the actual presentation.
