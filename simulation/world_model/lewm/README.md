# lewm: the LeWM comparison (reference only, not part of the pipeline)

This package is kept as it was written on the M2_Kuba branch. It was never run: no `lewm` folder
exists under `data/`, `results/` or the data archive, and its README section at the time said
"notebook training, GPU memory fit and control performance have not been observed". Its imports
(`world_model.vision.*`, `clutter_pipeline`, `full_task`) point at the M2 code, so it does not
import against the reorganised packages. To run it, use the code it was written for:

```bash
git worktree add ../world_model_project_m2 b4b0e93e      # the "New architecture" commit, M2 modules intact
cd ../world_model_project_m2/simulation
../.venv/bin/python -m pip install einops                  # the only extra dependency (vendored LeWM code)
python -m world_model.lewm.pipeline --pilot --out data/lewm_native_pilot --export ../results/lewm_native_pilot
```

It needs a finished `q_clutter_v1` campaign (`world_model.vision.clutter_pipeline` on that commit) as its
source run; that run was shared through Git LFS (`simulation/data/q_clutter_v1`) and is not on this
machine.

## Why it was there

The M2 plan compared two ways of planning in a latent space on the same recordings:

- the project's own route: a frozen pretrained encoder (V-JEPA 2) plus small heads trained on top of
  it, `Q` to read the cube, `D` to imagine, and later `R` for contacts, with the SAC as the prior;
- the route of the LeWorldModel paper (arXiv 2603.19312, section 3.2): train a small encoder and a
  transformer predictor together from scratch on the task's own frames and actions, and plan by
  driving the predicted latent towards the latent of a goal image, with no readout, no reward model
  and no policy prior.

The second route is what `lewm/` implements: `upstream/jepa.py` and `upstream/module.py` are the
official MIT code at commit `8edfeb33` (the JEPA wrapper, the autoregressive predictor, the SIGReg
regulariser), `model.py` adapts them to this simulator (ViT-Tiny encoder at 224 px, 192-value latent,
three images five control steps apart as context, the five executed commands between two images as
one action, a goal-image cost for the CEM), and `pipeline.py` is a seven-stage runner (prepare, train,
encode, specify goal images, offline evaluation, live control, compare) built on the M2 `clutter_pipeline`
campaign. The point was an honest whole-system comparison: can a world model trained end to end on
17k task frames plan the task from a goal image, against the frozen-encoder system that gets the goal
as numbers.

## How it differs from the model in use now

| | LeWM (this package) | the current world model (`world_model/`) |
|---|---|---|
| encoder | ViT-Tiny trained from scratch on the task frames, 192-value latent | V-JEPA 2 ViT-L, frozen, pretrained on video; 1024-value spatial latent |
| what is learned | encoder, action embedder and predictor jointly, prediction MSE + SIGReg | only small heads on the frozen latent: Q, R, D |
| predictor | transformer over 3 image latents, one step = 5 control steps | MLP `D(z, p, a)`, one step = one control step, robot state `p` as input |
| reads the cube? | no readout at all | `Q(z, p)` gives cube xyz and held, used by the policy and the planner |
| goal | a goal image, encoded by the same encoder | the target position as numbers in the reward |
| planner score | latent distance between the final imagined latent and the goal latent | the task reward on Q's readings plus R's contact penalties, discounted |
| prior | none, plain CEM from scratch (population 128, 10 iterations) | the frozen SAC's action sequence as the guide, candidate 0 |
| data | the `q_clutter_v1` clutter recordings | `combined_test1`: full task at 3 cm, obstacle pairs, the full mix |
| status | prepared, never run | the numbers in `CHANGES_FROM_M2.md` |

In short: LeWM trains the whole representation for this one task and plans towards a picture;
the current system keeps a general pretrained representation fixed, learns only how to read and
predict it, and plans around a policy that already solves the task.
