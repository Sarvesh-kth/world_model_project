# controller

The controllers, run on fresh scenes and compared on the same seeds. One command, `--methods` picks the
controller:

```bash
python -m controller.control_pipeline --methods mpc --random-ends --test-episodes 5                       # 5 scenes, live window
python -m controller.control_pipeline --methods mpc --random-ends --obstacles 0 2 --test-episodes 20 --headless --out data/mpc_02
python -m controller.control_pipeline --methods rl_q rl_true --random-ends --obstacles 0 3 --test-episodes 20 --headless --out data/sac_03
python -m controller.control_pipeline --methods jepa_mpc --random-ends --obstacles 0 2 --test-episodes 20 --headless --cost-to-go --out data/planner_02
python -m controller.control_clutter --methods mpc jepa_mpc --headless                                   # the six held-out obstacle scenes
bash sac_priv_eval.sh mpc 10                                                                             # 0/1/2/3 obstacles, 10 scenes each, then rl_q and rl_true
```

| method | perception | action | what it tests |
|---|---|---|---|
| `scripted` | exact state | the scripted policy | the upper bound of the simulator |
| `rl_true` | exact state | the simulator-trained SAC (`--baseline-run`, default `models/sac_priv_v1`) | the policy with perfect perception |
| `rl_q` | cameras through `Q` | the same SAC on the estimate | is the perception good enough to drive a policy |
| `mpc` | cameras through `Q`, `D`, `R`, the obstacle-only depth map through `O` | CEM over raw action sequences imagined by `D`, no rules on the motion or the gripper | can a plain search in the latent do the task |
| `jepa_mpc` | the same, plus the cost-to-go `C` with `--cost-to-go` | CEM over waypoints, a go-to rule turns them into actions | the earlier planner, with a few rules |
| `sac_imagined` | cameras through `Q` | a SAC trained only in imagination (`--imagined-run`) | can a policy learn the task from the world model alone |

Always `--random-ends` (cube and goal anywhere on the table; without it the layout file's one fixed scene).
The world model comes from `--models-run` (default `models/v3c`; a run folder works with `--tag`). Frames,
gifs and a per-step csv land in `data/<out>/episodes/<method>/seed_*/`, the summary in
`../results/<out>/summary.{txt,json,csv}`; `--resume` continues an interrupted run. Seeds are
`--seed + 20000 + i`, 20494010 upward by default. The numbers are in `../../results/README.md`.

```
control_pipeline.py   the command above: WorldModels (the online encoder, Q, D, R, O, C), Policy (a SAC), episode(), the summaries
control_clutter.py    the same controllers on the six held-out obstacle layouts of data/obstacles_v2
mpc.py                the latent MPC: CEM over action sequences, imagine(), the score; knobs in mpc_config.py
mpc_config.py         every knob of mpc with the reason and the measurement behind it; copy it and pass --mpc-config for a variant
mpc_variants/         labelled variants: rule_gripper, plain_scoring, pure_no_steady, penalty_2
planner.py            the waypoint planner jepa_mpc: proposals, the go-to rule, forecasts(), the stage potential, its flags
imagined_sac.py       a SAC trained only on transitions imagined by D
sac_scratch.py        the imagined-SAC experiment: two policies (empty / mixed starts), then both on 30 + 30 camera scenes
mpc_counterfactual.py one recorded mpc scene run again executing the 5th / 25th / 64th best plan, with all-candidate replays
compare.py            placed and failure kinds per run folder;  diagnose.py  one episode as text plus diagnosis.png
replay.py             an episode redrawn in a window or as gif / mp4 from its saved states, with the imagined plans on top
```

## mpc: one decision

Every control step (0.3-0.5 s with the GPU shared, 0.24 s alone; the V-JEPA encode of the new frames is
another 0.25 s):

```
  frames (16 static + 16 wrist) --V-JEPA--> z          depth + segmentation --> obstacle-only map (running max)
  z, p --Q--> cube xyz, held
  64 candidates: 8 raw actions (dx dy dz dyaw gripper) each, sampled around the previous plan shifted by one step,
                 the gripper as one switch time per plan
  for t in 1..8:  D(z_t, p_t, a_t) -> z_t+1, p_t+1;  Q -> cube, held;  R -> penalties;  O(map window at the imagined hand) -> penalties
  score = discounted task reward on Q's readings (GoalReward, as the SAC is trained with) + the potential of every step
          + a pull towards the cube, then towards the goal + penalties - empty-grip cost - carry wobble
  CEM: keep the 8 best, re-centre, 3 rounds; the first action of the best plan goes to the arm
```

What is hand-made is only in the score and in what may be imagined: a pull that does not fade with
distance, a glide slope and a carry height, a grasp counts only if open fingers reached the cube before
closing, a released cube stays where it was let go, a placement counts after 3 settled steps. Each knob in
`mpc_config.py` says what went wrong without it. `python -m controller.mpc --self-check` checks the planner
logic on the CPU.

## jepa_mpc: the waypoint planner

The same models, but a candidate is a waypoint for the hand plus when the gripper switches; a go-to rule
makes the 8 actions (travel at the carry height, descend only when aligned within 2 cm, stand still while
closing). The score is a stage potential from `Q`'s readings (reach < held < placed) plus pulls (straight
line, or `C`'s path length with `--cost-to-go`) plus the penalties. Eight fixed proposals (hold, previous
plan, to the cube, grasp, above the cube, carry, lower, release) are always in the population.

Rules on top of the models and the flags that drop them:

| rule | flag | why it exists |
|---|---|---|
| gripper envelope: an imagined hold counts only if the hand was within 1.5 / 2 cm of the cube when the fingers closed | `--no-grasp-prior` | `D` imagines a hold for fingers closing mid-travel (no such closes in the data); load-bearing, 1/7 without |
| release rule: a held cube is released only with the real hand within half the success radius of the goal | `--no-release-rule` | plans opened 4-9 cm short, `Q` reads a just-released cube toward the hand |
| width rules: fingers under 2 cm hold nothing, fingers shut at 3.5-6.5 cm hold something | `--no-width-gate` | `Q`'s held lags a real grasp by ~12 steps (16-frame time mean); costs nothing to drop (14/20 vs 15/20) |

`--obstacle-source model` (default) scores the imagined hand with `O`; `map` is the old geometric variant (a
height map with hand and cube masks, a clearance penalty, an A* route); `none` leaves `R` alone. The data to
retire the first two rules is collected (`data/episodes_hack`, `../../NEXT.md`).

## Tools

```bash
python -m controller.compare data/mpc_02 data/planner_02 --seeds        # placed / never reached / grasp failed / grasped, not placed, per seed
python -m controller.diagnose data/mpc_02/episodes/mpc/seed_20494010    # text diagnosis + diagnosis.png (no GPU, a second)
python -m controller.replay data/mpc_02/episodes/mpc/seed_20494010      # window: space pause, arrows step, up/down speed, R restart
python -m controller.replay data/mpc_02/episodes/mpc/seed_20494010 --video --speed 0.5    # replay.gif + .mp4 + last frame
python -m controller.replay ... --video --all-paths                     # every candidate of every round (needs --save-all-plans at run time)
```

Replay colours: thick green the chosen imagined hand path, faint green the five runner-ups, orange to red the
other candidates (the redder, the farther from the chosen path), purple the imagined cube, blue the hand's
actual trail. Any method replays (from the saved simulator states); only `mpc` has the plans on top.

## The two experiments behind the imagined-SAC table

```bash
python -u -m controller.imagined_sac --out data/sac_imagined_v1                      # one policy, needs data/episodes_v3 with latents.npy assembled
python -u -m controller.sac_scratch --steps 40000 --episodes 30 --out data/sac_scratch_v1   # both policies + 30 empty + 30 obstacle camera scenes
python -u -m controller.mpc_counterfactual --source-episode data/mpc_02/episodes/mpc/seed_20494010 --out data/counterfactual
```

`imagined_sac.py` is the exact-state SAC's recipe (behaviour cloning, critic warmup, a cloning term in every
actor update) with every training transition imagined: an episode starts at a recorded state of the data,
`D` imagines the next (z, p), `Q` reads the cube and held, the reward is the same `GoalReward` on `Q`'s
readings plus `R`'s penalties; the checkpoint is picked by imagined placements. It never improved on its
cloned start (14-18/20 cloned, 10/20 after 40k imagined transitions on the exact state): the imagination is
faithful along the recorded actions (cube 1.0 -> 1.5 cm over 10 steps, 10-step return correlation 0.98),
but the updates move the policy off them, and the critic barely fits a reward where 0.5 % of the steps
carry the placement bonus. `mpc_counterfactual.py` answers "what if the MPC executed a worse plan": on the
scene it was run on even the 64th best plan still placed, slower and less precisely (171 steps and 6.4 cm against
125 and 4.4 cm for the best), because the next step re-plans from the new image; only switching to the 64th
after 60 good steps lost the cube (`../../results/control_release_v1/runs/mpc_counterfactual_seed20496039/summary.json`).

## The live window

Without `--headless` and with a display a MuJoCo window in a separate process mirrors the episode (the
episode renders through EGL, the window needs GLX, `window.py` finds a setup). `--speed` paces it.
