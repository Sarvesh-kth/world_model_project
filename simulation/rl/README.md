# rl

The SAC policies and the simulator wrapper every controller runs in.

```
task_control.py   GoalReward (the shaped reward), policy_observation (what the policy sees), hand_window (the obstacle
                  map around the hand), TaskSession (the simulator wrapper every controller runs in), make_rl_env (gym wrapper)
rl_control.py     the training: demonstrations and recorded episodes -> behaviour cloning -> critic warmup -> SAC with a BC term,
                  validation every eval_every steps, the best checkpoint kept as sac_best.zip
rl_baseline.py    the command: train, test on fresh seeds, write the run folder the controllers load
```

Two policies exist:

- `models/sac`, the simulator-trained SAC for the MPC's scenes (cube and goal anywhere, 0-3
  obstacles); the controllers' default `--baseline-run`.
- a SAC on one fixed layout (`configs/fixed_scene_layout.json`), 41 inputs, no obstacles, which
  `rl_baseline` trains without `--random-ends`.

```bash
# the simulator SAC on the MPC's scenes: CPU only, 300k steps planned (the packaged one is the 20k checkpoint)
python -u -m rl.rl_baseline --out data/sac_new --random-ends --obstacles 0 3 --obstacle-map both \
  --recorded-run data/episodes --cube-noise 0.01 --net 256 256 --demos 500 --rl-steps 300000 \
  --eval-every 20000 --validation-episodes 30 --validation-empty 10 --validation-obstacles 1 3 --test-episodes 30
# the fixed-layout SAC
python -m rl.rl_baseline --out data/rl_baseline_v2
# then on the cameras, same scenes as the MPC
SAC=data/sac_new bash ../evaluate.sh rl_q 10
```

`data/episodes/manifest.json` is enough for `--recorded-run` (no frames or latents needed). A run folder
has `models/sac_best.zip` (the policy), `models/bc_initial.zip` (the cloned actor before RL),
`models/validation_history.json`, `pipeline.json` (config, layout and options; the controller reads
`obstacle_map` from it), `evaluations.json` and `summary.json` (the cloned actor and the policy on fresh seeds).

## What the policy sees and is rewarded with

`policy_observation`: joint positions and velocities, grasp site, yaw, gripper width and command, the cube
relative to the table and to the hand, the goal relative to the table and to the cube, held, the reward's
memory (was grasped, held last step, ever lifted, settle fraction, succeeded, potential), the rest height and
the fraction of the episode left: 41 numbers. The same function is called with the exact cube (`rl_true`,
training) and with `Q`'s estimate (`rl_q`), which is what makes one policy run on both. With `--obstacle-map`
a map follows: `window` is the 9x9 cells of the 16x16 obstacle-only depth map around the hand's cell plus the
hand's offset in its cell (83 numbers, the obstacle model's own view), `both` adds the map pooled to 8x8
(147), `full` the whole map (256). The old `--obstacles` alone adds the 8x8 height map (64).

`GoalReward`: potential-based shaping on the environment's reach / grasp / lift / transport terms (gamma
0.99, so standing still earns nothing), +15 once for a placement (lifted 4 cm, released, within 7 cm for 15
steps), the penalties 0.5 collision + 0.3 proximity + 0.2 table hit + 1.0 drop + 0.01 action, -10 on a
failed episode, -0.01 per step. The planners score imagined states with the same formula on `Q`'s readings
(`predicted_reward`).

`TaskSession` wraps the environment for every controller: with `random_ends` a whole new scene per episode
from the environment's sampler and the models' own config (`models/world_model/info.json`), `obstacles=(low, high)`
fresh obstacles, the obstacle map kept as a running maximum over the episode, `cube_noise` on the cube the
policy is shown, the episode end (success, a fallen cube, 300 steps) and `result()` with everything the
summaries report.

## Training, and what moved it

Demonstrations (scripted successes, `--demos`) and the recorded episodes of `--recorded-run` (every state,
action and reward term of the manifest, the reward replayed from the recorded contacts and proximity; only
the clean scripted placements join the demonstrations in the cloning set, the rest fills the replay buffer)
-> behaviour cloning of the actor (240 epochs) -> 2,000 critic-only updates -> SAC with a cloning term in
every actor update (weight 100). Validation on 10 empty and 20 obstacle scenes every 20k steps picks the
checkpoint.

What was tried for the MPC's scenes (validation placements of 30):

| obstacle input | demonstrations | cloning only | best with RL |
|---|---:|---:|---:|
| the whole 16x16 map | 200 | 13 | 15 at 60k, 1,000-1,900 contact steps, worse with time |
| 9x9 window around the hand | 200 | 11 | 17 at 20k |
| window + the map pooled 8x8 | 500 | 18 | 20 at 20k (9 empty, 11 obstacles) = `models/sac` |
| window + pooled map | 1200 | 20 | 18 at 20k, the fewest contacts |

A net over the whole map has to memorise layouts; the window is the same geometry seen from the hand
wherever it is. The policy needs far more data than the world model: 401 recorded episodes trained the
whole world model, while the policy needed 500 demonstrations on top of those episodes, and 1,200 to get
there from cloning alone. Demonstrations are the lever; RL adds a placement or two in the first 20-40k steps and then
drifts. The next step would be training on `Q`-like perception (a lagged, noisy "held", 2 cm of cube noise)
and a retry after a slip, which is where the policy loses half of its placements on the cameras.

Seeds never overlap: demonstrations `seed + attempt`, training `seed + 1000` upward, validation
`seed + 10000 + i`, test `seed + 990000 + i`, the controllers `seed + 20000 + i`.
