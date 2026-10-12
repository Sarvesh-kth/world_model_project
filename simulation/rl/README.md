# RL: the exact-state SAC baseline

A SAC policy trained on the exact simulator state and then frozen. In the controller comparison it is the
actor-critic: `rl_true` runs it on the exact state, `rl_q` on `Q`'s camera estimates. The planner does not
use it.

```
task_control.py   GoalReward (the shaped reward), policy_observation (the 41 inputs, 105 with the height map),
                  TaskSession (the simulator wrapper every controller runs in), make_rl_env (gym wrapper)
rl_control.py     demonstrations -> behaviour cloning -> critic warmup -> SAC with a BC term, validation, checkpoint selection
rl_baseline.py    the command: train, evaluate on fresh seeds, write the run folder the controllers load
```

```bash
python -m rl.rl_baseline --out data/rl_baseline_v1                      # about 1 h; the frozen policy in models/sac_best.zip
python -m rl.rl_baseline --out data/rl_obstacles_v2 --obstacles 1 3     # obstacle-aware: fresh obstacles per episode, the 8x8 height map in the observation
```

## The task as the policy sees it

Observation (`policy_observation`): joint positions and velocities, grasp site, yaw, gripper width and
command, cube position, cube relative to the hand, goal, goal relative to the cube, held, a reward memory
(was grasped, held last step, ever lifted, settle fraction, succeeded, potential), the rest height, the
fraction of the episode left, and with `--obstacles` the 64 height-map cells. The same function is called
with the exact cube (`rl_true`, training) and with `Q`'s estimate (`rl_q`), which is what makes the policy
transferable; the reward memory makes the task Markov.

Reward (`GoalReward`): potential-based shaping on the environment's reach / grasp / lift / transport terms
(`gamma 0.99`), +15 once for a placement (lifted 4 cm, released, within 7 cm for 15 steps), the penalties
0.5 collision + 0.3 proximity + 0.2 table hit + 1.0 drop + 0.01 action, -10 on a failed episode, -0.01 per step.

`TaskSession` wraps the environment for every controller: fresh obstacles per seed when asked, the
statistics, the control reward, the episode end (success, a fallen cube, 300 steps) and `result()`:
`task_success`, steps, returns, `ever_held`, lifts, goal distances, contact counts, the layout.

## Training

Demonstrations (100 scripted successes) -> behaviour cloning of the actor (240 epochs) -> the
demonstrations fill the replay buffer -> 2,000 critic-only steps -> SAC (SB3, 128-128, lr 3e-5, 20k steps)
with a BC loss on every actor update (weight 100). Every 2,500 steps the policy is validated on 20 fixed
seeds and the best checkpoint is kept as `sac_best.zip`. The policy in `data/rl_baseline_v1` (tracked, Git
LFS) placed 20/20 on its fixed layout; it only knows that layout (`configs/grade_e_layout.json`).

Seeds never overlap: demonstrations `seed + attempt`, training `seed + 1000` upward, validation
`seed + 10000 + i`, test `seed + 990000 + i`, the controllers `seed + 20000 + i`.
