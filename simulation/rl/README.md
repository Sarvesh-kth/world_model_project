# Reinforcement learning: the exact-state SAC baseline

The SAC policy is trained on the exact simulator state and then frozen. It is used three ways:
`rl_true` runs it on the exact state (what the policy can do with perfect perception), `rl_q` runs
it on `Q`'s camera estimates (is the perception good enough to drive it), and the planner uses its
action sequence as the guide the CEM searches around (`controller/README.md`).

```
task_control.py   GoalReward (the reward), policy_observation (the 41 inputs), TaskSession (the simulator
                  wrapper every controller runs in), make_rl_env (gym wrapper for SB3)
rl_control.py     demonstrations -> behaviour cloning -> SAC, validation and checkpoint selection
rl_baseline.py    the command: train, evaluate on fresh seeds, write the run folder the others load
```

## The task as the policy sees it, `task_control.py`

**Observation**, `policy_observation(p, xyz, held, memory, goal, rest_z, remaining)`, 41 numbers. The same
function is called with the exact cube position (`rl_true`, training) and with `Q`'s estimate (`rl_q`,
planner), which is what makes the policy transferable. Fixed scales turn centimetres into numbers near 1.
`origin` is `(0, 0, rest_z)`, the cube's resting height.

| index | content | scale |
|---|---|---|
| 0-6 | joint positions | / pi |
| 7-13 | joint velocities | / 2 |
| 14-16 | grasp site xyz - origin | x 5 |
| 17 | hand yaw | / pi |
| 18 | gripper width | x 25 |
| 19 | gripper command | +1 open / -1 closed |
| 20-22 | cube xyz - origin | x 5 |
| 23-25 | cube - grasp site | x 20 |
| 26-28 | goal - origin | x 5 |
| 29-31 | goal - cube | x 5 |
| 32 | held | 0 / 1 (probability for `rl_q`) |
| 33-38 | reward memory: was grasped, held last step, ever lifted, settle fraction, succeeded, potential | |
| 39 | rest_z | |
| 40 | fraction of the episode left | |

The reward memory makes the task Markov for the policy: whether the cube has been lifted already
decides what the next move should be. For the camera controllers the memory is advanced from `Q`'s
estimates (`predicted_reward`), never from the true state.

**Reward**, `GoalReward(cfg)`, built on `environment.rewards.Rewards`:

```
potential  Phi = 0.5 reach + 0.5 grasp + 0.5 lift + 1.0 transport        (weights from rewards.weights, 0 when the episode is over)
progress   gamma * Phi(next) - Phi(now)                                   gamma = 0.99
goal       +15 once, when the cube has been lifted >= 4 cm, released, and rests within 7 cm of B for 15 steps
penalties  0.5 collision + 0.3 proximity + 0.2 table_hit + 1.0 drop + 0.01 action, and -10 when the episode fails
time       -0.01 per step
```

Potential-based shaping means holding still earns nothing and the shaped rewards telescope: over an
episode they sum to the potential gained, so the policy is paid for reaching, grasping, lifting and
carrying but cannot farm any of it. `predicted_reward` evaluates the same formula on an estimated state;
contacts are unknown there (the penalty head `R` adds them in the planner).

**TaskSession(cfg, layout, jitter)** wraps `PickPlaceEnv`: `reset(seed)` jitters the cube start by
`jitter` per axis and returns the exact-state observation; `step(action)` steps the environment, updates
the statistics, computes the control reward and ends the episode on success, a fallen cube or the
deadline (300 steps). Success is judged here, not by the environment: ever held, lifted 4 cm, now
released, within `success_radius` of the goal, within 2 cm of the rest height, for `settle_steps` steps.
`result()` is what every controller reports: `task_success`, `steps`, returns, `ever_held`,
`maximum_lift_cm`, `minimum_goal_distance_cm`, `final_goal_distance_cm`, contact counts, the layout.
`make_rl_env` is the gymnasium wrapper SB3 trains on (each reset takes the next seed).

## Training, `rl_control.py`

`train(root, cfg, layout, args)` in five steps:

1. **Demonstrations** (`demonstrations`): the scripted policy runs on seeds `seed, seed+1, ...` until
   `--demos` (100) episodes succeeded, failed attempts are dropped, every step is one transition
   `(obs, action, next_obs, reward, done, truncated)`. Saved to `demos.npz`, reused on a rerun.
2. **Behaviour cloning**: the SAC actor's mean is fitted to the demonstrated actions with Adam (lr 1e-3,
   `--bc-epochs` 240, batches of 256); the loss is the MSE of the four motion numbers plus twice the MSE
   of the gripper. The actor's log-std is then set to -2 so exploration starts narrow around the clone.
3. **Replay buffer**: all demonstration transitions are added, so the critic sees successful episodes
   from the first update. The cloned actor is validated and saved as `bc_initial.zip`.
4. **Critic warmup**: `--critic-warmup` (2000) gradient steps with the actor frozen, so a random critic
   does not pull the cloned actor apart in the first updates.
5. **SAC** (SB3, MlpPolicy 128-128, lr 3e-5, `ent_coef auto_0.005`, gamma 0.99, batch 256,
   `learning_starts` 1000, `--rl-steps` 20000). A hook on the actor optimiser adds
   `--bc-weight` (100) times the BC loss on 256 random demonstration samples to every actor update, so RL
   improves on the demonstrations without wandering away from them. Every `--eval-every` (2500) steps the
   policy is validated on `--validation-episodes` (20) fixed seeds (`seed + 10000 + i`); the checkpoint with
   the most placements, then the smallest goal distance, is kept as `sac_best.zip`.

RL was observed to erode the warm start after a few thousand steps in later experiments, which is why the
selection is by validation and the best checkpoint is often an early one.

## The command, `rl_baseline.py`

```bash
python -m rl.rl_baseline --out data/rl_baseline_v1
```

Loads `configs/grade_e.yml` (no posture noise, cube) with `terminate_on_success` off (the session ends
episodes itself) and `configs/grade_e_layout.json` (cube at (0.18, -0.23), target at (0.10, 0.27), no
obstacles), trains as above, then evaluates `bc_initial.zip` and `sac_best.zip` on `--test-episodes` (100)
fresh seeds (`seed + 990000 + i`). The run folder:

```
data/rl_baseline_v1/
  pipeline.json          inputs: config, layout, options   <- the collectors read config and layout from here
  demos.npz, demos.json  the demonstrations and every attempt's outcome
  models/sac_best.zip    the frozen policy the controllers load
  models/bc_initial.zip  the cloned actor before RL
  models/rl_training.json  BC validation, every validation, the selected checkpoint, settings
  evaluations.json       bc_true and rl_true, every test episode's result
  summary.json           the headline numbers
  rl_monitor.csv, rl_logs/  SB3 logs
```

The policy in `data/rl_baseline_v1` passed the 90 % placement gate on 100 fresh episodes and places
20/20 on the controller benchmark (`CHANGES_FROM_M2.md`). Videos of it come from the controller:
`python -m controller.control_pipeline --methods rl_true`.

Seeds, so nothing overlaps: demonstrations `seed + attempt`, training environment `seed + 1000` upward,
validation `seed + 10000 + i`, test `seed + 990000 + i`; the controller uses its own seed plus 20000
(default 20494010 upward) and the collectors their own (30464005, 40464005 upward).
