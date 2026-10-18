# What could be done next

State on 18 October 2026: the latent MPC (`mpc`) places 26/40 on fresh scenes with 0-3 obstacles with zero
contacts; the simulator-trained SAC matches it on the exact state (27/40, 91 contact steps) and loses half
through the cameras (14/40). The numbers and the reasons are in `results/README.md`. Everything below runs
from `simulation/` with the venv active.

## 1. The SAC on Q-like perception

The policy was trained on the exact cube and an immediate contact flag, and on the cameras it closes 1-2 cm
off, loses the cube after a few steps and carries nothing to the goal. Train it on what it will see: a lagged
and noisy "held" (Q's held lags a real grasp by about 12 steps), 2 cm of cube noise instead of 1
(`--cube-noise 0.02`), or Q's own readings from the recorded latents of `episodes_v3`; and a retry after a
slip. Demonstrations are the lever (`rl/README.md`), so more of them on 1-3 obstacle scenes first. Then
`bash sac_priv_eval.sh rl_q 20 data/<run>` against the MPC on the same seeds.

## 2. Retire the gripper envelope and the release rule of `jepa_mpc` (data collected, chain half done)

`data/episodes_hack` has 150 episodes (20,455 states) of `configs/hack_mix.yml`: fingers shut mid-travel
toward the cube (`travel_close`), the cube let go 5-14 cm short of the goal and then placed properly
(`short_release`), and misses. On a fresh clone first pull the data through LFS and unpack it (root
`README.md`, Install), then:

```bash
python -u -m world_model.prepare_episodes --episodes data/episodes_hack/episodes_raw --run data/episodes_hack
python -u -m world_model.encode --run data/episodes_hack --pca data/episodes_v3/features/pca.npz --resume   # ~1.5 h on an RTX 4060, ~30 min on the cluster
python -u -m world_model.merge_runs --runs data/episodes_v3 data/episodes_hack --out data/episodes_v4 --overwrite
bash world_model/run_combined.sh data/episodes_v4 v4 40 retrain          # Q, R, D, O, C -> attempts/v4 (D is the long stage)
# package as cluster_round.sh does: models/v4/{models/*.pt, info.json, features/{pca.npz, meta.json without keys}}
B="--models-run models/v4 --headless --test-episodes 20 --random-ends --obstacles 0 2 --methods jepa_mpc"
python -u -m controller.control_pipeline $B --out data/control_v6c_02
python -u -m controller.control_pipeline $B --no-grasp-prior --out data/control_v6n_02
python -u -m controller.control_pipeline $B --no-release-rule --out data/control_v6r_02
python -u -m controller.control_pipeline $B --no-grasp-prior --no-release-rule --no-width-gate --out data/control_v6x_02
```

If `--no-grasp-prior` holds its rate with the v4 models, make it the default; same for the release rule. The
width rules cost nothing in the ablation (14/20 vs 15/20, noise) and can be switched off now. The `mpc`
would gain from the same data: its `GRASP_NEEDS_CONTACT` and `RELEASED_STAYS_PUT` checks stand in for it.

## 3. Grasping, the failure every variant shares

Seeds 17, 20, 23 fail in every variant: the hand never grasps beside a box or at the table edge, Q 1-2 cm
off; two of the ten local empty scenes have Q 4-5 cm off along y all episode (cubes near y = -0.3 / -0.4).
Options: more `lateral` and `miss` episodes at those spots; a readout trained with the planner's own episodes
(the controller saves frames, online latents and the true state per episode); a finer static grid for Q only.

## 4. Variance

The same 20 seeds move by about 10 % between runs of the same code (the fp16 encoder, and the CEM amplifies
small differences). Quote 20-scene rates, and for a decision between two variants run both twice or use 40
scenes. A deterministic torch setting (`torch.use_deterministic_algorithms`, fp32 encoder) would make runs
repeatable at a cost in speed; worth it for ablations.

## 5. Speed (now about 2 Hz with the GPU to itself)

Per step: render + maps 36 ms, V-JEPA encode of two 16-frame clips about 250 ms, planner 180-270 ms, C
12 ms. Cheapest cuts, each a flag or a small change, none measured for the success rate yet: 8 wrist frames
(`encode.py --wrist-clip-frames 8`, needs a re-encode and retrain), 2 CEM rounds, population 48, encode every
second step and plan on the last latent in between, R retrained as penalties only. 10 Hz needs a different
encoder.

## 6. The cost-to-go, honestly

`C` is learned, but its labels are a grid planner on synthetic scenes: a distilled planner. On 0-2 obstacles
it is within noise of the plain obstacle model (15/20 vs 16/20) and it fixed the one failure the plain model
has (parking beside an obstacle holding the cube). Keep it optional unless 3-obstacle scenes matter; then
measure both on 40 scenes. The `mpc` does not use it yet; its pull is the straight line, which is why it
sometimes waits in front of a tall obstacle near the pick-to-place line (3 of its 9 failures on the cluster's
obstacle scenes).

## 7. Twenty scenes per count on the cluster

The status table has 10 scenes per obstacle count; 20 per count (`bash sac_priv_eval.sh mpc 20`, then
`rl_q` and `rl_true`, about 5 hours on a free GPU) would settle the rates.
