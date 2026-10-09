# What changed from the M2_Kuba architecture, and why

Branch `world_model_testing`, 9 October 2026. Diagrams: `diagrams/jepa_architecture_old.png`
(problem points P1-P5) and `diagrams/jepa_architecture_new.png` (changes C1-C8). Commands: README,
section "Spatial latent, penalty head and the fixed planner". Every number here comes from a folder under `results/` or `simulation/data/`.

The short version: the pipeline on M2_Kuba had working pieces (frozen V-JEPA, a readout Q, a
dynamics model D, a frozen SAC policy, a CEM planner) wired so that the camera could not actually
tell the models where the cube was, and the planner then amplified that. Nothing in the new version
is a different method; the same pieces are kept and each one is made to do its job.

## 1. Where the pipeline stood

The team's comparison (`results/control_frozen_v2`, 20 episodes on the fixed scene) was:
scripted controller 20/20, SAC with exact state 19/20, the same SAC fed the camera estimates (rl_q)
0/20, and the planner (jepa_mpc) 0/20 with 692 fallbacks. So the robot could do the task when it
knew where the cube was, and could not do it from the camera.

## 2. The five problems

### P1. The latent threw the cube away

V-JEPA 2 turns a 64-frame clip into 32 x 16 x 16 = 8192 tokens, one per 2-frame x 16-px x 16-px
"tubelet", each a 1024-value vector. M2_Kuba then took the mean of all 8192 tokens as the latent z.
In the static camera the cube is about 10 px across, so it falls inside one of the 256 spatial
cells; the arm covers a few hundred cells. After averaging, moving the cube by 11 cm changes z by
8.5 units (token norm about 98), while the patches at the cube's location change by 90-100. The
cube's contribution is below the noise of everything else. The team's own "positions test" (same
robot state and actions, cube under the gripper versus 10 cm away) gave a 12.6-14.4 cm error for
D+Q versus 15.4 cm for a baseline that never sees an image: as bad as not looking.

### P2. Q learned to guess the cube from the robot

The readout Q(z, p) predicts the cube position and whether it is held from the latent and the
20 proprioceptive values p (joint positions and velocities, end-effector xyz and yaw, finger width,
gripper command). On a fixed scene with the cube at A plus or minus 1 cm, and trajectories that all
go to A, the robot's own state tells you the phase of the task and therefore where the cube is. The
proprio-only twin of Q beat the visual Q (0.19/0.50/0.15 cm versus 0.25/0.77/0.23 cm). In closed
loop this shortcut breaks the moment the policy does anything unusual: in rl_q episode 20394005 the
gripper closed on nothing, Q reported held = 0.98 and the cube 11 cm above the table, and the SAC
carried air to B.

### P3. D could not imagine a grasp

SplitDynamics has two heads: a robot head that predicts the next p from (p, a) and a visual head
that predicts the next z from (z, p, a, p'). The finger width is part of p, so it was predicted from
the robot state and the action alone. Whether the fingers stop at 4.5 cm on a cube or close to zero
on air depends on the image, which that head never saw. In every imagined rollout the cube stayed on
the table; the held-cube height error after 30 steps was 10.8 cm, and the "close versus open" effect
on cube height was 12 cm wrong.

### P4. The planner's score was noise, and it resampled the gripper

The CEM asks the frozen SAC for an 8-action guide, samples 64 action sequences around it with
standard deviation 0.6 per action per step, rolls each through D and Q, scores each by the imagined
reward, keeps 8 elites, repeats three times and executes the first action of the best candidate.

Three things went wrong inside that loop.

- The run used `--terminal-weight 0`, so the score was only the discounted sum of the GoalReward
  shaping terms. That reward is potential based (r_t = gamma * Phi(s_t+1) - Phi(s_t)), and a
  discounted sum of such terms telescopes to gamma^8 * Phi(s_8) - Phi(s_0): the planner was
  ranking candidates by the potential of the imagined state 0.8 s ahead, a difference of about
  0.002 per step, below what D's errors produce. In `results/control_test1` the candidates' scores
  differed by 0.01 and the SAC guide, always one of the 64 candidates, was chosen in 0-2 of 300 steps.
- The gripper action is a sign (open or close). Sampling N(1, 0.6) flips it in about 5 % of the
  sampled steps, and with 64 candidates over 8 steps there is always a candidate that closes early.
  Because of P5 the imagination rewarded that (see below), so the real gripper closed beside the
  cube, then Q said "held" and the arm carried nothing to B. Every one of the 0/5 episodes in
  `results/control_test1` ends this way.
- Using the SAC's own critic as the terminal value instead (`--terminal-weight 1`) made it
  worse: candidates beat the guide by 0.5-1.5 purely from D's errors in the imagined state, the
  planner left the guide in 75 % of steps, and the result was 2/5 (`results/control_test2`).

### P5. The data could not teach "closed on nothing"

The full-task data had 30,642 states, every one of them from the frozen SAC or the scripted recovery.
Those controllers only close the gripper on the cube, so the data contains zero states with the
fingers closed on air. To Q and D, "gripper closed" and "cube held" were the same event. Any
controller that ever closes early is then lost: the imagination predicts a grasp, the readout
confirms it, and nothing in the loop can say otherwise.

## 3. The eight changes

### C1. A shorter, faster clip (encode.py)

The clip went from 64 frames (6.4 s of history) to 16 (1.6 s), the model runs in fp16 instead of
bf16, and the frame preprocessing (resize to 292 px, centre crop 256, normalise) runs on the GPU
instead of the Hugging Face CPU processor. Reason: the controller encodes the camera once per
control step (10 Hz, so 100 ms of simulated time per step), and the planner also encodes once per
step. At 0.98 s per clip a 300-step episode spent 5 minutes in the encoder alone and a 20-episode
comparison took hours; at 0.15 s it is 45 s per episode. The simulator does not run in real time,
so this is throughput, not feasibility; on a real robot at 10 Hz only the 8-frame clip (0.07 s)
would keep up. Everything that matters
for the task is in the current frame, and 16 frames still cover the grasp and the lift. bf16 was
dropped because it moves every patch feature by 18 % of its norm relative to fp32 (fp16: 3 %), at the
same speed; that noise matters once single patches carry the signal. The old setting is still
available (`--clip-frames 64 --dtype bf16 --pooling mean_all`).

### C2. Keep where things are: spatial pooling plus PCA (encode.py)

Instead of averaging all tokens, the tokens are reshaped to their 8 time slices x 16 x 16 spatial
grid, and three steps turn that into a 1024-value latent of the same size the rest of the pipeline
expects:

1. Time-mean: average the 8 time slices of the 16-frame clip, so each of the 256 spatial cells is
   one 1024-value vector. This was the key step. The obvious alternative, keeping the last time
   slice only, was unlearnable for D: two consecutive 0.1 s frames moved the top PCA component by
   31 % of its whole-dataset variance (V-JEPA patch features are context mixed, so a background
   patch changes when the arm moves anywhere). Averaging over time brings that frame-to-frame
   jitter down to 4 %.
2. 2x2 spatial average: average each block of 2x2 neighbouring cells, giving an 8x8 grid (each cell
   about 3 cm of table). This halves the jitter again (to 1 % of the variance) while a linear probe
   can still locate the cube from the grid to within 0.86 cm (versus 2.49 cm from proprio alone).
3. PCA to 16 values per cell: a 1024 x 16 projection fitted once, label free, on the patch vectors of
   300 training clips. It keeps 65 % of a cube-induced patch change but only 45 % of background
   change, so the cube's share of the latent goes up. The basis is saved as `features/pca.npz` and
   reused online by the controller, and `--pca` lets other datasets reuse the same basis so their
   features can be trained on together.

z is then 8 x 8 x 16 = 1024 values: a map of the scene rather than a summary of it. Effect: the SAC
fed these estimates placed 5/5 (`results/control_test3`) where it had placed 0/20, with Q reading
the cube within 0.3-0.5 cm per axis from the live camera on scenes it never saw.

### C3. One training set with everything in it (collect_full.py, collect_obstacles.py, prepare_episodes.py, merge_runs.py)

Three datasets were collected and merged into one run with one PCA basis
(`data/combined_test1`, 58,010 states from 400 trajectories):

- the full task with the cube start jittered by 3 cm instead of 1 cm, so Q has to look at the image
  rather than guess (the frozen SAC still succeeds there, so the data stays mostly successful);
- 24 obstacle scenes collected in pairs: the SAC on the empty table, the identical actions replayed
  through 1-3 corridor obstacles, and a scripted route-around demo. Because each obstacle trajectory
  has an empty twin with the same robot motion, a model can only predict the contact penalties from
  the image;
- the project's original collector's full mix (successful, noisy scripted, deliberate drops,
  deliberate collisions, random actions): 120 episodes with 2,586 states where the gripper is closed
  on nothing and 4,740 states with a contact or proximity penalty.

Q, the penalty head and D are trained once on all of it (tag `combined`). Effect on the same 20
test seeds: rl_q 18/20 to 20/20 and the planner 16/20 to 19/20 compared with models trained on the
full-task data alone (`results/control_final_full20` versus `results/control_final_20`).

### C4. The held reading is gated by the finger width (control_pipeline.py)

The readout's "held" probability is set to zero whenever the measured finger width is under 2 cm.
The cube is 4.5 cm wide, so fingers closed to under 2 cm are holding nothing. This uses the robot's
own finger sensor (part of p, which every controller already receives), not simulator state. It is a
hand rule rather than a learned one, chosen because even the combined Q only reaches a held F1 of
0.96 while the finger width alone reaches 0.99. Effect: it removes the "carrying air" state
entirely. With it, the SAC on camera estimates goes from 17/20 to 20/20 and the planner's failures
on the 20 seeds drop from four to one (`results/control_test4_20` versus `results/control_final_20`).
The one remaining failure is a layout where the gripper hovers 3 mm too low and pushes the cube on
every re-grasp.

### C5. D is held to the recorded cube (train.py dynamics --robot-sees-z --task-weight)

Two changes to the dynamics model. SplitDynamicsZ gives the robot head the latent as an extra
input, so the predicted finger width can depend on what is between the fingers. And a
task-consistency term is added to the loss: the readout Q is trained first and frozen, and on every
imagined state of an 8-step rollout it must still output the recorded cube position and held label.
With plain latent MSE a rollout that forgets the cube barely changes the loss, because the cube is
1 of 64 cells while the arm dominates the latent. Effect offline: the imagined grasp F1 after 30
steps 0.56 to 0.67, the held-cube height error 10.8 to 8.6 cm; the first variant to predict part of
the lift on the clear far-offset pairs.

### C6. A penalty head for obstacles (train.py reward)

A small MLP R(z, p) is trained on the recorded, unweighted reward components to output the proximity
term (in [-1, 0], ramping from 0 at 6 cm from an obstacle to -1 at contact), the probability of an
obstacle contact and the probability of a table hit. The planner adds the weighted prediction to every
imagined step of every candidate. On the held-out obstacle scenes the collision AUC is 0.995 on real
latents and 0.99 on latents imagined by D 8 steps ahead; the paired data design means that number
cannot come from the robot state. Effect in closed loop: the planner with the head places 3/6 with
465 contact steps, the same planner with the head switched off 2/6 with 829, and the SAC on camera
estimates 2/6 with 794 (`simulation/data/control_obstacles_final` and `_nopen`).

### C7. The planner is anchored to the policy (control_pipeline.py)

The CEM loop itself is unchanged (64 candidates, 8 steps, 3 iterations, 8 elites). Four things around it changed.

- The gripper follows the SAC guide in every candidate (`--free-gripper` restores the old
  sampling). Whether to close is the policy's decision; the search is over where the arm goes.
- The guide is always candidate 0 and is scored with the same D, Q and penalty head as the others. A
  candidate is executed only if its imagined score beats the guide's forecast by a margin of 0.25
  (`--guide-margin`); otherwise the guide's action is executed. Model noise then cannot move the
  arm, while a forecast collision (several steps at -0.3 to -0.5 each) can. On the empty table the
  score gaps are 0.02 median and 0.05 at the 90th percentile, so the guide is kept in 97 % of
  steps; on obstacle scenes the planner leaves it exactly when the penalty head fires.
- Half of each candidate's noise variance is shared over its whole horizon (`--cem-smooth 0.5`),
  with the per-step standard deviation lowered to 0.3. Independent per-step noise averages out: over
  8 steps it spreads the candidates less than 1 cm sideways, so no candidate could ever route around
  an obstacle even when the head predicted one. With the shared component the hard obstacle scenes
  went from 299 contact steps to 71.
- The terminal value from the SAC critic stays available (`--terminal-weight`) but is 0 by default,
  for the reason in P4.
- The imagined penalties are multiplied by `--penalty-scale` (default 2). At 1 the planner reacts one or
  two steps before a wall and clips it (13-17 contact steps on the scenes it solves); at 3 it is so
  cautious it will not approach a cube standing beside a wall (scenes 19 and 21 lost). At 2 it places
  4/6 obstacle scenes with 0 / 5 / 5 / 20 contact steps (`simulation/data/control_obstacles_final_ps2`).

Effect with the same models as the 0/5 run: 5/5 (`results/control_test1` versus `results/control_test3`).

### C8. Two small fixes

The validity check on predicted finger width rejected anything above 0.082 m; the open gripper
measures 0.083 and D overshoots it by a few millimetres during the carry, so during the carry every
candidate was invalid and the planner fell back to the guide (138 fallbacks in 5 episodes; now 4 in
20). And the pipeline forwarded boolean flags to its episode subprocesses as `--flag False`, which
argparse rejects; they are forwarded as bare flags now.

## 4. What was deliberately not changed

The simulator and its 5-dim action interface, the frozen V-JEPA weights and pinned revision, the
frozen SAC checkpoint, the GoalReward formula, the Q and D network sizes, the CEM population,
iterations, elites and horizon, and the evaluator (strict placement within 7 cm, settled, after a
4 cm lift).

## 5. Final numbers

Empty table, 20 fresh seeds, combined models (`results/control_final_20`): scripted 20/20, SAC with
exact state 20/20, SAC on camera estimates 20/20, planner 19/20 with a median 1.0 cm from B.

Six held-out scenes with 1-3 blocking corridor obstacles (`simulation/data/control_obstacles_final*`):
SAC with exact state 1/6 (923 contact steps), SAC on camera estimates 2/6 (794), planner with the
penalty head at scale 1 3/6 (465, placed at 2.3, 0.8 and 0.3 cm), without it 2/6 (829), and with the
default scale 2 **4/6** (placed at 1.7, 2.6, 1.3 and 4.8 cm with 0, 5, 5 and 20 contact steps; 22 and 23
fail before the grasp as before). Scenes 18, 19 and 21 are solved only by the planner, and only with
the penalty head.

## 6. What the image is and is not responsible for (honest reading)

On the empty-table benchmark the cube only moves 1 cm around its nominal spot, so a controller that
assumes the nominal position is almost as accurate as Q (0.65 cm versus 0.50 cm during the approach).
That benchmark does not need vision; it shows the loop is sound, not that the latent is read. The
places where the image provably carries the result are the obstacle scenes (the penalty head can only
see obstacles in the image, by construction of the paired data) and the 3 cm-jittered and
random-layout data, where the visual Q beats the proprio-only one by a factor of about two
(1.5/2.9 cm versus 2.6/4.7 cm). The "held" signal is now mostly the finger sensor. The `--blind` control
runs the whole loop with the proprio-only readout in place of Q(z, p): the SAC then never reaches the
cube, 0/5 on the empty seeds and 0/4 on the obstacle scenes (`results/control_blind_20`,
`simulation/data/control_obstacles_blind`), while the same loop with the image gives 20/20 and 2/6.

## 7. Still open

- Contact-free avoidance cannot be guaranteed with this structure: the planner looks 0.8 s ahead
  and only deviates from a prior that always heads straight, while the proximity signal starts 6 cm
  from an obstacle. Raising the penalty scale trades contacts for refusing to approach a cube beside a
  wall. Zero contacts needs an obstacle-aware prior: a SAC retrained with the layout in its observation,
  or the scripted route planner as the guide the CEM refines. Not started.

- Scenes 22 and 23: walls right beside the cube put Q 1-2 cm off before the grasp. More obstacle
  data near the pick, or a finer grid for the cube region.
- Detours longer than the 8-step horizon: a 12-16 step horizon drifted with the current D (0/2 on
  the hard scenes). Structured lateral candidates would not need a longer D.
- Scene 18: the penalty head over-predicts contact on a narrow pass and the chosen detour hit the wall.
- The combined Q is about 1 cm less precise on the standard scene than a model trained on it alone.
