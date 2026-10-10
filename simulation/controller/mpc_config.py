# Tuning knobs of the standalone MPC planner, method "mpc" in controller/control_pipeline.py.
# Edit the numbers and rerun. Every mpc episode saves a copy of the values it used (mpc_settings.json in its
# episode folder), so runs can always be told apart. To keep several versions side by side, copy this file
# and pass --mpc-config path/to/the_copy.py.
#
# What the planner does every control step (10 per second):
#   1. start from the previous step's plan shifted by one step (the very first plan: no motion)
#   2. sample POPULATION plans of HORIZON actions around it, with noise NOISE_STD (NOISE_SMOOTH of it shared
#      over the whole plan)
#   3. imagine every plan with D, read every imagined state with Q, score it: GoalReward on Q's reading plus
#      R's collision penalties plus a pull towards the cube (towards B once it is carried)
#   4. keep the best ELITES plans, re-centre and narrow the sampling on them, repeat ITERATIONS times
#   5. execute only the first action of the best plan; the next step starts over from the new camera image
# There is no SAC anywhere in this loop, which is what makes it the standalone baseline (jepa_mpc is the same
# search anchored to the SAC's plan). Once the cube has been let go at B the search stops and the arm rises.
#
# The defaults are the best found on 2026-10-10: 9/20 placed on the 20 empty-table test seeds (Mac, MPS;
# every seed reaches the cube, 9 of 10 grasped cubes are placed, the grasp fails in 10). The plain jepa_mpc
# scoring, which fails without the SAC (no pull from afar, phantom grasps, placing beyond the horizon), is
#   GRIPPER = "sampled", REACH_PULL = 0.0, CUBE_STAYS_PUT = False, HELD_NEEDS_CLOSED = False,
#   PLACED_AFTER = 15, RETREAT_AFTER_PLACE = False
# Earlier steps on the same seeds: REACH_PULL 1 without CUBE_STAYS_PUT and CARRY_HEIGHT 8/20 (4 never reached
# the cube); + CUBE_STAYS_PUT and REACH_PULL 3 4/20 (5 set the cube down at B without the 4 cm lift).


# how far ahead the planner imagines, in control steps of 0.1 s; longer sees further, but D's errors grow
HORIZON = 8

# candidate plans per round; more finds better plans, each control step takes proportionally longer
POPULATION = 64

# best plans kept per round to re-centre the sampling; at least 2 and below POPULATION
ELITES = 8

# sample-and-refit rounds per control step
ITERATIONS = 3

# noise of the first round, in action units: actions live in [-1, 1], and 1 means 4 cm per step
NOISE_STD = 0.5

# floor on the noise after refitting, so the search never collapses onto a single plan
NOISE_MIN = 0.1

# share of the noise variance that is the same for the whole plan: 0 independent per step, 1 constant;
# shared noise lets candidates curve around things instead of jittering in place
NOISE_SMOOTH = 0.5

# who decides the gripper:
#   "sampled"  CEM samples open/close like every other action, the planner decides on its own
#   "rule"     a fixed rule from Q's reading: close when the cube is between the fingers, open once it is
#              lowered over the target (an ablation that takes the grasp decision away from the planner)
GRIPPER = "rule"

# the "rule" gripper: close when the gripper is this close to the cube centre (m)
RULE_CLOSE_DISTANCE = 0.02

# the "rule" gripper: release when the held cube is this close to the target horizontally (m) ...
RULE_RELEASE_RADIUS = 0.03

# ... and at most this far above its resting height (m)
RULE_RELEASE_HEIGHT = 0.02

# what a plan is scored with:
#   "reward"    the same score as jepa_mpc: discounted sum of the GoalReward on Q's readings plus R's penalties
#   "progress"  adds the task potential of every imagined state, so getting there early pays
#               (the first ablation to try if the planner wanders instead of committing)
SCORE = "reward"

# weight of the per-step potential when SCORE = "progress"
PROGRESS_WEIGHT = 1.0

# extra pull that does not fade with distance: minus this times the metres from the gripper to the cube
# (from the cube to B once it is held), averaged over the imagined steps; 0 = off, the same score as jepa_mpc.
# The task reward's reach term is nearly flat beyond 30 cm, so an 8-step plan sees no reason to approach;
# at 1 the first-frame margin of "towards the cube" over "stand still" was only 0.03, below the search noise
REACH_PULL = 3.0

# the pull towards an unheld cube measures to Q's reading of the current frame, not to the cube in imagination
# (an unheld cube does not move; off the expert route D imagines it drifting along with the arm, and in 4 of
# 20 benchmark seeds the arm went to B believing it was approaching the cube)
CUBE_STAYS_PUT = True

# part of the pull while carrying: the cube may be at most this times its distance to B (plus 1 cm) above its
# resting height, so it comes down as it arrives (0.5: 10 cm high at 20 cm away, on the table at B). Under 1,
# so getting closer to B always pays. Without it the planner carried the cube to B and hovered 5-10 cm high
PLACE_SLOPE = 0.5

# part of the pull while carrying: farther than 10 cm from B the cube should be at least this far above its
# resting height (m). The task counts a placement only if the cube was lifted 4 cm, and its lift reward stops
# growing at about 2 cm; with REACH_PULL 3 the planner dragged the cube along 2-4 cm low in 5 of 20 seeds and
# set it down at B without it counting. At most the glide slope's height 10 cm from B
CARRY_HEIGHT = 0.05

# in imagination the cube counts as placed after this many settled steps (the real test needs 15, beyond the
# 8-step horizon, and letting go first loses the carrying reward, so placing always looked like a loss)
PLACED_AFTER = 3

# once the cube has been let go at B: stop searching, keep the fingers open and lift the arm away, so it
# cannot grab or knock the cube while it settles (False: keep planning; the rule gripper then re-grabbed
# the cube and the twitching pushed it off the target)
RETREAT_AFTER_PLACE = True

# in imagination the cube only counts as held while the fingers are closed around something (under 6 cm;
# open is 8 cm). False = as jepa_mpc. D and Q learned from carries only, so near B they imagine a held
# cube even with the fingers wide open
HELD_NEEDS_CLOSED = True

# discount of the imagined rewards (the SAC and jepa_mpc use 0.99)
DISCOUNT = 0.99

# multiplier on R's imagined collision / proximity / table penalties (jepa_mpc uses 2; 3 was too cautious)
PENALTY_SCALE = 2.0

# imagined paths saved per step for the replay viewer: the chosen plan plus this many runner-ups
SAVE_RUNNER_UPS = 5
