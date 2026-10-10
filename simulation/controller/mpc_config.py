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
#      R's collision penalties
#   4. keep the best ELITES plans, re-centre and narrow the sampling on them, repeat ITERATIONS times
#   5. execute only the first action of the best plan; the next step starts over from the new camera image
# There is no SAC anywhere in this loop, which is what makes it the standalone baseline (jepa_mpc is the same
# search anchored to the SAC's plan).


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
GRIPPER = "sampled"

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

# discount of the imagined rewards (the SAC and jepa_mpc use 0.99)
DISCOUNT = 0.99

# multiplier on R's imagined collision / proximity / table penalties (jepa_mpc uses 2; 3 was too cautious)
PENALTY_SCALE = 2.0

# imagined paths saved per step for the replay viewer: the chosen plan plus this many runner-ups
SAVE_RUNNER_UPS = 5
