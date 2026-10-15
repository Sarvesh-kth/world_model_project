# Tuning knobs of the standalone MPC planner, method "mpc" in controller/control_pipeline.py.
# Edit the numbers and rerun. Every mpc episode saves a copy of the values it used (mpc_settings.json in its
# episode folder), so runs can always be told apart. To keep several versions side by side, copy this file
# and pass --mpc-config path/to/the_copy.py.
#
# What the planner does every control step (10 per second):
#   1. start from the previous step's plan shifted by one step (the very first plan: no motion)
#   2. sample POPULATION plans of HORIZON actions around it, with noise NOISE_STD (NOISE_SMOOTH of it shared
#      over the whole plan); the gripper as one switch time per plan
#   3. imagine every plan with D, read every imagined state with Q, score it: GoalReward on Q's reading plus
#      R's penalties plus the obstacle model O's at the imagined hand plus a pull towards the cube (towards B
#      once it is carried)
#   4. keep the best ELITES plans, re-centre and narrow the sampling on them, repeat ITERATIONS times
#   5. execute only the first action of the best plan; the next step starts over from the new camera image
# No policy and no hand-coded motion anywhere in this loop: every action is a sampled raw action. jepa_mpc
# (planner.py) is the rule-based counterpart, its candidates are waypoints that a go-to rule turns into actions.
#
# The defaults are the PURE version (Calle, 2026-10-11): the planner decides every action, the gripper too,
# with no hand-coded behaviour. What is hand-made is only the score (the pull and the checks on what may be
# imagined). Two other configurations, for clearly labelled comparison rows:
#   rule-based gripper (9/20 on a Mac, 2026-10-10): GRIPPER = "rule", SCORE = "reward", RETREAT_AFTER_PLACE = True,
#     GRASP_NEEDS_CONTACT = False, PLACED_CLEARANCE = 0.0, CARRY_WEIGHT = 1.0, EMPTY_GRIP_COST = 0.0,
#     PLACED_RADIUS = 1.0, PULL_WHOLE_WAY = False, HELD_Z_OFFSET = 0.0, CARRY_STEADY = 0.0
#   the bare task reward, no pull and no checks (never touched the cube on the old models): GRIPPER = "sampled", SCORE = "reward",
#     REACH_PULL = 0.0, CUBE_STAYS_PUT = False, HELD_NEEDS_CLOSED = False, GRASP_NEEDS_CONTACT = False,
#     PLACED_AFTER = 15, PLACED_CLEARANCE = 0.0, EMPTY_GRIP_COST = 0.0, PLACED_RADIUS = 1.0, PULL_WHOLE_WAY = False,
#     CARRY_STEADY = 0.0


# --- the search ---

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
#   "switch"   the planner: each candidate plan switches the gripper at most once, at a sampled step (keep
#              the current command until then, the opposite after); the switch time is refitted like the
#              motions. One switch per 0.8 s plan is enough to close or release, and avoids plans that flicker
#   "sampled"  the planner, open/close sampled independently at every step (plans flicker open and closed)
#   "rule"     NOT the planner: a fixed rule from Q's reading closes the gripper at the cube and opens it over
#              B (the rule-based comparison row)
GRIPPER = "switch"

# the "rule" gripper only: close when the gripper is this close to the cube centre (m) ...
RULE_CLOSE_DISTANCE = 0.02

# ... release when the held cube is this close to the target horizontally (m) ...
RULE_RELEASE_RADIUS = 0.03

# ... and at most this far above its resting height (m)
RULE_RELEASE_HEIGHT = 0.02


# --- the score: the task reward ---

# what a plan is scored with:
#   "reward"    the discounted sum of the GoalReward on Q's readings plus R's and O's penalties
#   "progress"  adds the task potential of every imagined state, so getting there early pays (the pure
#               version: with "reward" a grasp at plan step 1 or 7 scored the same and the planner procrastinated)
SCORE = "progress"

# weight of the per-step potential when SCORE = "progress"
PROGRESS_WEIGHT = 1.0

# with SCORE = "progress": after an imagined placement the potential stays at its maximum (every shaped term
# fulfilled: 2.5) instead of the reward's 0 for a finished episode. With 0 a plan lost potential for every step
# after placing (about 2.0 while carrying near B), so the best plan always placed at the last step that still
# fit in the horizon, and the release never came (v3c smoke, seed 20494010: from step 73 on every chosen plan
# expected a placement, the cube stayed held 1-3 cm from B until the episode ended)
PLACED_KEEPS_POTENTIAL = True

# discount of the imagined rewards (the SAC uses 0.99)
DISCOUNT = 0.99

# multiplier on R's imagined collision / proximity / table / empty-grasp penalties and on the obstacle model O's
# (v3c jepa_mpc uses 1 too). With O (v3c, 20 obstacle scenes, seeds 20496010-029, cluster) 1 and 2 both gave 0
# obstacle contact steps; 1 placed 13/20, 2 placed 11/20 (stuck in front of an obstacle more often). On the old
# models (R only, no O) 1 collided far more (287 contact steps vs 12), so the choice depends on O
PENALTY_SCALE = 1.0

# in imagination the cube counts as placed after this many settled steps (the real test needs 15, beyond the
# 8-step horizon, and letting go first loses the carrying reward, so placing always looked like a loss)
PLACED_AFTER = 3

# in imagination the cube counts as placed only within this distance of B (m), instead of the task's 7 cm: a
# margin for Q's few cm of error. With 7 cm the planner let go 8-13 cm from B in 5 of 7 carries (Q thought it
# was inside); letting go beyond twice this distance also counts as a drop in the task reward
PLACED_RADIUS = 0.03


# --- the score: the pull (hand-made, does not fade with distance) ---

# minus this times the metres from the gripper to the cube (from the cube to B once it is held), averaged over
# the imagined steps; 0 = off (the bare task reward). The task reward's reach term is nearly flat beyond
# 30 cm, so an 8-step plan saw no reason to approach; at 1 the first-frame margin of "towards the cube" over
# "stand still" was only 0.03, below the search noise
REACH_PULL = 3.0

# the pull measures the whole way still to go: before the grasp the gripper's way to the cube plus the cube's
# way from where it rests to B (and the lift it still needs), after the grasp only the cube's way. Finishing a
# stage then never costs; with False grasping made the pull jump from the cm to the cube to the half metre to
# B, and the pure planner never closed the fingers (reaching the cube, then waiting with open fingers)
PULL_WHOLE_WAY = True

# the pull towards an unheld cube measures to Q's reading of the current frame, not to the cube in imagination
# (an unheld cube does not move; off the expert route D imagines it drifting along with the arm, and in 4 of
# 20 benchmark seeds the arm went to B believing it was approaching the cube)
CUBE_STAYS_PUT = True

# while carrying, the cube may be at most this times its distance to B (plus 1 cm) above its resting height,
# so it comes down as it arrives (0.5: 10 cm high at 20 cm away, on the table at B). Under 1, so getting closer
# to B always pays. Without it the planner carried the cube to B and hovered 5-10 cm high
PLACE_SLOPE = 0.5

# while carrying, the cube should be at least this far above its resting height (m) wherever the glide slope
# allows it (farther than (CARRY_HEIGHT - 1 cm) / PLACE_SLOPE from B): the task counts a placement only if the
# cube was lifted 4 cm, and its lift reward stops growing at about 2 cm
CARRY_HEIGHT = 0.05

# Q reads a carried cube this much too high (m); the pull's heights subtract it. The old Q (combined_test1) said
# 6-7 cm while the cube was 2.7-3.8 cm up (13 carries, 2026-10-11), so the planner carried it too low for the
# 4 cm the task needs: 0.03 then. The v3c Q has no such bias: on the 5,977 held states of episodes_v3's val and
# test splits its height error is -0.03 cm on average (0.13 cm MAE), at every lift height (2026-10-14)
HELD_Z_OFFSET = 0.0

# how much more a missing centimetre of CARRY_HEIGHT costs than a centimetre of distance: at 1 the planner
# still carried the cube only 3.5 cm high once (cluster, seed 20494025) and the placement did not count
CARRY_WEIGHT = 5.0

# while the cube is not held, the gripper should be above a cone over the cube: at least this times its
# horizontal distance from the cube beyond 2 cm above the cube's centre (1: 8 cm up at 10 cm away). Below it
# every centimetre costs like CARRY_WEIGHT centimetres of distance, so the hand comes down onto the cube from
# above. 0 = off. Without it the pull (straight-line distance) brought the hand down to cube height 5-10 cm
# away, it pushed the cube from the side, and Q's reading (pulled towards a nearby hand, lagging a pushed cube)
# kept it pushing: 3 of 10 v3c camera scenes, two cubes pushed off the table edge
APPROACH_SLOPE = 1.0

# (Sarvesh's idea) cost of a carried cube that keeps changing height: minus this times the mean change of the
# imagined gripper's vertical speed (m per step, per step) while carrying. Steady lifting and lowering cost
# nothing, up-down wobble does: in run F the carried cube reversed direction every 2-3 steps (about 0.3 cm per
# step), since between CARRY_HEIGHT and the glide slope nothing in the score cared about its height. 0 = off
CARRY_STEADY = 30.0

# once the cube has been let go at B the pull turns around: the gripper should keep at least this far from the
# cube (m) instead of reaching for it, so it neither grabs nor knocks the cube while it settles; 0 = off
PLACED_CLEARANCE = 0.10


# cost of commanding the fingers closed while nothing is held, per imagined step (averaged over the plan): the
# hand stays open on the way to the cube, and after a missed grasp opening again pays, so a retry can come out
# of the score instead of a rule. Small against the grasp reward (0.5 and more), so closing at the cube still pays
EMPTY_GRIP_COST = 0.3


# --- what may be imagined (checks on Q's readings of imagined states) ---

# the cube only counts as held while the imagined fingers are closed around something (under 6 cm; open is
# 8 cm). D and Q learned from carries only, so near B they imagined a held cube with the fingers wide open
HELD_NEEDS_CLOSED = True

# a grasp needs open fingers arriving at the cube and closing there: unless it is really held now, an imagined
# grasp only counts after the imagined gripper came within CONTACT_RADIUS of where Q sees the cube now with the
# fingers still open. Without it a plan that closes the fingers anywhere imagined a grasp (D and Q never saw
# failed grasps); requiring only the contact, the planner closed early and drove closed fingers to the cube
GRASP_NEEDS_CONTACT = True

# a cube let go in imagination stays (in x and y) where the fingers last held it, Q's held reading (0.1-0.2 cm
# off), instead of where Q reads it on the imagined latent: a cube resting near B is read about 2 cm from B
# wherever it really is (10 v3c camera scenes, 2026-10-14: let go 5.2, 8.2 and 9.2 cm from B, read 2.1, 1.8 and
# 2.2 cm), so the planner imagined a placement and let go up to 9 cm short (one scene ended 7.5 cm out, the
# success radius is 7)
RELEASED_STAYS_PUT = True

# how close the imagined gripper has to come to the cube for a grasp to count (m): with 0.03 the planner closed
# 3-4 cm away, 2-3 cm above the cube's centre, over its top; the good grasps closed within 0.5-1.8 cm of Q's cube
CONTACT_RADIUS = 0.02


# --- rule-based variants only ---

# once the cube has been let go at B: stop searching, keep the fingers open and lift the arm away (a
# hand-coded behaviour, used by the rule-based comparison row; the pure version uses PLACED_CLEARANCE)
RETREAT_AFTER_PLACE = False


# --- logging ---

# imagined paths saved per step for the replay viewer: the chosen plan plus this many runner-ups
SAVE_RUNNER_UPS = 5
