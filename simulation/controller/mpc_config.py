# Knobs of the mpc planner. Every episode saves the values it ran with (mpc_settings.json), and a copy of
# this file can be passed with --mpc-config for a variant (mpc_variants/).
# Each control step: shift the previous plan by one, sample POPULATION plans of HORIZON actions around it,
# imagine them with D, score them, keep the ELITES best, refit ITERATIONS times, execute the first action.

# --- search ---

# control steps of 0.1 s the planner looks ahead, D's errors grow beyond its training windows
HORIZON = 8

# candidate plans per round
POPULATION = 64

# best plans kept per round to re-centre the sampling
ELITES = 8

# sample-and-refit rounds per control step
ITERATIONS = 3

# noise of the first round in action units (actions live in [-1, 1])
NOISE_STD = 0.5

# floor on the noise after refitting so the search never collapses onto one plan
NOISE_MIN = 0.1

# share of the noise that is the same for the whole plan, so candidates curve instead of jittering
NOISE_SMOOTH = 0.5

# who decides the gripper: "switch" = the planner with one switch time per plan, "sampled" = the planner
# with open/close drawn every step (flickers), "rule" = close at the cube and open over the goal from Q's reading
GRIPPER = "switch"

# "rule" gripper only: close when this close to the cube centre (m)
RULE_CLOSE_DISTANCE = 0.02

# ... release when the held cube is this close to the target horizontally (m)
RULE_RELEASE_RADIUS = 0.03

# ... and at most this far above its resting height (m)
RULE_RELEASE_HEIGHT = 0.02

# --- score: task reward ---

# "reward" = discounted sum of the GoalReward on Q's readings plus R's and O's penalties
# "progress" = the same plus the task potential of every imagined state, so getting there early pays
SCORE = "progress"

# weight of the per-step potential with SCORE = "progress"
PROGRESS_WEIGHT = 1.0

# after an imagined placement the potential stays at its maximum instead of dropping to 0, otherwise the
# best plan always put the release at the last step of the horizon and never let go
PLACED_KEEPS_POTENTIAL = True

# discount of the imagined rewards
DISCOUNT = 0.99

# multiplier on R's and O's imagined penalties
PENALTY_SCALE = 1.0

# in imagination the cube counts as placed after this many settled steps (the real test needs more than the
# horizon, so placing never looked worth it)
PLACED_AFTER = 3

# in imagination the cube counts as placed only within this distance of the goal (m), tighter than the task's
# radius as a margin for Q's error
PLACED_RADIUS = 0.03

# --- score: the pull (does not fade with distance) ---

# minus this times the metres from the gripper to the cube (from the cube to the goal once held), averaged
# over the plan; 0 = off. The task reward is nearly flat far from the cube, so a short plan saw no reason to approach
REACH_PULL = 3.0

# the pull measures the whole way still to go (gripper to cube, then cube to goal), so finishing a stage
# never costs; without this grasping made the pull jump and the planner never closed the fingers
PULL_WHOLE_WAY = True

# the pull towards an unheld cube uses Q's reading of the current frame, not the imagined cube (off the
# expert route D imagines it drifting along with the arm)
CUBE_STAYS_PUT = True

# while carrying, the cube may be at most this times its distance to the goal (plus 1 cm) above its resting
# height, so it comes down as it arrives instead of hovering over the goal
PLACE_SLOPE = 0.5

# while carrying, the cube should be at least this far above its resting height (m) wherever the slope
# allows it, so the lift counts for the task
CARRY_HEIGHT = 0.05

# Q reads a carried cube this much too high (m), subtracted from the heights in the pull
HELD_Z_OFFSET = 0.0

# how much more a missing centimetre of CARRY_HEIGHT costs than a centimetre of distance
CARRY_WEIGHT = 5.0

# while the cube is not held the gripper should stay inside a cone over it (this times the horizontal
# distance), so the hand comes down from above instead of pushing the cube from the side; 0 = off
APPROACH_SLOPE = 1.0

# (Sarvesh's idea) cost on a carried cube that keeps changing height: the mean change of the imagined
# vertical speed while carrying, so steady lifting and lowering are free and wobble is not; 0 = off
CARRY_STEADY = 30.0

# once the cube has been let go at the goal the pull turns around and the gripper keeps at least this far
# from it (m), so it neither grabs nor knocks the cube while it settles; 0 = off
PLACED_CLEARANCE = 0.10

# cost per imagined step of commanding the fingers closed while nothing is held, so the hand stays open on
# the way in and a missed grasp opens again for a retry
EMPTY_GRIP_COST = 0.3

# --- checks on what may be imagined ---

# the cube only counts as held while the imagined fingers are closed around something
HELD_NEEDS_CLOSED = True

# an imagined grasp only counts after the open gripper came within CONTACT_RADIUS of Q's cube, otherwise
# a plan that closes the fingers anywhere imagined a grasp (D and Q never saw failed grasps)
GRASP_NEEDS_CONTACT = True

# a cube let go in imagination stays where the fingers last held it instead of where Q reads it on the
# imagined latent, which drifts toward the goal
RELEASED_STAYS_PUT = True

# how close the imagined gripper has to come to the cube for a grasp to count (m)
CONTACT_RADIUS = 0.02

# --- rule-based variants only ---

# once the cube has been let go at the goal: stop searching, keep the fingers open and lift the arm away
RETREAT_AFTER_PLACE = False

# --- logging ---

# imagined paths saved per step for the replay viewer: the chosen plan plus this many runner-ups
SAVE_RUNNER_UPS = 5
