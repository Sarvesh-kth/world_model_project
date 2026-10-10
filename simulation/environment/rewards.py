import numpy as np

COMPONENTS = ("reach", "grasp", "lift", "transport", "place", "success",
              "fail", "collision", "proximity", "table_hit", "drop", "action")
BONUSES = COMPONENTS[:6]
PENALTIES = COMPONENTS[6:]


# Staged reward from the privileged sim state, each component is stored unweighted
# so training can pick any subset later, the weights only decide the logged total
class Rewards:

  def __init__(self, cfg):
    self.cfg = cfg.rewards
    self.success_radius = cfg.episode.success_radius
    self.safe_distance = cfg.rewards.safe_distance
    self.table_height = cfg.table.height
    self.reset()

  def reset(self):
    self.placed = False
    self.succeeded = False
    self.was_grasped = False
    self.prev_grasped = False

  # Where the episode is currently at
  @property
  def stage(self):
    if self.placed:
      return "done"
    if self.prev_grasped:
      return "transport"
    if self.was_grasped:
      return "grasp"
    return "reach"

  def compute(self, state, action):
    c = dict.fromkeys(COMPONENTS, 0.0)
    obj = state["object_pos"]
    place = state["place_pos"]
    grasped = state["grasped"]
    dist_to_obj = float(np.linalg.norm(state["ee_pos"] - obj))
    dist_to_place = float(np.linalg.norm(obj[:2] - place[:2]))
    height = obj[2] - self.table_height

    # One time bonus when the object is let go and is resting at the target
    settled = not grasped and abs(obj[2] - state["object_rest_z"]) < 0.02
    at_target = dist_to_place < self.success_radius
    if self.was_grasped and settled and at_target and not self.placed:
      self.placed = True
      c["place"] = 1.0

    # Shaped terms, tanh keeps them bounded, reach only matters while the object still has to be picked up
    if grasped:
      c["grasp"] = 1.0
      c["lift"] = float(np.clip(height / self.cfg.lift_height, 0.0, 1.0))
      c["transport"] = 1.0 - float(np.tanh(3.0 * dist_to_place))
    elif not self.placed:
      c["reach"] = 1.0 - float(np.tanh(5.0 * dist_to_obj))

    # One time penalty when the object is released away from the target
    if self.prev_grasped and not grasped and not self.placed and dist_to_place > 2 * self.success_radius:
      c["drop"] = -1.0

    # Per step penalties, proximity ramps from 0 at safe_distance to 1 when any arm link touches an obstacle
    c["collision"] = -float(state["obstacle_contacts"])
    c["proximity"] = -float(np.clip(1.0 - state["obstacle_distance"] / self.safe_distance, 0.0, 1.0))
    c["table_hit"] = -float(state["table_contacts"])
    c["action"] = -float(np.mean(np.square(action[:-1])))

    # One time bonus the first step the object sits placed at the target
    success = self.placed and at_target and settled
    if success and not self.succeeded:
      self.succeeded = True
      c["success"] = 1.0

    # One time penalty when the episode ends without success (gave up, object fell off, ran out of time)
    if state["failed"]:
      c["fail"] = -1.0

    self.was_grasped |= grasped
    self.prev_grasped = grasped

    w = self.cfg.weights
    total = sum(w[k] * c[k] for k in COMPONENTS)
    return total, c, success
