import dataclasses
import mujoco
import numpy as np

# Obstacle kinds, all tall and thin so the arm has to go around them and not over
# wall: thin tall box 
# box: small square tall box with a random yaw
# cylinder: thin tall pillar
# Mujoco box sizes are half extents, cylinder size is (radius, half length)

OBSTACLE_COLORS = [
  (0.85, 0.15, 0.10, 1.0),
  (0.95, 0.55, 0.05, 1.0),
  (0.60, 0.10, 0.60, 1.0),
]

@dataclasses.dataclass
class Obstacle:
  kind: str
  pos: np.ndarray
  yaw: float
  # wall, box: (half_length, half_width, half_height), cylinder: (radius, half_height)
  size: np.ndarray

  # Top of the obstacle above the table
  @property
  def height(self):
    if self.kind == "cylinder":
      return 2.0 * float(self.size[1])
    return 2.0 * float(self.size[2])

  # Radius of a circle around pos that covers the whole footprint, used by the route planner
  @property
  def footprint_radius(self):
    if self.kind == "cylinder":
      return float(self.size[0])
    return float(np.hypot(self.size[0], self.size[1]))

  def to_dict(self):
    return {"kind": self.kind, "pos": [float(v) for v in self.pos],
            "yaw": float(self.yaw), "size": [float(v) for v in self.size]}

  @classmethod
  def from_dict(cls, d):
    return cls(d["kind"], np.asarray(d["pos"], dtype=float), float(d["yaw"]),
               np.asarray(d["size"], dtype=float))

# Sample obstacles in the corridor between pick and place (xy on the table)
# there is always at least one, the tall one, None when even that did not fit
def sample_obstacles(cfg, rng, pick, place):
  for _ in range(20):
    obstacles = _sample_once(cfg, rng, pick, place)
    if obstacles:
      return obstacles
  return None

def _sample_once(cfg, rng, pick, place):
  ocfg = cfg.task.obstacles
  kinds = list(ocfg.kinds)
  n = rng.integers(ocfg.count[0], ocfg.count[1] + 1)
  direction = place - pick
  length = np.linalg.norm(direction)
  direction = direction / length
  normal = np.array([-direction[1], direction[0]])
  obstacles = []
  for _ in range(n):
    kind = kinds[rng.integers(len(kinds))]
    yaw, size = _sample_size(kind, ocfg.kinds[kind], rng, direction)
    # only the first placed obstacle has to be tall, the rest can be anything (size[-1] is the half height for every kind)
    if obstacles:
      size[-1] = _uniform(rng, ocfg.extra_height) / 2
    # Try 50 times to find a spot along the corridor that is far enough from pick, place and the other obstacles
    for _ in range(50):
      t = rng.uniform(*ocfg.corridor_span)
      offset = rng.uniform(-ocfg.lateral_offset, ocfg.lateral_offset)
      pos = pick + direction * (t * length) + normal * offset
      far_from_ends = np.linalg.norm(pos - pick) >= ocfg.clearance and np.linalg.norm(pos - place) >= ocfg.clearance
      far_from_others = all(np.linalg.norm(pos - o.pos) >= ocfg.spacing for o in obstacles)
      if far_from_ends and far_from_others and on_table(cfg, pos):
        obstacles.append(Obstacle(kind, pos, yaw, size))
        break
  return obstacles

# Sample the yaw and size for one obstacle kind, sizes in the config are a value or a [low, high] range
def _sample_size(kind, kcfg, rng, direction):
  if kind == "wall":
    # across the corridor, with a bit of random twist
    yaw = np.arctan2(direction[1], direction[0]) + np.pi / 2 + rng.uniform(-0.3, 0.3)
    size = np.array([_uniform(rng, kcfg.length) / 2, _uniform(rng, kcfg.thickness) / 2,
                     _uniform(rng, kcfg.height) / 2])
  elif kind == "box":
    yaw = rng.uniform(-np.pi, np.pi)
    size = np.array([_uniform(rng, kcfg.width) / 2, _uniform(rng, kcfg.width) / 2,
                     _uniform(rng, kcfg.height) / 2])
  else:
    yaw = 0.0
    size = np.array([_uniform(rng, kcfg.radius), _uniform(rng, kcfg.height) / 2])
  return yaw, size

# Add the obstacle to the scene as body obstacle_<index> sitting on the table
# geoms are named obstacle_geom_<index>, the env uses that name to label collisions
def add_obstacle(spec, cfg, index, ob):
  quat = [np.cos(ob.yaw / 2), 0, 0, np.sin(ob.yaw / 2)]
  body = spec.worldbody.add_body(name=f"obstacle_{index}",
                                 pos=[ob.pos[0], ob.pos[1], cfg.table.height], quat=quat)
  rgba = list(OBSTACLE_COLORS[index % len(OBSTACLE_COLORS)])
  name = f"obstacle_geom_{index}"
  if ob.kind == "cylinder":
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[ob.size[0], ob.size[1], 0], pos=[0, 0, ob.size[1]], rgba=rgba)
  else:
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(ob.size[:3]),
                  pos=[0, 0, ob.size[2]], rgba=rgba)
  return body

# Check the xy position is on the table, with a small margin from the edge
def on_table(cfg, pos, margin=0.04):
  c = np.array(cfg.table.center)
  h = np.array(cfg.table.half_size) - margin
  return bool(np.all(np.abs(np.asarray(pos) - c) <= h))

# A fixed number or a [low, high] range from the config
def _uniform(rng, value):
  if isinstance(value, (list, tuple)):
    return float(rng.uniform(value[0], value[1]))
  return float(value)
