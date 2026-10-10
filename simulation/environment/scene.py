import dataclasses
import pathlib

import mujoco
import numpy as np

from .obstacles import Obstacle, add_obstacle, sample_obstacles

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
# Arm
PANDA_XML = ASSETS / "franka_emika_panda" / "panda.xml"
# Camera
D435I_XML = ASSETS / "realsense_d435i" / "d435i.xml"
# Objects
OBJECTS_DIR = ASSETS / "objects"
# Textures to give life
TEXTURES_DIR = ASSETS / "textures"

# Spawn points on the table for object
END_NAMES = ("left", "right", "near", "far")

# Have drop location on opposite end of spawn points
OPPOSITE_END = {"left": "right", "right": "left", "near": "far", "far": "near"}


# Everything that changes between episodes, to_dict / from_dict so a layout can be saved and replayed
@dataclasses.dataclass
class EpisodeLayout:
  object_name: str
  object_scale: float
  object_mass: float
  object_friction: float
  object_color: tuple
  pick_end: str
  place_end: str
  pick_pos: np.ndarray
  place_pos: np.ndarray
  obstacles: list

  # Readable description of episode, for printing
  def summary(self):
    return {
      "object": self.object_name,
      "scale": round(self.object_scale, 3),
      "mass": round(self.object_mass, 3),
      "pick_end": self.pick_end,
      "place_end": self.place_end,
      "pick": [round(float(p), 3) for p in self.pick_pos],
      "place": [round(float(p), 3) for p in self.place_pos],
      "obstacles": [o.kind for o in self.obstacles],
    }

  def to_dict(self):
    return {
      "object": self.object_name,
      "scale": float(self.object_scale),
      "mass": float(self.object_mass),
      "friction": float(self.object_friction),
      "color": [float(c) for c in self.object_color],
      "pick_end": self.pick_end,
      "place_end": self.place_end,
      "pick": [float(p) for p in self.pick_pos],
      "place": [float(p) for p in self.place_pos],
      "obstacles": [o.to_dict() for o in self.obstacles],
    }

  # Load from dict
  @classmethod
  def from_dict(cls, d):
    return cls(d["object"], float(d["scale"]), float(d["mass"]), float(d["friction"]),
               tuple(d["color"]), d["pick_end"], d["place_end"],
               np.asarray(d["pick"], dtype=float), np.asarray(d["place"], dtype=float),
               [Obstacle.from_dict(o) for o in d["obstacles"]])


# Sample an object color
def _sample_color(rng):
  hue = rng.uniform(0, 1)
  c = np.array([abs(hue * 6 - 3) - 1, 2 - abs(hue * 6 - 2), 2 - abs(hue * 6 - 4)])
  c = np.clip(c, 0, 1) * 0.7 + 0.2
  return (*c, 1.0)


# Sample a layout for the episode, have an rng to generate random scenarios
def sample_layout(cfg, rng, holdout=False):
  task = cfg.task
  pool = task.object.holdout if holdout else task.object.pool
  name = pool[rng.integers(len(pool))]
  scale = rng.uniform(*task.object.scale)
  mass = rng.uniform(*task.object.mass)
  friction = rng.uniform(*task.object.friction)
  color = _sample_color(rng) if task.object.randomize_color else (0.5, 0.5, 0.5, 1.0)

  pick_end = task.pick_end
  if pick_end == "random":
    pick_end = END_NAMES[rng.integers(len(END_NAMES))]
  place_end = task.place_end
  if place_end == "opposite":
    place_end = OPPOSITE_END[pick_end]
  elif place_end == "random":
    choices = [e for e in END_NAMES if e != pick_end]
    place_end = choices[rng.integers(len(choices))]

  # Try 200 times to get pick and place far enough apart with at least one obstacle in between
  for _ in range(200):
    pick = _sample_in_end(cfg, rng, pick_end)
    place = _sample_in_end(cfg, rng, place_end)
    if np.linalg.norm(pick - place) < task.min_pick_place_dist:
      continue
    obstacles = sample_obstacles(cfg, rng, pick, place)
    if obstacles is not None:
      break
  else:
    raise RuntimeError("could not sample a layout; check task.ends / task.obstacles config")
  return EpisodeLayout(name, scale, mass, friction, color, pick_end, place_end,
                       pick, place, obstacles)


# Sample a position for pick or place near the end
def _sample_in_end(cfg, rng, end):
  zone = cfg.task.ends[end]
  base = np.array(cfg.robot.base_pos[:2])
  # Have a low and high, hard to pick if object too close to arm base , have only used x y for now
  lo, hi = cfg.task.reach_range
  # Try 200 times to sample a position within a certain reach range from robot base position
  for _ in range(200):
    p = np.array([rng.uniform(*zone.x), rng.uniform(*zone.y)])
    if lo <= np.linalg.norm(p - base) <= hi:
      return p
  raise RuntimeError(f"end zone '{end}' unreachable with reach_range {cfg.task.reach_range}")


# This creates the Mujoco model for one layout
def build_scene(cfg, layout):
  spec = mujoco.MjSpec()
  spec.modelname = "panda_pick_place"
  spec.option.timestep = cfg.sim.timestep
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  # Elliptic cone plus a high impratio makes friction stiff, otherwise a held object slowly slides out of the fingers
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = cfg.sim.impratio
  # Set the extent and center of the scene, only matters for the viewer
  spec.stat.extent = 1.5
  spec.stat.center = [0.2, 0, 0.8]

  # Add assets, cameras, obstacles, robot, etc to scene
  _add_assets(spec)
  _add_arena(spec, cfg)
  _add_place_marker(spec, cfg, layout)
  for i, ob in enumerate(layout.obstacles):
    add_obstacle(spec, cfg, i, ob)
  _add_object(spec, cfg, layout)
  _add_robot(spec, cfg)
  _add_static_camera(spec, cfg)

  return spec.compile()


# Add textures and materials that the geoms below refer to by name
def _add_assets(spec):
  spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                   builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[0.45, 0.60, 0.75], rgb2=[0.90, 0.93, 0.96],
                   width=512, height=3072)
  # Add the wood texture
  spec.add_texture(name="wood", type=mujoco.mjtTexture.mjTEXTURE_2D,
                   file=str(TEXTURES_DIR / "wood.png"))
  # Add the floor texture
  spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                   file=str(TEXTURES_DIR / "floor.png"))
  # Add the wood material
  m = spec.add_material(name="wood", texrepeat=[2, 2], reflectance=0.05)
  m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wood"
  # Add the floor material
  m = spec.add_material(name="floor", texrepeat=[8, 8], reflectance=0.02)
  m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
  # Add the steel material
  spec.add_material(name="steel", rgba=[0.45, 0.47, 0.50, 1], reflectance=0.4,
                    specular=0.6, shininess=0.5)


# Add the arena to the scene, the floor, table, pedestal
def _add_arena(spec, cfg):
  w = spec.worldbody
  w.add_light(pos=[0.3, 0, 2.2], dir=[0, 0, -1],
              type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
              diffuse=[0.55, 0.55, 0.55], specular=[0.2, 0.2, 0.2], castshadow=False)
  w.add_light(pos=[0.9, -0.6, 1.8], dir=[-0.4, 0.35, -1], diffuse=[0.45, 0.45, 0.45])
  w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1],
             material="floor")

  # Add the table top, box sizes are half extents so the top surface lands exactly at table.height
  cx, cy = cfg.table.center
  hx, hy = cfg.table.half_size
  top_z = cfg.table.height
  t = cfg.table.top_thickness
  w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
             pos=[cx, cy, top_z - t / 2], size=[hx, hy, t / 2],
             material="wood", friction=[1.0, 0.005, 0.0001])

  # Four legs
  leg_h = (top_z - t) / 2
  for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
    w.add_geom(name=f"table_leg_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), leg_h],
               size=[0.025, 0.025, leg_h], material="steel")

  # Add the pedestal, the base of the robot
  bx, by, bz = cfg.robot.base_pos
  w.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX,
             pos=[bx, by, bz / 2], size=[0.13, 0.13, bz / 2], material="steel")


# Add the place marker to the scene, the green disc that shows the place position, no collisions
def _add_place_marker(spec, cfg, layout):
  x, y = layout.place_pos
  spec.worldbody.add_geom(
    name="place_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=[x, y, cfg.table.height + 0.0015], size=[0.055, 0.0015, 0],
    rgba=[0.1, 0.8, 0.15, 0.9], contype=0, conaffinity=0)
  spec.worldbody.add_site(name="place_site", pos=[x, y, cfg.table.height],
                          size=[0.005, 0.005, 0.005], group=4)


# Adds the object by loading its mesh, it falls onto the table during settle
def _add_object(spec, cfg, layout):
  spec.add_mesh(name="object_mesh",
                file=str(OBJECTS_DIR / f"{layout.object_name}.stl"),
                scale=[layout.object_scale] * 3)
  # Drop the object 0.06m above the table height
  drop_z = cfg.table.height + 0.06
  body = spec.worldbody.add_body(name="object",
                                 pos=[layout.pick_pos[0], layout.pick_pos[1], drop_z])
  body.add_freejoint(name="object_joint")
  body.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="object_mesh", rgba=list(layout.object_color),
                mass=layout.object_mass,
                friction=[layout.object_friction, 0.01, 0.002],
                condim=4, solimp=[0.95, 0.99, 0.001, 0.5, 2], solref=[0.004, 1])


# Adds the robot to the scene, the panda arm and gripper, plus the wrist camera
def _add_robot(spec, cfg):
  panda = mujoco.MjSpec.from_file(str(PANDA_XML))
  # The menagerie gripper is soft, scale it up so heavy objects do not slip out
  grip = panda.actuator("actuator8")
  grip.gainprm[0] *= cfg.robot.grip_scale
  grip.biasprm[1:3] *= cfg.robot.grip_scale
  grip.forcerange *= cfg.robot.grip_scale
  # Add the grasp site to the hand, this is the point the controller moves
  hand = panda.body("hand")
  hand.add_site(name="grasp_site", pos=[0, 0, 0.1034], size=[0.005, 0.005, 0.005],
                group=4)

  # Camera body, tilted towards the fingertips so the gripper stays in the lower part of the frame
  d435i = mujoco.MjSpec.from_file(str(D435I_XML))
  tilt = _rot_y(np.deg2rad(-cfg.cameras.wrist.tilt_deg))
  body_rot = tilt @ _rot_z(np.pi / 2)
  # Add the mount to the hand
  mount = hand.add_frame(pos=[0.058, 0.0425, 0.02], quat=_mat_to_quat(body_rot))
  mount.attach_body(d435i.body("d435i"), "wristcam_", "")

  # Add the actual camera to the wrist, mujoco cameras look down their -z axis
  optical = tilt @ np.array([0, 0, 1.0])
  z_cam = -optical
  x_cam = np.array([0, 1.0, 0])
  y_cam = np.cross(z_cam, x_cam)
  cam_rot = np.column_stack([x_cam, y_cam, z_cam])
  hand.add_camera(name="wrist", pos=[0.058, 0, 0.034], fovy=cfg.cameras.wrist.fovy,
                  quat=_mat_to_quat(cam_rot))

  # Attach the panda arm to the world at the base position
  frame = spec.worldbody.add_frame(pos=cfg.robot.base_pos)
  frame.attach_body(panda.body("link0"), "", "")


# Adds the static camera to the scene, the camera that is looking at the table
def _add_static_camera(spec, cfg):
  cam = cfg.cameras.static
  pos = np.array(cam.pos, dtype=float)
  target = np.array(cam.lookat, dtype=float)
  z = pos - target
  z /= np.linalg.norm(z)
  x = np.cross([0, 0, 1], z)
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  spec.worldbody.add_camera(name="static", pos=pos, fovy=cam.fovy, xyaxes=[*x, *y])


# define rotation matrix for y axis
def _rot_y(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# define rotation matrix for z axis
def _rot_z(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# convert rotation matrix to quaternion
def _mat_to_quat(mat):
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).ravel())
  return quat
