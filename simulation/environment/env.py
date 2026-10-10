import mujoco
import numpy as np

from .config import load_config
from .control import ArmController
from .rewards import Rewards
from . import scene


# Mujoco environment for the panda pick and place with obstacles
# Observation dict
#   proprio: qpos(7), qvel(7), ee_pos(3), ee_yaw(1), gripper width(1), gripper open(1)
#   goal: place position in the robot base frame
#   state: privileged sim state (object pose, ee pos, place pos)
class PickPlaceEnv:

  def __init__(self, cfg=None, seed=None):
    self.cfg = cfg if cfg is not None else load_config()
    self.rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
    # number of ik updates per control step
    self.ik_updates = self.cfg.control.ik.hz // self.cfg.control.hz
    # number of sim steps per ik update
    self.steps_per_ik = int(round(1.0 / (self.cfg.sim.timestep * self.cfg.control.ik.hz)))
    # record a frame every this many control steps
    self.frames_every = self.cfg.control.hz // self.cfg.cameras.record_hz
    self.model = None
    self.data = None
    self.layout = None
    self._renderer = None

  def reset(self, seed=None, layout=None, holdout=False):
    if seed is not None:
      self.rng = np.random.default_rng(seed)

    # Sample a layout for the episode, or replay the one passed in
    self.layout = layout or scene.sample_layout(self.cfg, self.rng, holdout=holdout)

    # A new model means a new renderer
    self.close()
    self.model = scene.build_scene(self.cfg, self.layout)
    self.data = mujoco.MjData(self.model)
    self._index_model()

    # Controller, with a slightly different posture target every episode so the elbow varies
    self.controller = ArmController(self.model, self.cfg)
    noise = self.cfg.robot.posture_noise
    posture = np.array(self.cfg.robot.home_qpos) + self.rng.uniform(-noise, noise, 7)
    self.controller.posture = np.clip(posture, self.controller.joint_range[:, 0],
                                      self.controller.joint_range[:, 1])

    self.reward = Rewards(self.cfg)
    self.step_count = 0
    self._success_steps = 0
    self._settle()
    self.controller.reset(self.data)
    self.goal = self._goal_vector()
    return self._observation(), {"layout": self.layout.summary()}

  # One control step, runs the ik updates and sim steps in between and computes the reward
  # give_up=True is the policy saying it has nothing left to try, the episode ends as a failure
  def step(self, action, give_up=False):
    self.controller.set_action(action, self.data)
    grasped_any = False
    obstacle_hit = False
    table_hit = False
    for i in range(self.ik_updates):
      self.controller.update_ctrl(self.data, (i + 1) / self.ik_updates)
      mujoco.mj_step(self.model, self.data, nstep=self.steps_per_ik)
      # Check contacts after every ik update so a quick brush is not missed
      g, o, t = self._check_contacts()
      grasped_any |= g
      obstacle_hit |= o
      table_hit |= t
    self.step_count += 1

    state = self._privileged_state(grasped_any, obstacle_hit, table_hit)
    state["obstacle_distance"] = self.obstacle_distance()

    # Failed when the object fell off the table, the policy gave up, or time ran out without success
    fell = state["object_pos"][2] < self.cfg.table.height - self.cfg.episode.fall_margin
    timeout = self.step_count >= self.cfg.episode.max_steps
    state["failed"] = (fell or give_up or timeout) and not self.reward.stage == "done"
    total, components, success = self.reward.compute(state, np.asarray(action))
    if success:
      self._success_steps += 1

    # Done once the object has stayed at the target for settle_steps, or right away on a failure
    placed = (self.cfg.episode.terminate_on_success
              and self._success_steps >= self.cfg.episode.settle_steps)
    terminated = placed or bool(state["failed"])
    truncated = timeout and not terminated
    info = {
      "reward_components": components,
      "stage": self.reward.stage,
      "grasped": grasped_any,
      "obstacle_contact": obstacle_hit,
      "table_contact": table_hit,
      "success": success,
      "failed": bool(state["failed"]),
      "object_pos": state["object_pos"].copy(),
    }
    return self._observation(), total, terminated, truncated, info

  @property
  def action_dim(self):
    return self.controller.action_dim

  @property
  def action_names(self):
    return self.controller.names

  # Camera pose in the robot base frame
  def camera_pose(self, camera):
    cam_id = self.model.cam(camera).id
    base_id = self.model.body("link0").id
    base_rot = self.data.xmat[base_id].reshape(3, 3)
    pos = base_rot.T @ (self.data.cam_xpos[cam_id] - self.data.xpos[base_id])
    rot = base_rot.T @ self.data.cam_xmat[cam_id].reshape(3, 3)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    return pos, quat

  def camera_poses(self):
    return {name: self.camera_pose(name) for name in ("static", "wrist")}

  # Yaw of the finger axis, for grasping
  def finger_axis_yaw(self):
    y = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 1]
    return float(np.arctan2(y[1], y[0]))

  # Yaw of the object's long axis (seen from above), so the fingers can line up with it
  def object_long_axis_yaw(self):
    g = self.model.geom("object_geom").id
    half = self.model.geom_aabb[g][3:]
    rot = self.data.geom_xmat[g].reshape(3, 3)
    extents = np.linalg.norm(rot[:2, :], axis=0) * half
    axis = rot[:, int(np.argmax(extents))]
    return float(np.arctan2(axis[1], axis[0]))

  # Highest point of the object right now, from its bounding box
  def object_top_z(self):
    g = self.model.geom("object_geom").id
    half = self.model.geom_aabb[g][3:]
    rot = self.data.geom_xmat[g].reshape(3, 3)
    reach = np.abs(rot[2, :]) @ half
    return float(self.data.geom_xpos[g][2] + reach)

  # Render the scene from a camera
  def render(self, camera="static"):
    r = self._get_renderer()
    r.update_scene(self.data, camera=camera)
    return r.render()

  # Render the depth image
  def render_depth(self, camera="static"):
    r = self._get_renderer()
    r.enable_depth_rendering()
    try:
      r.update_scene(self.data, camera=camera)
      return r.render()
    finally:
      r.disable_depth_rendering()

  # Render all cameras
  def render_all(self):
    return {name: self.render(name) for name in ("static", "wrist")}

  # Point cloud from the depth image, in the camera, world or robot base frame
  def point_cloud(self, camera="static", frame="base", stride=None, max_depth=None):
    pc_cfg = self.cfg.cameras.pointcloud
    stride = pc_cfg.stride if stride is None else stride
    max_depth = pc_cfg.max_depth if max_depth is None else max_depth

    # Stride to reduce resolution
    depth = self.render_depth(camera)[::stride, ::stride]
    h, w = depth.shape
    full_h = self.cfg.cameras.height
    fovy = np.deg2rad(self.model.cam(camera).fovy[0])
    f = 0.5 * full_h / np.tan(fovy / 2) / stride
    cx, cy = (w - 1) / 2, (h - 1) / 2
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = depth < max_depth
    d = depth[valid]

    # Unproject the pixels, mujoco cameras look down -z
    pts_cam = np.stack([(u[valid] - cx) / f * d, (cy - v[valid]) / f * d, -d], axis=1)
    if frame == "camera":
      return pts_cam.astype(np.float32)
    cam_id = self.model.cam(camera).id
    rot = self.data.cam_xmat[cam_id].reshape(3, 3)
    pts = pts_cam @ rot.T + self.data.cam_xpos[cam_id]
    if frame == "world":
      return pts.astype(np.float32)
    if frame == "base":
      base_id = self.model.body("link0").id
      base_rot = self.data.xmat[base_id].reshape(3, 3)
      base_pos = self.data.xpos[base_id]
      return ((pts - base_pos) @ base_rot).astype(np.float32)
    raise ValueError(f"unknown frame: {frame}")

  def point_clouds(self, frame="base"):
    return {name: self.point_cloud(name, frame=frame) for name in ("static", "wrist")}

  # Make the renderer only when needed
  def _get_renderer(self):
    if self._renderer is None:
      cams = self.cfg.cameras
      self._renderer = mujoco.Renderer(self.model, cams.height, cams.width)
    return self._renderer

  def close(self):
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None

  # Look up all the ids we need once per model
  def _index_model(self):
    m = self.model
    self.object_body_id = m.body("object").id
    self.object_geom_ids = {m.geom("object_geom").id}
    self.obstacle_geom_ids = {g for g in range(m.ngeom)
                              if (m.geom(g).name or "").startswith("obstacle_geom")}
    self.left_finger_geoms = {g for g in range(m.ngeom)
                              if m.geom_bodyid[g] == m.body("left_finger").id}
    self.right_finger_geoms = {g for g in range(m.ngeom)
                               if m.geom_bodyid[g] == m.body("right_finger").id}
    self.table_geom_id = m.geom("table").id

    # Every geom hanging off link0 belongs to the robot
    robot_root = m.body("link0").id
    self.robot_geom_ids = set()
    for g in range(m.ngeom):
      b = m.geom_bodyid[g]
      while b != 0 and b != robot_root:
        b = m.body_parentid[b]
      if b == robot_root:
        self.robot_geom_ids.add(g)

    # collision geoms of the arm, for the distance to the obstacles
    self.robot_collision_geoms = [g for g in sorted(self.robot_geom_ids)
                                  if m.geom_contype[g] or m.geom_conaffinity[g]]
    self.site_id = m.site("grasp_site").id
    self.finger_qpos_ids = [m.joint("finger_joint1").qposadr[0],
                            m.joint("finger_joint2").qposadr[0]]
    self.arm_qpos_ids = [m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
    self.arm_dof_ids = [m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)]
    self.object_qpos_adr = m.joint("object_joint").qposadr[0]

  # Put the arm at home, give the object a random yaw and let everything settle for half a second
  def _settle(self):
    home = self.cfg.robot.home_qpos
    self.data.qpos[self.arm_qpos_ids] = home
    self.data.qpos[self.finger_qpos_ids] = 0.04
    yaw = self.rng.uniform(-np.pi, np.pi)
    self.data.qpos[self.object_qpos_adr + 3:self.object_qpos_adr + 7] = [
      np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    self.data.ctrl[:7] = home
    self.data.ctrl[7] = 255
    mujoco.mj_forward(self.model, self.data)
    mujoco.mj_step(self.model, self.data, nstep=int(0.5 / self.model.opt.timestep))
    self.object_rest_z = float(self.data.xpos[self.object_body_id][2])

  # Check for contacts between the robot, object, obstacles and table
  def _check_contacts(self):
    left = right = False
    obstacle = False
    table = False
    finger_geoms = self.left_finger_geoms | self.right_finger_geoms
    for i in range(self.data.ncon):
      pair = {self.data.contact[i].geom1, self.data.contact[i].geom2}
      # Object touching each finger
      if pair & self.object_geom_ids:
        if pair & self.left_finger_geoms:
          left = True
        if pair & self.right_finger_geoms:
          right = True
      # Robot or the carried object touching an obstacle
      if pair & self.obstacle_geom_ids:
        if pair & (self.robot_geom_ids | self.object_geom_ids):
          obstacle = True
      # Arm (not fingers) touching the table
      if self.table_geom_id in pair:
        if pair & (self.robot_geom_ids - finger_geoms):
          table = True

    # Grasped when both fingers touch the object while the gripper is closed
    grasped = left and right and not self.controller.gripper_open
    return grasped, obstacle, table

  # Closest any part of the arm gets to any obstacle, capped at rewards.safe_distance
  def obstacle_distance(self):
    cap = self.cfg.rewards.safe_distance
    best = cap
    fromto = np.zeros(6)
    for g in self.robot_collision_geoms:
      for o in self.obstacle_geom_ids:
        best = min(best, mujoco.mj_geomDistance(self.model, self.data, g, o, cap, fromto))
    return float(max(best, 0.0))

  # Privileged state right now, used by the scripted policy
  def privileged_state(self):
    grasped, obstacle, table = self._check_contacts()
    return self._privileged_state(grasped, obstacle, table)

  def _privileged_state(self, grasped, obstacle_hit, table_hit):
    return {
      "object_pos": self.data.xpos[self.object_body_id].copy(),
      "object_quat": self.data.xquat[self.object_body_id].copy(),
      "object_rest_z": self.object_rest_z,
      "ee_pos": self.data.site_xpos[self.site_id].copy(),
      "place_pos": np.array([*self.layout.place_pos, self.cfg.table.height]),
      "grasped": grasped,
      "obstacle_contacts": 1.0 if obstacle_hit else 0.0,
      "table_contacts": 1.0 if table_hit else 0.0,
    }

  # Goal vector is the place position relative to the robot base
  def _goal_vector(self):
    place = np.array([*self.layout.place_pos, self.cfg.table.height])
    return place - np.array(self.cfg.robot.base_pos)

  # Observation contains the proprioceptive state of robot, eef pose, gripper state, goal, and object state
  def _observation(self):
    d = self.data
    gripper_width = float(d.qpos[self.finger_qpos_ids].sum())
    proprio = np.concatenate([
      d.qpos[self.arm_qpos_ids],
      d.qvel[self.arm_dof_ids],
      d.site_xpos[self.site_id],
      [self.finger_axis_yaw(), gripper_width, 1.0 if self.controller.gripper_open else -1.0],
    ]).astype(np.float32)
    state = np.concatenate([
      d.xpos[self.object_body_id],
      d.xquat[self.object_body_id],
      d.site_xpos[self.site_id],
      [*self.layout.place_pos, self.cfg.table.height],
    ]).astype(np.float32)
    return {"proprio": proprio, "goal": self.goal.astype(np.float32), "state": state}
