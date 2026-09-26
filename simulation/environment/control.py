import mujoco
import numpy as np

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
GRIPPER_ACTUATOR = "actuator8"
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

# Action layout for a config, (dx, dy, dz, gripper) or (dx, dy, dz, dyaw, gripper) when yaw is on
def action_names(cfg):
  names = ["dx", "dy", "dz"]
  if cfg.control.yaw.enabled:
    names.append("dyaw")
  names.append("gripper")
  return names

# Moves the end effector by small deltas using damped least squares IK
# actions are in [-1, 1], position deltas scale by control.max_delta and get clipped to the workspace box
# orientation is always top down, rotated by the commanded yaw
class ArmController:

  def __init__(self, model, cfg):
    self.cfg = cfg.control
    self.dt = 1.0 / self.cfg.ik.hz
    self.yaw_enabled = self.cfg.yaw.enabled
    self.names = action_names(cfg)
    self.home_qpos = np.array(cfg.robot.home_qpos)
    # posture the null space pulls towards, env randomizes it every episode
    self.posture = self.home_qpos.copy()
    self.site_id = model.site("grasp_site").id
    self.joint_ids = [model.joint(n).id for n in ARM_JOINTS]
    self.dof_ids = [model.joint(n).dofadr[0] for n in ARM_JOINTS]
    self.qpos_ids = [model.joint(n).qposadr[0] for n in ARM_JOINTS]
    self.act_ids = [model.actuator(n).id for n in ARM_ACTUATORS]
    self.gripper_act_id = model.actuator(GRIPPER_ACTUATOR).id
    self.joint_range = model.jnt_range[self.joint_ids]
    self.ws_low = np.array(self.cfg.workspace.low)
    self.ws_high = np.array(self.cfg.workspace.high)
    self.model = model
    self.gripper_open = True

  @property
  def action_dim(self):
    return len(self.names)

  # Called after the arm has settled at home, takes the current pose as the starting target
  def reset(self, data):
    self.target_pos = data.site_xpos[self.site_id].copy()
    self.step_start = self.target_pos.copy()
    # Build a top down orientation that keeps the current finger direction
    mat = data.site_xmat[self.site_id].reshape(3, 3)
    z = np.array([0.0, 0.0, -1.0])
    x = mat[:, 0] - mat[:, 0].dot(z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).ravel())
    self.base_quat = quat
    self.yaw = 0.0
    self.q_des = data.qpos[self.qpos_ids].copy()
    self.gripper_open = True

  # Turn an action into a new position target, yaw and gripper command
  def set_action(self, action, data):
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    if len(action) != self.action_dim:
      raise ValueError(f"expected {self.action_dim}-dim action {self.names}, got {len(action)}")
    ee = data.site_xpos[self.site_id]
    self.step_start = ee.copy()
    self.target_pos = np.clip(ee + action[:3] * self.cfg.max_delta, self.ws_low, self.ws_high)
    if self.yaw_enabled:
      # a quarter turn each way covers every grasp (the gripper is symmetric) and keeps joint7 in range
      self.yaw = np.clip(self.yaw + action[3] * self.cfg.yaw.max_delta, -np.pi / 2, np.pi / 2)
    self.gripper_open = action[-1] > 0

  # Top down quaternion rotated by the current yaw
  def target_quat(self):
    if not self.yaw_enabled:
      return self.base_quat
    rot = np.array([np.cos(self.yaw / 2), 0, 0, np.sin(self.yaw / 2)])
    out = np.empty(4)
    mujoco.mju_mulQuat(out, rot, self.base_quat)
    return out

  # One IK update, progress in (0, 1] is how far into the control step we are
  # the position target slides from the start pose to the commanded one so the arm moves smoothly
  def update_ctrl(self, data, progress=1.0):
    ik = self.cfg.ik
    target = self.step_start + progress * (self.target_pos - self.step_start)
    pos_err = target - data.site_xpos[self.site_id]

    # Orientation error as a rotation vector
    site_quat = np.empty(4)
    mujoco.mju_mat2Quat(site_quat, data.site_xmat[self.site_id])
    quat_err = np.empty(4)
    conj = np.empty(4)
    mujoco.mju_negQuat(conj, site_quat)
    mujoco.mju_mulQuat(quat_err, self.target_quat(), conj)
    ori_err = np.empty(3)
    mujoco.mju_quat2Vel(ori_err, quat_err, 1.0)

    # Jacobian of the grasp site, only the arm dofs
    jacp = np.zeros((3, self.model.nv))
    jacr = np.zeros((3, self.model.nv))
    mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
    jac = np.vstack([jacp[:, self.dof_ids], jacr[:, self.dof_ids]])

    # Desired twist, capped
    v = ik.pos_gain * pos_err
    w = ik.ori_gain * ori_err
    v_norm = np.linalg.norm(v)
    if v_norm > ik.max_lin_vel:
      v *= ik.max_lin_vel / v_norm
    w_norm = np.linalg.norm(w)
    if w_norm > ik.max_ang_vel:
      w *= ik.max_ang_vel / w_norm
    twist = np.concatenate([v, w])

    # Damped least squares, plus a null space pull towards the posture
    reg = ik.damping ** 2 * np.eye(6)
    pinv = jac.T @ np.linalg.inv(jac @ jac.T + reg)
    qvel = pinv @ twist
    nullspace = np.eye(7) - pinv @ jac
    qvel += nullspace @ (ik.posture_gain * (self.posture - data.qpos[self.qpos_ids]))
    peak = np.abs(qvel).max()
    if peak > ik.max_joint_vel:
      qvel *= ik.max_joint_vel / peak

    # Integrate into the joint target, never let it run too far ahead of the actual joints
    q = data.qpos[self.qpos_ids]
    self.q_des = self.q_des + qvel * self.dt
    self.q_des = np.clip(self.q_des, q - ik.max_lead, q + ik.max_lead)
    self.q_des = np.clip(self.q_des, self.joint_range[:, 0], self.joint_range[:, 1])
    data.ctrl[self.act_ids] = self.q_des
    data.ctrl[self.gripper_act_id] = GRIPPER_OPEN if self.gripper_open else GRIPPER_CLOSED
