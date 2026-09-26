import csv
import json
import pathlib
import cv2
import numpy as np
from environment.control import action_names
from environment.rewards import COMPONENTS

INDEX_FIELDS = ["episode", "folder", "policy", "object", "pick_end", "place_end",
                "steps", "success", "collisions", "seed"]
CAMERAS = ("static", "wrist")
POSE_AXES = ("x", "y", "z", "qw", "qx", "qy", "qz")

# Buffers one episode and writes it out as one folder
# episode_XXXXXX/
#   data.csv     one row per recorded step, keyed by serial
#   meta.json    layout (replayable), policy variant, seed, goal, config
#   images/      static_<serial>.jpg, wrist_<serial>.jpg
#   pcl/         static_<serial>.npy, wrist_<serial>.npy when point clouds are on
class EpisodeWriter:

  def __init__(self, out_dir, cfg, write_index=True):
    self.dir = pathlib.Path(out_dir)
    self.dir.mkdir(parents=True, exist_ok=True)
    self.cfg = cfg
    self.write_index = write_index
    self.action_names = action_names(cfg)

  def start_episode(self, layout, policy, seed, goal, meta=None):
    self.layout = layout
    self.policy = policy
    self.seed = seed
    self.goal = np.asarray(goal, dtype=float)
    self.meta = meta or {}
    self.rows = []
    self.images = {cam: [] for cam in CAMERAS}
    self.pointclouds = {cam: [] for cam in CAMERAS}
    self.collisions = 0
    self.success = False

  # Store one step, images are jpeg encoded right away to keep memory down
  def add_record(self, step, action, obs, reward, info, cam_poses, images, pointclouds=None):
    serial = len(self.rows) + 1
    self.rows.append(self._row(serial, step, action, obs, reward, info, cam_poses))
    quality = [cv2.IMWRITE_JPEG_QUALITY, self.cfg.data.jpeg_quality]
    for cam in CAMERAS:
      # opencv wants BGR
      ok, jpeg = cv2.imencode(".jpg", images[cam][..., ::-1], quality)
      if not ok:
        raise RuntimeError("jpeg encoding failed")
      self.images[cam].append(jpeg)
      if pointclouds:
        self.pointclouds[cam].append(pointclouds[cam].astype(np.float16))
    self.collisions += int(info["obstacle_contact"])
    self.success |= bool(info["success"])

  # Write everything to disk and return the folder and the index row
  def finish_episode(self, episode_id):
    folder = f"episode_{episode_id:06d}"
    ep_dir = self.dir / folder
    img_dir = ep_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(ep_dir / "data.csv", "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
      writer.writeheader()
      writer.writerows(self.rows)
    for cam in CAMERAS:
      for i, jpeg in enumerate(self.images[cam]):
        (img_dir / f"{cam}_{i + 1}.jpg").write_bytes(jpeg.tobytes())
      if self.pointclouds[cam]:
        pcl_dir = ep_dir / "pcl"
        pcl_dir.mkdir(exist_ok=True)
        for i, pc in enumerate(self.pointclouds[cam]):
          np.save(pcl_dir / f"{cam}_{i + 1}.npy", pc)
    meta = {
      "episode": episode_id,
      "policy": self.policy,
      "policy_meta": self.meta,
      "seed": self.seed,
      "goal": list(self.goal),
      "layout": self.layout.to_dict(),
      "steps": len(self.rows),
      "success": self.success,
      "collisions": self.collisions,
      "config": self.cfg,
    }
    with open(ep_dir / "meta.json", "w") as f:
      json.dump(meta, f, indent=1, default=str)
    row = {
      "episode": episode_id, "folder": folder, "policy": self.policy,
      "object": self.layout.object_name,
      "pick_end": self.layout.pick_end, "place_end": self.layout.place_end,
      "steps": len(self.rows),
      "success": self.success, "collisions": self.collisions,
      "seed": self.seed,
    }
    if self.write_index:
      append_index(self.dir, [row])
    return ep_dir, row

  # One csv row, action, joint state, object pose, rewards, flags and camera poses
  def _row(self, serial, step, action, obs, reward, info, cam_poses):
    row = {"serial": serial, "time": round(step / self.cfg.control.hz, 4)}
    row.update({f"action_{n}": v for n, v in zip(self.action_names, action)})
    p = obs["proprio"]
    row.update({f"joint_pos_{i + 1}": p[i] for i in range(7)})
    row.update({f"joint_vel_{i + 1}": p[7 + i] for i in range(7)})
    row.update(ee_x=p[14], ee_y=p[15], ee_z=p[16], ee_yaw=p[17],
               gripper_width=p[18], gripper_cmd=p[19])
    s = obs["state"]
    row.update({f"object_{a}": v for a, v in zip(POSE_AXES, s[:7])})
    row.update(place_x=s[10], place_y=s[11], place_z=s[12])
    row.update(goal_x=self.goal[0], goal_y=self.goal[1], goal_z=self.goal[2])
    row["reward_total"] = reward
    row.update({f"reward_{k}": info["reward_components"][k] for k in COMPONENTS})
    row.update(grasped=int(info["grasped"]),
               obstacle_contact=int(info["obstacle_contact"]),
               table_contact=int(info["table_contact"]),
               success=int(info["success"]), stage=info["stage"])
    for cam, (pos, quat) in cam_poses.items():
      row.update({f"{cam}_cam_{a}": round(float(v), 6)
                  for a, v in zip(POSE_AXES, [*pos, *quat])})
    return row

# Append rows to index.csv, writing the header if the file is new
def append_index(out_dir, rows):
  path = pathlib.Path(out_dir) / "index.csv"
  new = not path.exists()
  with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if new:
      w.writerow(INDEX_FIELDS)
    for row in rows:
      w.writerow([row[k] for k in INDEX_FIELDS])

# Next free episode id in a folder, so re-running appends instead of overwriting
def next_episode_id(out_dir):
  folders = sorted(pathlib.Path(out_dir).glob("episode_*"))
  if not folders:
    return 0
  return max(int(p.name.split("_")[1]) for p in folders) + 1

# Read one episode folder back, rows as dicts of floats, images and point clouds as sorted paths
def read_episode(ep_dir):
  ep_dir = pathlib.Path(ep_dir)
  with open(ep_dir / "data.csv") as f:
    rows = []
    for raw in csv.DictReader(f):
      rows.append({k: v if k == "stage" else float(v) for k, v in raw.items()})
  with open(ep_dir / "meta.json") as f:
    meta = json.load(f)
  out = {"rows": rows, "meta": meta, "images": {}, "pointclouds": {}}
  by_serial = lambda p: int(p.stem.split("_")[1])
  for cam in CAMERAS:
    out["images"][cam] = sorted((ep_dir / "images").glob(f"{cam}_*.jpg"), key=by_serial)
    if (ep_dir / "pcl").exists():
      out["pointclouds"][cam] = sorted((ep_dir / "pcl").glob(f"{cam}_*.npy"), key=by_serial)
  return out

def load_image(path):
  return cv2.imread(str(path))[..., ::-1]

def load_pointcloud(path):
  return np.load(path).astype(np.float32)
