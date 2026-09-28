"""Replay a saved source state and execute a one-step action branch."""

import csv
import json

import cv2
import numpy as np

from environment import EpisodeLayout, PickPlaceEnv, load_config
from .prepare import A_COLUMNS


def read_episode(episodes, sample):
  folder = episodes / sample["episode"]
  with (folder / "data.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
  meta = json.loads((folder / "meta.json").read_text())
  if sample["source"] >= len(rows):
    raise ValueError(f"source serial {sample['source']} has no next row in {folder}")
  return folder, rows, meta


def replay_branch(sample, rows, meta, action, camera, replay_mode=None):
  # Replaying the recorded prefix also restores the controller's targets and reward state.
  cfg = load_config(overrides=meta["config"])
  last_error = None
  for mode in ([replay_mode] if replay_mode else ["fixed", "sampled"]):
    env = PickPlaceEnv(cfg)
    try:
      layout = EpisodeLayout.from_dict(meta["layout"]) if mode == "fixed" else None
      env.reset(seed=int(meta["seed"]), layout=layout)
      if mode == "sampled" and env.layout.to_dict() != meta["layout"]:
        last_error = "sampled layout differs from the recorded layout"
        continue
      ended = False
      for row in rows[:sample["source"]]:
        recorded_action = [float(row[name]) for name in A_COLUMNS]
        obs, _, terminated, truncated, _ = env.step(recorded_action)
        if terminated or truncated:
          ended = True
          break
      if ended:
        last_error = f"{mode} replay ended before the selected source state"
        continue
      p_error = float(np.max(np.abs(obs["proprio"] - sample["p"])))
      object_pose = [float(rows[sample["source"] - 1][f"object_{axis}"])
                     for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")]
      object_error = float(np.max(np.abs(obs["state"][:7] - object_pose)))
      if p_error > 1e-3 or object_error > 1e-3:
        last_error = (f"{mode} replay differs from recorded state: "
                      f"p={p_error:.6g}, object pose={object_error:.6g}")
        continue
      source_state = np.concatenate((env.data.qpos, env.data.qvel, env.data.act,
                                     env.data.ctrl, env.controller.target_pos,
                                     env.controller.q_des,
                                     [env.controller.yaw, float(env.controller.gripper_open)]))
      next_obs, _, _, _, _ = env.step(action)
      rgb = env.render(camera)
      ok, jpeg = cv2.imencode(".jpg", rgb[..., ::-1],
                              [cv2.IMWRITE_JPEG_QUALITY, cfg.data.jpeg_quality])
      if not ok:
        raise RuntimeError("could not JPEG-encode branch frame")
      return (next_obs["proprio"].copy(), jpeg.tobytes(), p_error,
              object_error, mode, source_state)
    finally:
      env.close()
  raise ValueError(f"could not replay the recorded source state: {last_error}")
