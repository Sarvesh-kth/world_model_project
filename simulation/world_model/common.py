import csv
import json
import pathlib

import numpy as np

# Small helpers that every stage shares: the run folder layout, json / csv writing, the manifest and the
# feature cache, and the metrics Q is judged by. Nothing in here touches the GPU.

SIMULATION = pathlib.Path(__file__).resolve().parents[1]

# The 20 robot numbers (p) and the 5 action numbers, in the order the collector writes them
P_COLUMNS = ([f"joint_pos_{i}" for i in range(1, 8)] + [f"joint_vel_{i}" for i in range(1, 8)]
             + ["ee_x", "ee_y", "ee_z", "ee_yaw", "gripper_width", "gripper_cmd"])
A_COLUMNS = ["action_dx", "action_dy", "action_dz", "action_dyaw", "action_gripper"]


# Write json through a temporary file, so an interrupted run never leaves a half written file behind
def write_json(path, value):
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


# Rows are dicts that may have different keys, the header is the union of them
def save_csv(path, rows):
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
    writer.writeheader()
    writer.writerows(rows)


# A run folder holds manifest.json (every recorded state and rollout), observations/ or episodes/ (the frames),
# features/ (one latent per state, written by encode.py) and attempts/<tag>/models/ (trained Q, R, D)
def load_run(root):
  manifest = json.loads((pathlib.Path(root) / "manifest.json").read_text())
  if manifest.get("schema") != "vision_consequences_v1":
    raise ValueError(f"{root} is not a run folder (no manifest with schema vision_consequences_v1)")
  return manifest


# The latent of every state, in manifest order, plus the encoder settings they were made with
def load_features(root, manifest):
  root = pathlib.Path(root)
  meta = json.loads((root / "features/meta.json").read_text())
  keys = [s["key"] for s in manifest["states"]]
  if meta["keys"] != keys:
    raise ValueError("features/meta.json lists other states than manifest.json; encode this run again")
  z = np.load(root / "features/latents.npy")
  if z.shape[0] != len(keys) or not np.isfinite(z).all():
    raise ValueError("features/latents.npy does not match the manifest or has nonfinite values")
  return z, meta


# Robot state p, cube position and held label of every state as arrays
def state_arrays(manifest):
  states = manifest["states"]
  p = np.asarray([s["p"] for s in states], np.float32)
  xyz = np.asarray([s["object_xyz"] for s in states], np.float32)
  held = np.asarray([s["held"] for s in states], np.float32)
  return p, xyz, held


# Sanity checks on a manifest before encoding or training on it, returns a list of problems (empty = fine)
def check_manifest(manifest):
  problems = []
  states, rollouts = manifest["states"], manifest["rollouts"]
  if not manifest.get("complete"):
    problems.append("collection did not finish")
  if len({s["key"] for s in states}) != len(states):
    problems.append("duplicate state keys")

  # a scene must sit in one split only, otherwise validation leaks into training
  splits = {}
  for s in states:
    splits.setdefault(s["scene"], set()).add(s["split"])
  if any(len(v) != 1 for v in splits.values()):
    problems.append("a scene appears in several splits")

  # rollouts are state[0] -a[0]-> state[1] -a[1]-> ... so there is one more state than actions
  for r in rollouts:
    if len(r["states"]) != len(r["actions"]) + 1:
      problems.append(f"rollout {r['id']} has {len(r['states'])} states for {len(r['actions'])} actions")
      continue
    for j, action in enumerate(r["actions"], 1):
      if states[r["states"][j]]["action_from_previous"] != action:
        problems.append(f"rollout {r['id']} step {j}: action does not match the next state")
        break

  for split in ("train", "val", "test"):
    labels = [s["held"] for s in states if s["split"] == split]
    if split == "train" and not labels:
      problems.append("no training states")
    if labels and (not any(labels) or all(labels)):
      problems.append(f"{split} split needs both held and not-held states")
  return problems


# How well predicted cube positions and held probabilities match the truth, in cm and as a classifier
def outcome_metrics(pred_xyz, probability, xyz, held):
  pred_xyz, probability, xyz, held = map(np.asarray, (pred_xyz, probability, xyz, held))
  guess = probability >= 0.5
  truth = held >= 0.5
  tp = int((guess & truth).sum())
  fp = int((guess & ~truth).sum())
  fn = int((~guess & truth).sum())
  precision = tp / max(tp + fp, 1)
  recall = tp / max(tp + fn, 1)
  return {"n": len(xyz),
          "xyz_mae_cm": (100 * np.abs(pred_xyz - xyz).mean(axis=0)).tolist(),
          "height_mae_cm": float(100 * np.abs(pred_xyz[:, 2] - xyz[:, 2]).mean()),
          "grasp_accuracy": float((guess == truth).mean()),
          "grasp_precision": precision, "grasp_recall": recall,
          "grasp_f1": 2 * precision * recall / max(precision + recall, 1e-12),
          "positive_labels": int(truth.sum()), "negative_labels": int((~truth).sum())}


# One recorded state for the manifest: robot state, cube pose, contact flags, reward and the frame history
# that is the camera clip ending at this state (newest frame last)
def record(env, key, scene, split, history, action=None, info=None, reward=0):
  obs = env._observation()
  grasp, obstacle, table = env._check_contacts()
  info = info or {}
  return {"key": key, "scene": scene, "split": split, "frames": history[-64:],
          "p": obs["proprio"].tolist(), "object_xyz": obs["state"][:3].tolist(),
          "object_quat": obs["state"][3:7].tolist(), "held": bool(grasp),
          "obstacle_contact": bool(obstacle), "table_contact": bool(table),
          "reward_stage": info.get("stage", env.reward.stage), "reward": float(reward),
          "reward_components": info.get("reward_components", {}),
          "action_from_previous": action, "time": float(env.data.time)}


# An empty manifest, every collector starts from this
def new_manifest(campaign, cfg, settings):
  return {"schema": "vision_consequences_v1", "campaign": campaign, "complete": False,
          "settings": settings, "config": cfg, "camera": "static", "control_hz": cfg["control"]["hz"],
          "p_columns": P_COLUMNS, "action_columns": A_COLUMNS,
          "states": [], "rollouts": [], "scene_settings": []}
