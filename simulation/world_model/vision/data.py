"""Portable experiment files and strict data/cache contracts; no GPU imports."""
import hashlib
import json
import pathlib
import platform
import subprocess

import numpy as np


def digest(path):
  return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write_json(path, value):
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def provenance():
  git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
  changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
  return {"python": platform.python_version(), "platform": platform.platform(),
          "git_commit": git.stdout.strip() if git.returncode == 0 else None,
          "git_dirty": bool(changes.stdout.strip()) if changes.returncode == 0 else None}


def load_run(root):
  root = pathlib.Path(root)
  manifest = json.loads((root / "manifest.json").read_text())
  if manifest.get("schema") != "vision_consequences_v1":
    raise ValueError("wrong manifest schema; use world_model.vision.collect")
  return manifest


def load_features(root, manifest):
  root = pathlib.Path(root)
  meta = json.loads((root / "features/meta.json").read_text())
  if meta["manifest_sha256"] != digest(root / "manifest.json"):
    raise ValueError("feature cache belongs to a different manifest; encode this run again")
  keys = [s["key"] for s in manifest["states"]]
  if meta["keys"] != keys:
    raise ValueError("feature order does not match the manifest")
  z = np.load(root / "features/latents.npy", mmap_mode="r")
  if meta["latents_sha256"] != digest(root / "features/latents.npy"):
    raise ValueError("feature array changed after encoding; do not mix it with existing checkpoints")
  if z.ndim != 2 or len(z) != len(keys) or not np.isfinite(z).all():
    raise ValueError("invalid/nonfinite feature vectors")
  return np.asarray(z), meta


def state_arrays(manifest):
  states = manifest["states"]
  p = np.asarray([s["p"] for s in states], np.float32)
  xyz = np.asarray([s["object_xyz"] for s in states], np.float32)
  grasp = np.asarray([s["held"] for s in states], np.float32)
  return p, xyz, grasp


def rollout_ids(manifest, split):
  return [r for r in manifest["rollouts"] if r["split"] == split]


def clip_paths(root, state, clip_frames=64):
  history = state["frames"][-clip_frames:]
  if not history:
    raise ValueError(f"no frame history: {state['key']}")
  history = [history[0]] * (clip_frames - len(history)) + history
  return [pathlib.Path(root) / p for p in history]


def outcome_metrics(pred_xyz, probability, xyz, grasp):
  pred_xyz, probability, xyz, grasp = map(np.asarray, (pred_xyz, probability, xyz, grasp))
  guess = probability >= 0.5
  truth = grasp >= 0.5
  tp, fp, fn = int((guess & truth).sum()), int((guess & ~truth).sum()), int((~guess & truth).sum())
  precision = tp / max(tp + fp, 1)
  recall = tp / max(tp + fn, 1)
  return {"n": len(xyz), "xyz_mae_cm": (100 * np.abs(pred_xyz - xyz).mean(axis=0)).tolist(),
          "height_mae_cm": float(100 * np.abs(pred_xyz[:, 2] - xyz[:, 2]).mean()),
          "grasp_accuracy": float((guess == truth).mean()), "grasp_precision": precision,
          "grasp_recall": recall, "grasp_f1": 2 * precision * recall / max(precision + recall, 1e-12),
          "grasp_brier": float(np.square(probability - grasp).mean()),
          "height_mae_when_held_cm": float(100*np.abs(pred_xyz[truth, 2]-xyz[truth, 2]).mean()) if truth.any() else None,
          "height_mae_when_not_held_cm": float(100*np.abs(pred_xyz[~truth, 2]-xyz[~truth, 2]).mean()) if (~truth).any() else None,
          "positive_labels": int(truth.sum()), "negative_labels": int((~truth).sum())}
