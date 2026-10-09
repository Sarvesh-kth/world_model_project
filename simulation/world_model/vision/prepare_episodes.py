"""Test 1: turn data_collection.collect episodes (success / scripted / drop / collide / random, with their
recorded reward components) into the vision manifest format, so encode / train / eval_reward
can train and test Q, the penalty head and D on the full data mix.
   usage: python -m world_model.vision.prepare_episodes --episodes data/episodes_test1/episodes --run data/episodes_test1
Each episode becomes one rollout (states = recorded rows, actions = the action stored on the next row);
episodes are split 70/15/15 by index. Only the static camera frames are referenced.
"""
import os

# offscreen rendering through EGL; set before mujoco is imported
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import csv
import json
import pathlib

import numpy as np

from environment.rewards import COMPONENTS
from world_model.prepare import P_COLUMNS, A_COLUMNS
from .data import provenance, write_json

CAMPAIGN = "episodes_test1"


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--episodes", type=pathlib.Path, required=True)
  p.add_argument("--run", type=pathlib.Path, required=True)
  p.add_argument("--val-fraction", type=float, default=.15)
  p.add_argument("--test-fraction", type=float, default=.15)
  args = p.parse_args()
  folders = sorted(args.episodes.glob("episode_*/data.csv"))
  if not folders:
    p.error("no episodes found")
  n = len(folders)
  n_test, n_val = round(args.test_fraction * n), round(args.val_fraction * n)
  cfg = json.loads((folders[0].parent / "meta.json").read_text())["config"]
  manifest = {"schema": "vision_consequences_v1", "campaign": CAMPAIGN, "complete": True,
              "settings": {"pair_tolerance": 1e-5, "collection": {k: str(v) for k, v in vars(args).items()}},
              "config": cfg, "camera": "static", "control_hz": cfg["control"]["hz"],
              "p_columns": P_COLUMNS, "action_columns": A_COLUMNS,
              "states": [], "rollouts": [], "scene_settings": [], "provenance": provenance()}
  kinds = {}
  for index, path in enumerate(folders):
    folder = path.parent
    meta = json.loads((folder / "meta.json").read_text())
    split = "test" if index >= n - n_test else "val" if index >= n - n_test - n_val else "train"
    with path.open(newline="") as f:
      rows = list(csv.DictReader(f))
    scene = folder.name
    kind = meta["policy"]
    kinds[kind] = kinds.get(kind, 0) + 1
    start = len(manifest["states"])
    history, actions = [], []
    for j, row in enumerate(rows):
      serial = int(row["serial"])
      frame = f"episodes/{scene}/images/static_{serial}.jpg"
      if not (args.run / frame).is_file() and not (folder / "images" / f"static_{serial}.jpg").is_file():
        raise FileNotFoundError(frame)
      history.append(frame)
      action = [float(row[c]) for c in A_COLUMNS]
      if j:
        actions.append(action)
      components = {k: float(row[f"reward_{k}"]) for k in COMPONENTS}
      manifest["states"].append({
          "key": f"{scene}:{serial:04d}", "scene": scene, "split": split, "frames": history[-64:],
          "p": [float(row[c]) for c in P_COLUMNS],
          "object_xyz": [float(row[c]) for c in ("object_x", "object_y", "object_z")],
          "object_quat": [float(row[c]) for c in ("object_qw", "object_qx", "object_qy", "object_qz")],
          "held": bool(int(row["grasped"])), "grasped_during_step": bool(int(row["grasped"])),
          "obstacle_contact": bool(int(row["obstacle_contact"])), "table_contact": bool(int(row["table_contact"])),
          "obstacle_contact_during_step": bool(int(row["obstacle_contact"])),
          "table_contact_during_step": bool(int(row["table_contact"])),
          "reward_stage": row["stage"], "reward": float(row["reward_total"]), "reward_components": components,
          "action_from_previous": action if j else None, "time": float(row["time"]),
          "view": "obstacles" if meta["layout"]["obstacles"] else "empty", "case": kind,
          "source_controller": kind, "task_success": bool(int(row["success"]))})
    manifest["rollouts"].append({"id": scene, "scene": scene, "split": split, "placement": "episode",
        "branch": kind, "view": "obstacles" if meta["layout"]["obstacles"] else "empty", "case": kind,
        "states": list(range(start, start + len(rows))), "actions": actions,
        "restore_p_error": 0.0, "restore_integration_error": 0.0,
        "result": {"task_success": any(int(r["success"]) for r in rows), "steps": len(rows),
                   "obstacle_contact_steps": sum(int(r["obstacle_contact"]) for r in rows)}})
    manifest["scene_settings"].append({"rollout": scene, "seed": meta["seed"], "layout": meta["layout"], "policy": kind})
  # the vision audit expects the run root to hold `episodes/<scene>/...` frames
  link = args.run / "episodes"
  args.run.mkdir(parents=True, exist_ok=True)
  if not link.exists():
    link.symlink_to(args.episodes.resolve())
  write_json(args.run / "manifest.json", manifest)
  write_json(args.run / "scene_settings.json", manifest["scene_settings"])
  penalised = sum(1 for s in manifest["states"] if any(s["reward_components"][k] for k in ("collision", "proximity", "table_hit")))
  print(f"{n} episodes -> {len(manifest['states'])} states; kinds {kinds}; "
        f"penalised states {penalised}; held {sum(s['held'] for s in manifest['states'])}")


if __name__ == "__main__":
  main()
