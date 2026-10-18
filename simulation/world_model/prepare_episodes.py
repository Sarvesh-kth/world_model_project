import argparse
import csv
import json
import pathlib

import numpy as np

from environment.rewards import COMPONENTS
from .common import P_COLUMNS, A_COLUMNS, new_manifest, write_json

# Turn a collected dataset into a run folder with a manifest for encode / train. Every episode is one rollout
# with both cameras and the height map. The split is by episode index modulo 7 (5 val, 6 test), so collecting
# more into the same folder and running this again keeps the old splits and only appends (encode --resume).
#   python -m world_model.prepare_episodes --episodes data/episodes/episodes_raw --run data/episodes


# Episodes recorded before the empty_grasp penalty existed get it from the row, any other missing component is 0
def empty_grasp(row, k):
  if k != "empty_grasp":
    return 0.0
  return -float(float(row["gripper_cmd"]) < 0 and float(row["gripper_width"]) < 0.03)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--episodes", type=pathlib.Path, required=True, help="folder with episode_*/ from data_collection.collect")
  p.add_argument("--run", type=pathlib.Path, required=True, help="run folder to create")
  args = p.parse_args()
  folders = sorted(args.episodes.glob("episode_*/data.csv"))
  if not folders:
    p.error("no episodes found")
  n = len(folders)
  # the newest episode's config, so a reward weight added later is in the manifest the models are trained with
  cfg = json.loads((folders[-1].parent / "meta.json").read_text())["config"]
  manifest = new_manifest("episodes", cfg, {k: str(v) for k, v in vars(args).items()})

  kinds = {}
  for index, path in enumerate(folders):
    folder = path.parent
    meta = json.loads((folder / "meta.json").read_text())
    split = {5: "val", 6: "test"}.get(index % 7, "train")
    with path.open(newline="") as f:
      rows = list(csv.DictReader(f))
    scene = folder.name
    kind = meta["policy"]
    kinds[kind] = kinds.get(kind, 0) + 1
    view = "obstacles" if meta["layout"]["obstacles"] else "empty"
    start = len(manifest["states"])
    history, wrist_history, actions = [], [], []
    heightmaps = np.load(folder / "heightmaps.npy")
    if len(heightmaps) != len(rows):
      raise ValueError(f"{folder}: {len(heightmaps)} height maps for {len(rows)} rows")

    # one state per recorded row, the action stored on a row is the one that led to it
    for j, row in enumerate(rows):
      serial = int(row["serial"])
      for camera, kept in (("static", history), ("wrist", wrist_history)):
        frame = f"episodes/{scene}/images/{camera}_{serial}.jpg"
        if not (folder / "images" / f"{camera}_{serial}.jpg").is_file():
          raise FileNotFoundError(frame)
        kept.append(frame)
      action = [float(row[c]) for c in A_COLUMNS]
      if j:
        actions.append(action)
      manifest["states"].append({
        "key": f"{scene}:{serial:04d}", "scene": scene, "split": split, "frames": history[-64:],
        "wrist_frames": wrist_history[-64:], "heightmap": heightmaps[j].reshape(-1).tolist(),
        "p": [float(row[c]) for c in P_COLUMNS],
        "object_xyz": [float(row[c]) for c in ("object_x", "object_y", "object_z")],
        "object_quat": [float(row[c]) for c in ("object_qw", "object_qx", "object_qy", "object_qz")],
        "held": bool(int(row["grasped"])),
        "obstacle_contact": bool(int(row["obstacle_contact"])), "table_contact": bool(int(row["table_contact"])),
        "reward_stage": row["stage"], "reward": float(row["reward_total"]),
        "reward_components": {k: float(row[f"reward_{k}"]) if f"reward_{k}" in row else empty_grasp(row, k) for k in COMPONENTS},
        "action_from_previous": action if j else None, "time": float(row["time"]),
        "view": view, "case": kind, "task_success": bool(int(row["success"]))})
    manifest["rollouts"].append({
      "id": scene, "scene": scene, "split": split, "view": view, "case": kind,
      "states": list(range(start, start + len(rows))), "actions": actions,
      "result": {"task_success": any(int(r["success"]) for r in rows), "steps": len(rows),
                 "obstacle_contact_steps": sum(int(r["obstacle_contact"]) for r in rows)}})
    manifest["scene_settings"].append({"rollout": scene, "seed": meta["seed"], "layout": meta["layout"], "policy": kind})

  # the frames stay where the collector wrote them, the run folder links to them
  manifest["complete"] = True
  args.run.mkdir(parents=True, exist_ok=True)
  link = args.run / "episodes"
  if not link.exists():
    link.symlink_to(args.episodes.resolve())
  write_json(args.run / "manifest.json", manifest)
  write_json(args.run / "scene_settings.json", manifest["scene_settings"])
  penalised = sum(1 for s in manifest["states"] if any(s["reward_components"][k] for k in ("collision", "proximity", "table_hit")))
  print(f"{n} episodes -> {len(manifest['states'])} states; kinds {kinds}; penalised states {penalised}; "
        f"held {sum(s['held'] for s in manifest['states'])}")


if __name__ == "__main__":
  main()
