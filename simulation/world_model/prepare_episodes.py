import argparse
import csv
import json
import pathlib

from environment.rewards import COMPONENTS
from .common import P_COLUMNS, A_COLUMNS, new_manifest, write_json

# Turn a data_collection.collect dataset (success / scripted / drop / collide / random episodes with their
# reward components) into a run folder with a manifest, so encode / train can use the full data mix.
# Every episode becomes one rollout, split 70 / 15 / 15 by episode index; only the static camera is used.
#   python -m data_collection.collect --config configs/full_mix.yml --episodes 120 --out data/episodes_test1/episodes_raw
#   python -m world_model.prepare_episodes --episodes data/episodes_test1/episodes_raw --run data/episodes_test1


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--episodes", type=pathlib.Path, required=True, help="folder with episode_*/ from data_collection.collect")
  p.add_argument("--run", type=pathlib.Path, required=True, help="run folder to create")
  p.add_argument("--val-fraction", type=float, default=.15)
  p.add_argument("--test-fraction", type=float, default=.15)
  args = p.parse_args()
  folders = sorted(args.episodes.glob("episode_*/data.csv"))
  if not folders:
    p.error("no episodes found")
  n = len(folders)
  n_test, n_val = round(args.test_fraction * n), round(args.val_fraction * n)
  cfg = json.loads((folders[0].parent / "meta.json").read_text())["config"]
  manifest = new_manifest("episodes", cfg, {k: str(v) for k, v in vars(args).items()})

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
    view = "obstacles" if meta["layout"]["obstacles"] else "empty"
    start = len(manifest["states"])
    history, actions = [], []

    # one state per recorded row, the action stored on a row is the one that led to it
    for j, row in enumerate(rows):
      serial = int(row["serial"])
      frame = f"episodes/{scene}/images/static_{serial}.jpg"
      if not (folder / "images" / f"static_{serial}.jpg").is_file():
        raise FileNotFoundError(frame)
      history.append(frame)
      action = [float(row[c]) for c in A_COLUMNS]
      if j:
        actions.append(action)
      manifest["states"].append({
        "key": f"{scene}:{serial:04d}", "scene": scene, "split": split, "frames": history[-64:],
        "p": [float(row[c]) for c in P_COLUMNS],
        "object_xyz": [float(row[c]) for c in ("object_x", "object_y", "object_z")],
        "object_quat": [float(row[c]) for c in ("object_qw", "object_qx", "object_qy", "object_qz")],
        "held": bool(int(row["grasped"])),
        "obstacle_contact": bool(int(row["obstacle_contact"])), "table_contact": bool(int(row["table_contact"])),
        "reward_stage": row["stage"], "reward": float(row["reward_total"]),
        "reward_components": {k: float(row[f"reward_{k}"]) for k in COMPONENTS},
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
