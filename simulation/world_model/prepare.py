"""Make one-step transitions from the collector's post-action observations."""

import argparse
import csv
import json
import os
import pathlib
import numpy as np

P_COLUMNS = ([f"joint_pos_{i}" for i in range(1, 8)]
             + [f"joint_vel_{i}" for i in range(1, 8)]
             + ["ee_x", "ee_y", "ee_z", "ee_yaw", "gripper_width", "gripper_cmd"])
A_COLUMNS = ["action_dx", "action_dy", "action_dz", "action_dyaw", "action_gripper"]


def vector(row, columns):
  return [float(row[name]) for name in columns]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--episodes", required=True, type=pathlib.Path)
  p.add_argument("--out", required=True, type=pathlib.Path)
  p.add_argument("--camera", choices=("static", "wrist"), default="static")
  p.add_argument("--clip-frames", type=int, default=64)
  p.add_argument("--stride", type=int, default=8, help="spacing between selected source steps")
  p.add_argument("--max-per-episode", type=int, default=0, help="0 means no cap")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  if args.clip_frames < 2 or args.stride < 1 or args.max_per_episode < 0:
    p.error("clip-frames >= 2, stride >= 1, max-per-episode >= 0 required")
  episodes = sorted(args.episodes.resolve().glob("episode_*/data.csv"))
  if len(episodes) < 2:
    p.error("at least two collected episodes are needed for an episode-level validation split")
  rng = np.random.default_rng(args.seed)
  val_count = max(1, round(0.2 * len(episodes)))
  val_names = {episodes[i].parent.name for i in rng.permutation(len(episodes))[:val_count]}
  samples = []
  for path in episodes:
    with path.open(newline="") as f:
      rows = list(csv.DictReader(f))
    if any(int(row["serial"]) != i for i, row in enumerate(rows, 1)):
      raise ValueError(f"nonconsecutive serials: {path}")
    if any(not (path.parent / "images" / f"{args.camera}_{i}.jpg").is_file()
           for i in range(1, len(rows) + 1)):
      raise FileNotFoundError(f"missing {args.camera} frames in {path.parent}")
    serials = list(range(1, len(rows), args.stride))
    if args.max_per_episode and len(serials) > args.max_per_episode:
      serials = [serials[i] for i in np.linspace(
        0, len(serials) - 1, args.max_per_episode, dtype=int)]
    for serial in serials:
      source, target = rows[serial - 1], rows[serial]
      samples.append({"episode": path.parent.name, "source": serial,
                      "target": serial + 1, "split": "val" if path.parent.name in val_names else "train",
                      "p": vector(source, P_COLUMNS), "action": vector(target, A_COLUMNS),
                      "next_p": vector(target, P_COLUMNS),
                      "object_xyz_label": vector(source, ["object_x", "object_y", "object_z"])})
  counts = {split: sum(s["split"] == split for s in samples) for split in ("train", "val")}
  if min(counts.values()) == 0:
    p.error(f"need train and validation transitions; got {counts}")
  args.out.parent.mkdir(parents=True, exist_ok=True)
  manifest = {"version": 1, "episodes_dir": os.path.relpath(
    args.episodes.resolve(), args.out.parent.resolve()),
    "camera": args.camera, "clip_frames": args.clip_frames,
    "clip_padding": "repeat_first_frame", "stride": args.stride,
    "proprio_columns": P_COLUMNS, "action_columns": A_COLUMNS, "samples": samples}
  args.out.write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"saved {len(samples)} transitions ({counts['train']} train, {counts['val']} val) to {args.out}")
  print("Each pair is post-action observation serial t, action stored on serial t+1, and observation serial t+1.")
  first = samples[0]
  print(f"example: {first['episode']} serial {first['source']} -> {first['target']}, "
        f"action={first['action']}, split={first['split']}")


if __name__ == "__main__":
  main()
