"""Add action-diverse one-step branches from training states to a manifest."""

import argparse
import hashlib
import json
import os
import pathlib

import numpy as np

from .replay import read_episode, replay_branch


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=pathlib.Path,
                      help="original prepared manifest; only its train states are branched")
  parser.add_argument("--out", required=True, type=pathlib.Path,
                      help="new manifest path; branch JPEGs go beside it")
  parser.add_argument("--axes", nargs="+", choices=("x", "y", "z"),
                      default=("x", "y", "z"))
  parser.add_argument("--max-sources", type=int, default=0,
                      help="0 uses every original training source")
  args = parser.parse_args()
  if args.max_sources < 0:
    parser.error("--max-sources must be nonnegative")
  if args.manifest.resolve() == args.out.resolve():
    parser.error("--out must be a new path so the original manifest stays unchanged")
  manifest = json.loads(args.manifest.read_text())
  if len(manifest["action_columns"]) != 5:
    parser.error("branch collection expects the five-action yaw-enabled controller")
  train = [s for s in manifest["samples"] if s["split"] == "train"]
  val = [s for s in manifest["samples"] if s["split"] == "val"]
  if not train or not val:
    parser.error("original manifest needs train and validation samples")
  if any("target_image" in s for s in manifest["samples"]):
    parser.error("use the original prepared manifest, not an already branched one")
  selected = train[:args.max_sources] if args.max_sources else train
  episodes = (args.manifest.resolve().parent / manifest["episodes_dir"]).resolve()
  images = args.out.parent / f"{args.out.stem}_branch_frames"
  images.mkdir(parents=True, exist_ok=True)
  augmented = []
  for n, sample in enumerate(selected, 1):
    _, rows, meta = read_episode(episodes, sample)
    p = np.asarray(sample["p"], dtype=np.float32)
    gripper = float(p[-1])
    mode = None
    source_state = None
    for axis in dict.fromkeys(args.axes):
      for sign, label in ((1.0, "plus"), (-1.0, "minus")):
        action = [0.0, 0.0, 0.0, 0.0, gripper]
        action["xyz".index(axis)] = sign
        next_p, jpeg, _, _, mode, replay_state = replay_branch(
          sample, rows, meta, action, manifest["camera"], mode)
        if source_state is None:
          source_state = replay_state
        elif not np.allclose(source_state, replay_state, rtol=0, atol=1e-8):
          raise ValueError(f"branches started from different states in {sample['episode']}")
        key = f"branch:{sample['episode']}:{sample['source']}:{label}_{axis}"
        path = images / f"{sample['episode']}_{sample['source']:04d}_{label}_{axis}.jpg"
        path.write_bytes(jpeg)
        branch = dict(sample, action=action, next_p=next_p.astype(float).tolist(),
                      target_key=key,
                      target_image=os.path.relpath(path.resolve(), args.out.parent.resolve()),
                      branch_action=f"{label}_{axis}")
        augmented.append(branch)
    print(f"[{n}/{len(selected)}] {sample['episode']} serial {sample['source']} "
          f"-> {2 * len(dict.fromkeys(args.axes))} branches", flush=True)
  out = dict(manifest)
  out["episodes_dir"] = os.path.relpath(episodes, args.out.parent.resolve())
  out["samples"] = manifest["samples"] + augmented
  out["branch_collection"] = {
    "source_manifest": str(args.manifest.resolve()),
    "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    "axes": list(dict.fromkeys(args.axes)),
    "source_count": len(selected), "branch_count": len(augmented),
    "validation_episodes_unchanged": True,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(out, indent=2) + "\n")
  print(f"saved {len(manifest['samples'])} original + {len(augmented)} train branches "
        f"({len(val)} original validation) to {args.out}")


if __name__ == "__main__":
  main()
