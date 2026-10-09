"""Test 1: put several encoded runs into one run directory so Q, the penalty head and D train on all of them.
The runs must share the same PCA basis (encode --pca) and clip settings; the merged run keeps each
run's splits, prefixes scene and state keys with the run name and links the frame folders.
   usage: python -m world_model.vision.merge_runs --runs data/full_test1 data/obstacles_test1_fullpca data/episodes_test1_fullpca --out data/combined_test1
"""
import os

# offscreen rendering through EGL; set before mujoco is imported
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import pathlib
import shutil

import numpy as np

from .data import load_run, load_features, digest, write_json, provenance

CAMPAIGN = "combined_test1"


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--runs", nargs="+", type=pathlib.Path, required=True)
  p.add_argument("--out", type=pathlib.Path, required=True)
  args = p.parse_args()
  if args.out.exists():
    p.error(f"{args.out} exists")
  merged, latents, metas = None, [], []
  for run in args.runs:
    manifest = load_run(run)
    z, meta = load_features(run, manifest)
    metas.append(meta)
    if meta["pca_sha256"] != metas[0]["pca_sha256"] or any(meta[k] != metas[0][k] for k in ("pooling", "clip_frames", "frame_stride", "dtype")):
      p.error(f"{run} was encoded with another basis or clip setting; re-encode it with --pca")
    name = run.name
    offset = 0 if merged is None else len(merged["states"])
    for s in manifest["states"]:
      s["key"] = f"{name}/{s['key']}"
      s["scene"] = f"{name}/{s['scene']}"
      s["frames"] = [f"{name}/{f}" for f in s["frames"]]
      s.setdefault("source_run", name)
    for r in manifest["rollouts"]:
      r["id"] = f"{name}/{r['id']}"
      r["scene"] = f"{name}/{r['scene']}"
      r["states"] = [i+offset for i in r["states"]]
    if merged is None:
      merged = {k: v for k, v in manifest.items() if k not in ("states", "rollouts", "scene_settings")}
      merged.update(campaign=CAMPAIGN, complete=True, states=[], rollouts=[], scene_settings=[],
                    settings={"runs": [str(r) for r in args.runs]}, provenance=provenance())
    merged["states"] += manifest["states"]
    merged["rollouts"] += manifest["rollouts"]
    merged["scene_settings"] += manifest.get("scene_settings", [])
    latents.append(np.asarray(z, np.float32))
    print(f"{name}: {len(manifest['states'])} states, {len(manifest['rollouts'])} rollouts", flush=True)
  args.out.mkdir(parents=True)
  for run in args.runs:
    (args.out / run.name).symlink_to(run.resolve())
  write_json(args.out / "manifest.json", merged)
  write_json(args.out / "scene_settings.json", merged["scene_settings"])
  features = args.out / "features"
  features.mkdir()
  shutil.copyfile(args.runs[0] / "features/pca.npz", features / "pca.npz")
  np.save(features / "latents.npy", np.concatenate(latents))
  first = metas[0]
  write_json(features / "meta.json", {**{k: first[k] for k in ("model", "model_revision", "pooling", "grid", "camera",
                                                              "clip_frames", "frame_stride", "dtype", "spatial_pool")},
             "manifest_sha256": digest(args.out / "manifest.json"), "latents_sha256": digest(features / "latents.npy"),
             "keys": [s["key"] for s in merged["states"]], "pca_sha256": digest(features / "pca.npz"),
             "merged_from": {str(r): m["latents_sha256"] for r, m in zip(args.runs, metas)}, "provenance": provenance()})
  print(f"merged {len(merged['states'])} states into {args.out}")


if __name__ == "__main__":
  main()
