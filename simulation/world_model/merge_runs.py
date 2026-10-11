import argparse
import pathlib
import shutil

import numpy as np

from .common import load_run, load_features, write_json

# Put several encoded runs into one run folder so Q, R and D train on all of them. The runs must have been
# encoded with the same PCA basis (encode.py --pca) and clip settings. State and scene keys are prefixed
# with the run name, the frame folders are linked, the latents are concatenated.
# --overwrite redoes the merge in place (after a run grew) and keeps the trained attempts/ of the folder.
#   python -m world_model.merge_runs --runs data/full_v2 data/obstacles_v2 data/episodes_v2 --out data/combined_v2


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--runs", nargs="+", type=pathlib.Path, required=True)
  p.add_argument("--out", type=pathlib.Path, required=True)
  p.add_argument("--overwrite", action="store_true", help="redo an existing merge, keeping its attempts/")
  args = p.parse_args()
  if args.out.exists():
    if not args.overwrite:
      p.error(f"{args.out} exists; add --overwrite to merge again")
    for entry in args.out.iterdir():
      if entry.name == "attempts" or entry.name == "logs":
        continue
      if entry.is_symlink() or entry.is_file():
        entry.unlink()
      else:
        shutil.rmtree(entry)

  merged, latents, first_meta, first_pca = None, [], None, None
  for run in args.runs:
    manifest = load_run(run)
    z, meta = load_features(run, manifest)
    pca = np.load(run / "features/pca.npz")
    if first_meta is None:
      first_meta, first_pca = meta, pca
    elif (any(meta.get(k) != first_meta.get(k) for k in ("pooling", "cameras", "clip_frames", "frame_stride", "dtype", "heightmap", "layout"))
          or sorted(pca.files) != sorted(first_pca.files)
          or not all(np.allclose(pca[k], first_pca[k]) for k in pca.files)):
      p.error(f"{run} was encoded with another basis or clip setting; encode it again with --pca")

    # prefix everything with the run name so keys stay unique, shift state indices by what is merged so far
    name = run.name
    offset = 0 if merged is None else len(merged["states"])
    for s in manifest["states"]:
      s["key"] = f"{name}/{s['key']}"
      s["scene"] = f"{name}/{s['scene']}"
      s["frames"] = [f"{name}/{f}" for f in s["frames"]]
      s["wrist_frames"] = [f"{name}/{f}" for f in s.get("wrist_frames", [])]
    for r in manifest["rollouts"]:
      r["id"] = f"{name}/{r['id']}"
      r["scene"] = f"{name}/{r['scene']}"
      r["states"] = [i + offset for i in r["states"]]
    if merged is None:
      merged = {k: v for k, v in manifest.items() if k not in ("states", "rollouts", "scene_settings")}
      merged.update(campaign="combined", complete=True, states=[], rollouts=[], scene_settings=[],
                    settings={"runs": [str(r) for r in args.runs]})
    merged["states"] += manifest["states"]
    merged["rollouts"] += manifest["rollouts"]
    merged["scene_settings"] += manifest.get("scene_settings", [])
    latents.append(np.asarray(z, np.float32))
    print(f"{name}: {len(manifest['states'])} states, {len(manifest['rollouts'])} rollouts", flush=True)

  args.out.mkdir(parents=True, exist_ok=True)
  for run in args.runs:
    (args.out / run.name).symlink_to(run.resolve())
  write_json(args.out / "manifest.json", merged)
  write_json(args.out / "scene_settings.json", merged["scene_settings"])
  features = args.out / "features"
  features.mkdir()
  shutil.copyfile(args.runs[0] / "features/pca.npz", features / "pca.npz")
  np.save(features / "latents.npy", np.concatenate(latents))
  write_json(features / "meta.json", {**{k: v for k, v in first_meta.items() if k not in ("keys", "seconds")},
                                      "keys": [s["key"] for s in merged["states"]],
                                      "merged_from": [str(r) for r in args.runs]})
  print(f"merged {len(merged['states'])} states into {args.out}")


if __name__ == "__main__":
  main()
