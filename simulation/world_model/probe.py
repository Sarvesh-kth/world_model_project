"""Diagnostic only: measure whether frozen visual features carry object position."""

import argparse
import hashlib
import json
import pathlib
import numpy as np


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--manifest", required=True, type=pathlib.Path)
  p.add_argument("--features", required=True, type=pathlib.Path)
  p.add_argument("--ridge", type=float, default=1.0)
  args = p.parse_args()
  manifest = json.loads(args.manifest.read_text())
  meta = json.loads((args.features / "meta.json").read_text())
  if meta["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
    p.error("feature cache does not match this manifest")
  z_all = np.load(args.features / "latents.npy")
  samples = manifest["samples"]
  z = np.stack([z_all[meta["index"][f"{s['episode']}:{s['source']}"]] for s in samples])
  y = np.asarray([s["object_xyz_label"] for s in samples], dtype=np.float32)
  train = np.asarray([i for i, s in enumerate(samples) if s["split"] == "train"])
  val = np.asarray([i for i, s in enumerate(samples) if s["split"] == "val"])
  mu, sigma = z[train].mean(axis=0), np.maximum(z[train].std(axis=0), 1e-3)
  x_train = (z[train] - mu) / sigma
  x_val = (z[val] - mu) / sigma
  y_mean = y[train].mean(axis=0)
  # Dual ridge form is cheap when there are fewer transitions than feature dimensions.
  weights = x_train.T @ np.linalg.solve(
    x_train @ x_train.T + args.ridge * np.eye(len(train)), y[train] - y_mean)
  predicted = x_val @ weights + y_mean
  probe_mae = np.abs(predicted - y[val]).mean(axis=0)
  constant_mae = np.abs(y_mean - y[val]).mean(axis=0)
  print(f"held-out object xyz MAE (metres): probe={probe_mae.round(4).tolist()} "
        f"constant baseline={constant_mae.round(4).tolist()}")
  print("Simulator object coordinates are used as diagnostic labels only, not as D inputs.")


if __name__ == "__main__":
  main()
