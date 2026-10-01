"""Encode an M2 manifest's clips with frozen V-JEPA 2, on any device.

Writes the same files as M2's world_model/encode.py (latents.npy + meta.json, including the
manifest checksum M2's train_dynamics.py verifies), so M2's training script runs on the result
unchanged. The difference: M2's script requires CUDA, this one also runs on Apple MPS or CPU.

    .venv/bin/python -m controller.experiments.encode_features --manifest M.json --out DIR
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from controller.adapters.m2_adapter import JEPAEncoder


def clip_endpoints(manifest):
    """{key: (episode, serial, target_image or None)} for every clip D needs, as M2 builds them."""
    endpoints = {}
    for s in manifest["samples"]:
        endpoints[f"{s['episode']}:{s['source']}"] = (s["episode"], s["source"], None)
        target_key = s.get("target_key", f"{s['episode']}:{s['target']}")
        endpoints[target_key] = (s["episode"], s["target"], s.get("target_image"))
    return endpoints


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    episodes = (root / manifest["episodes_dir"]).resolve()
    endpoints = clip_endpoints(manifest)
    keys = sorted(endpoints)
    encoder = JEPAEncoder(device=args.device)
    print(f"{len(keys)} clips on {encoder.device} ({encoder.dtype})", flush=True)

    cache = {}  # decoded frames of the current episode; neighbouring clips share 63 of 64 frames

    def frame(path):
        if path not in cache:
            if len(cache) > 400:
                cache.clear()
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(path)
            cache[path] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cache[path]

    vectors, index, t0 = [], {}, time.time()
    for n, key in enumerate(keys, 1):
        episode, serial, target_image = endpoints[key]
        frames = []
        for i in range(serial - manifest["clip_frames"] + 1, serial + 1):
            if target_image and i == serial:
                frames.append(frame(root / target_image))
            else:
                frames.append(frame(episodes / episode / "images" / f"{manifest['camera']}_{max(1, i)}.jpg"))
        index[key] = len(vectors)
        vectors.append(encoder.encode_clip(np.stack(frames)))
        if n % 50 == 0 or n == len(keys):
            print(f"encoded {n}/{len(keys)} clips ({(time.time() - t0) / n:.2f} s/clip)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "latents.npy", np.stack(vectors).astype(np.float32))
    (args.out / "meta.json").write_text(json.dumps({
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "model": encoder.model_name, "model_revision": getattr(encoder.model.config, "_commit_hash", None),
        "camera": manifest["camera"], "clip_frames": manifest["clip_frames"],
        "pooling": "mean_all_encoder_tokens", "index": index,
        "device": str(encoder.device), "dtype": str(encoder.dtype),
    }, indent=2) + "\n")
    print(f"saved {len(vectors)} vectors to {args.out}")


if __name__ == "__main__":
    main()
