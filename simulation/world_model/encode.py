"""Cache mean-pooled frozen V-JEPA 2 features for selected camera clips."""

import argparse
import hashlib
import json
import pathlib
import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

MODEL = "facebook/vjepa2-vitl-fpc64-256"


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--manifest", required=True, type=pathlib.Path)
  p.add_argument("--out", required=True, type=pathlib.Path)
  p.add_argument("--model", default=MODEL)
  args = p.parse_args()
  if not torch.cuda.is_available():
    p.error("CUDA is unavailable in this Python environment; install a compatible PyTorch CUDA build")
  manifest = json.loads(args.manifest.read_text())
  manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
  episodes = (args.manifest.resolve().parent / manifest["episodes_dir"]).resolve()
  endpoints = {}
  for sample in manifest["samples"]:
    source_key = f"{sample['episode']}:{sample['source']}"
    target_key = sample.get("target_key", f"{sample['episode']}:{sample['target']}")
    endpoints[source_key] = (sample["episode"], sample["source"], None)
    target = (sample["episode"], sample["target"], sample.get("target_image"))
    if target_key in endpoints and endpoints[target_key] != target:
      p.error(f"conflicting clip endpoint {target_key}")
    endpoints[target_key] = target
  keys = sorted(endpoints)
  if not endpoints:
    p.error("manifest has no clip endpoints")
  processor = AutoVideoProcessor.from_pretrained(args.model)
  model = AutoModel.from_pretrained(args.model, attn_implementation="sdpa")
  model.to("cuda", dtype=torch.bfloat16).eval()
  if manifest["clip_frames"] % model.config.tubelet_size:
    p.error("clip length must be divisible by the V-JEPA tubelet size")
  vectors = []
  index = {}
  with torch.inference_mode():
    for n, key in enumerate(keys, 1):
      episode, serial, target_image = endpoints[key]
      frames = []
      for i in range(serial - manifest["clip_frames"] + 1, serial + 1):
        path = ((args.manifest.resolve().parent / target_image)
                if target_image and i == serial else
                episodes / episode / "images" / f"{manifest['camera']}_{max(1, i)}.jpg")
        frame = cv2.imread(str(path))
        if frame is None:
          raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
      inputs = processor(video, return_tensors="pt").to("cuda")
      with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = model(**inputs, skip_predictor=True).last_hidden_state
      z = tokens.float().mean(dim=1).squeeze(0).cpu().numpy()
      index[key] = len(vectors)
      vectors.append(z)
      if n % 10 == 0 or n == len(keys):
        print(f"encoded {n}/{len(keys)} clips", flush=True)
  args.out.mkdir(parents=True, exist_ok=True)
  np.save(args.out / "latents.npy", np.stack(vectors).astype(np.float32))
  (args.out / "meta.json").write_text(json.dumps({
    "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_sha256,
    "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
    "camera": manifest["camera"], "clip_frames": manifest["clip_frames"],
    "pooling": "mean_all_encoder_tokens", "index": index,
  }, indent=2) + "\n")
  print(f"saved {len(vectors)} vectors of size {len(vectors[0])} to {args.out}")


if __name__ == "__main__":
  main()
