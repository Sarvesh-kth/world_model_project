"""Encode every real branch clip with the frozen base V-JEPA encoder (CUDA)."""
import argparse
import json
import pathlib
import time

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

from .audit import audit
from .data import load_run, digest, write_json, clip_paths, provenance


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--run", required=True, type=pathlib.Path)
  p.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
  p.add_argument("--resume", action="store_true", help="resume an interrupted cache for this exact run/model")
  args = p.parse_args()
  if not torch.cuda.is_available():
    p.error("CUDA unavailable: use the notebook CUDA virtualenv, not the Mac")
  manifest = load_run(args.run)
  issues = audit(args.run, manifest, images=False)["problems"]
  if issues:
    p.error("dataset audit failed: " + "; ".join(issues[:5]))
  out = args.run / "features"
  signature = {"manifest_sha256": digest(args.run / "manifest.json"), "model": args.model}
  done, vectors, revision = 0, None, None
  partial = out / "latents.partial.npy"
  if out.exists():
    if not args.resume or (out / "meta.json").exists():
      p.error("features exists; use --resume for an interrupted encoding (completed caches are preserved)")
    progress = json.loads((out / "progress.json").read_text())
    if any(progress[k] != v for k, v in signature.items()):
      p.error("interrupted cache belongs to a different run/model")
    done = progress["encoded"]
    revision = progress.get("model_revision")
    if not 0 <= done <= len(manifest["states"]):
      p.error("invalid progress count")
    if done:
      if not partial.exists() and (out / "latents.npy").exists():
        partial = out / "latents.npy"
      vectors = np.load(partial, mmap_mode="r+")
      if len(vectors) != len(manifest["states"]) or not np.isfinite(vectors[:done]).all():
        p.error("invalid interrupted cache")
    print(f"resuming after {done} encoded clips", flush=True)
  else:
    out.mkdir()
    write_json(out / "progress.json", {**signature, "encoded": 0})
  processor = AutoVideoProcessor.from_pretrained(args.model, revision=revision)
  model = AutoModel.from_pretrained(args.model, revision=revision, attn_implementation="sdpa")
  model.to("cuda", dtype=torch.bfloat16).eval()
  signature["model_revision"] = getattr(model.config, "_commit_hash", None)
  write_json(out / "progress.json", {**signature, "encoded": done})
  keys = [s["key"] for s in manifest["states"]]
  start = time.monotonic()
  with torch.inference_mode():
    for i, state in enumerate(manifest["states"], 1):
      if i <= done:
        continue
      frames = []
      for path in clip_paths(args.run, state):
        bgr = cv2.imread(str(path))
        if bgr is None:
          raise FileNotFoundError(f"{state['key']}: {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
      video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
      inputs = processor(video, return_tensors="pt").to("cuda")
      with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = model(**inputs, skip_predictor=True).last_hidden_state
      z = tokens.float().mean(dim=1).squeeze(0).cpu().numpy()
      if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError(f"invalid V-JEPA vector at {state['key']}")
      if vectors is None:
        vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                            shape=(len(keys), len(z)))
      vectors[i-1] = z
      if i % 20 == 0 or i == len(manifest["states"]):
        vectors.flush()
        write_json(out / "progress.json", {**signature, "encoded": i})
        print(f"encoded {i}/{len(manifest['states'])} clips; "
              f"{(time.monotonic()-start)/max(i-done,1):.2f}s/clip; peak CUDA "
              f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
  vectors.flush()
  del vectors
  if partial != out / "latents.npy":
    partial.replace(out / "latents.npy")
  write_json(out / "meta.json", {"manifest_sha256": digest(args.run / "manifest.json"),
             "latents_sha256": digest(out / "latents.npy"),
             "keys": keys, "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
             "pooling": "mean_all_encoder_tokens", "camera": manifest["camera"], "clip_frames": 64,
             "provenance": {**provenance(), "torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name()}})
  (out / "progress.json").unlink()
  shape = np.load(out / "latents.npy", mmap_mode="r").shape
  print(f"saved {shape[0]} frozen vectors of size {shape[1]}; no predictor/fine-tuning")


if __name__ == "__main__":
  main()
