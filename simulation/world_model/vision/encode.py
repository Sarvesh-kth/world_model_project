"""Encode every state's camera clip with the frozen V-JEPA encoder (CUDA).

--pooling picks what becomes the latent z:
  mean_all      mean over all 32x16x16 tokens -> 1024 values (the original; object information is lost)
  spatial_last  the 16x16 patch grid of the last time slice, PCA each patch to --pca-dims values
  spatial_mean  the same grid with the tokens averaged over the time slices first (default; 8x more stable)
--spatial-pool N averages NxN neighbouring patches before the PCA (2 -> 8x8 grid). The PCA basis is fitted label
free on a few hundred training clips, saved next to the features and reused online by control_pipeline; --pca
reuses another run's basis so several runs can be trained on together (merge_runs). One vector per state, so
train / test / control read the cache the same way whichever pooling was used."""
import os

# offscreen rendering through EGL; set before mujoco is imported
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import functools
import json
import pathlib
import shutil
import time

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

from .audit import audit
from .data import load_run, digest, write_json, clip_paths, provenance

# pinned revision of the team's feature cache, so old and new features come from identical weights
PINNED_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
GRID = 16


def select_history(history, clip_frames, stride):
  # newest frame last; walk back `stride` frames at a time; repeat the oldest frame when history is short
  picked = [history[max(len(history)-1-i*stride, 0)] for i in range(clip_frames)]
  return picked[::-1]


@functools.lru_cache(maxsize=2048)
def read_rgb(path):
  # consecutive states share most of their clip, so decoded frames are reused
  bgr = cv2.imread(path)
  if bgr is None:
    raise FileNotFoundError(path)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_frames(root, state, clip_frames=64, stride=1):
  frames = [read_rgb(str(pathlib.Path(root) / name)) for name in select_history(state["frames"], clip_frames, stride)]
  return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


class Encoder:
  """Frozen V-JEPA plus the chosen pooling. control_pipeline uses the same class online."""

  DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

  def __init__(self, model, revision, pooling, pca=None, device="cuda", clip_frames=64, stride=1, dtype="fp16",
               spatial_pool=1):
    self.clip_frames, self.stride, self.dtype, self.spatial_pool = clip_frames, stride, dtype, spatial_pool
    self.processor = AutoVideoProcessor.from_pretrained(model, revision=revision)
    self.model = AutoModel.from_pretrained(model, revision=revision, attn_implementation="sdpa")
    # fp16 deviates ~3% of a token norm from fp32, bf16 ~18%, at the same speed
    self.model.to(device, dtype=self.DTYPES[dtype]).eval()
    self.revision = getattr(self.model.config, "_commit_hash", None) or revision
    self.pooling, self.device = pooling, device
    self.pca = None
    if pca is not None:
      self.pca = {"mean": torch.as_tensor(pca["mean"], device=device),
                  "components": torch.as_tensor(pca["components"], device=device)}

  def preprocess(self, video):
    # same as the HF video processor (shortest edge 292, centre crop 256, rescale, normalise) but on the GPU;
    # the CPU processor was the bottleneck (0.32 s/clip vs 0.15 s/clip of encoder time)
    x = video.to(self.device).float() / 255
    x = torch.nn.functional.interpolate(x, size=(292, 292), mode="bilinear", align_corners=False, antialias=True)
    x = x[:, :, 18:274, 18:274]
    mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
    std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)

  @torch.inference_mode()
  def tokens(self, video):
    with torch.autocast(self.device, dtype=self.DTYPES[self.dtype], enabled=self.dtype != "fp32"):
      out = self.model(pixel_values_videos=self.preprocess(video), skip_predictor=True).last_hidden_state
    return out.float().squeeze(0)                       # (32*16*16, 1024)

  # the grid before PCA, used to fit the basis
  def grid(self, video):
    out = self.tokens(video)
    if self.pooling == "mean_all":
      return out.mean(0)
    t = out.shape[0] // (GRID * GRID)
    out = out.reshape(t, GRID, GRID, -1)                 # tubelets are flattened (t, h, w)
    g = out[-1] if self.pooling == "spatial_last" else out.mean(0)      # (16, 16, 1024)
    n = self.spatial_pool
    if n > 1:
      g = g.reshape(GRID // n, n, GRID // n, n, -1).mean((1, 3))        # (16/n, 16/n, 1024)
    return g

  def encode(self, video):
    g = self.grid(video)
    if self.pooling == "mean_all":
      return g.cpu().numpy()
    g = (g - self.pca["mean"]) @ self.pca["components"].T   # (16, 16, k)
    return g.reshape(-1).cpu().numpy()

  # online use: list of RGB uint8 arrays, newest last
  def encode_frames(self, frames):
    picked = select_history(frames, self.clip_frames, self.stride)
    return self.encode(torch.from_numpy(np.stack(picked)).permute(0, 3, 1, 2))

  @property
  def name(self):
    if self.pooling == "mean_all":
      return "mean_all_encoder_tokens"
    return f"{self.pooling}_pca{self.pca['components'].shape[0]}_grid{GRID // self.spatial_pool}"


def fit_pca(encoder, root, manifest, clips, dims, seed):
  rng = np.random.default_rng(seed)
  train = [s for s in manifest["states"] if s["split"] == "train"]
  chosen = [train[i] for i in rng.choice(len(train), min(clips, len(train)), replace=False)]
  grids = []
  start = time.monotonic()
  for i, state in enumerate(chosen, 1):
    grids.append(encoder.grid(load_frames(root, state, encoder.clip_frames, encoder.stride)).reshape(-1, encoder.model.config.hidden_size).cpu())
    if i % 20 == 0:
      print(f"pca fit: {i}/{len(chosen)} clips, {(time.monotonic()-start)/i:.2f}s/clip", flush=True)
  x = torch.cat(grids)                                   # (clips*256, 1024)
  mean = x.mean(0)
  _, s, v = torch.pca_lowrank(x - mean, q=dims, center=False, niter=4)
  explained = (s**2 / ((x - mean)**2).sum()).tolist()
  print(f"pca: {dims} components explain {sum(explained):.3f} of patch variance", flush=True)
  return {"mean": mean.numpy(), "components": v.T.contiguous().numpy(), "explained": explained,
          "fit_keys": [s["key"] for s in chosen]}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--run", required=True, type=pathlib.Path)
  p.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
  p.add_argument("--revision", default=PINNED_REVISION)
  p.add_argument("--pooling", choices=("mean_all", "spatial_last", "spatial_mean"), default="spatial_mean")
  p.add_argument("--pca-dims", type=int, default=16)
  p.add_argument("--spatial-pool", type=int, default=2, help="average NxN patches before PCA; 1 keeps 16x16")
  p.add_argument("--clip-frames", type=int, default=16, help="frames per clip, even; 64 is the original")
  p.add_argument("--frame-stride", type=int, default=1, help="control steps between clip frames")
  p.add_argument("--pca-clips", type=int, default=300, help="training clips used to fit the patch PCA")
  p.add_argument("--pca", type=pathlib.Path, help="reuse this pca.npz instead of fitting one, so several runs share a basis")
  p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--resume", action="store_true")
  args = p.parse_args()
  if not torch.cuda.is_available():
    p.error("CUDA unavailable")
  if args.clip_frames < 2 or args.clip_frames % 2 or args.frame_stride < 1:
    p.error("clip-frames must be even and >= 2 (tubelet size 2); frame-stride >= 1")
  manifest = load_run(args.run)
  issues = audit(args.run, manifest, images=False)["problems"]
  if issues:
    p.error("dataset audit failed: " + "; ".join(issues[:5]))
  out = args.run / "features"
  signature = {"manifest_sha256": digest(args.run / "manifest.json"), "model": args.model,
               "pooling": args.pooling, "pca_dims": args.pca_dims,
               "clip_frames": args.clip_frames, "frame_stride": args.frame_stride, "dtype": args.dtype,
               "spatial_pool": args.spatial_pool}
  done, vectors = 0, None
  partial = out / "latents.partial.npy"
  if out.exists():
    if not args.resume or (out / "meta.json").exists():
      p.error("features exists; use --resume for an interrupted encoding")
    progress = json.loads((out / "progress.json").read_text())
    if any(progress[k] != v for k, v in signature.items()):
      p.error("interrupted cache belongs to a different run/model/pooling")
    done = progress["encoded"]
    if done:
      vectors = np.load(partial, mmap_mode="r+")
    print(f"resuming after {done} encoded clips", flush=True)
  else:
    out.mkdir()
    write_json(out / "progress.json", {**signature, "encoded": 0})
  encoder = Encoder(args.model, args.revision, args.pooling, clip_frames=args.clip_frames, stride=args.frame_stride,
                    dtype=args.dtype, spatial_pool=args.spatial_pool)
  if args.pooling != "mean_all":
    basis = out / "pca.npz"
    if args.pca is not None and not basis.exists():
      # same basis as another run, so their features can be trained on together
      shutil.copyfile(args.pca, basis)
      write_json(out / "pca_fit.json", {"copied_from": str(args.pca.resolve()), "sha256": digest(basis)})
    if not basis.exists():
      pca = fit_pca(encoder, args.run, manifest, args.pca_clips, args.pca_dims, args.seed)
      np.savez(basis, mean=pca["mean"], components=pca["components"], explained=np.asarray(pca["explained"]))
      write_json(out / "pca_fit.json", {"clips": pca["fit_keys"], "explained": pca["explained"]})
    saved = np.load(basis)
    encoder.pca = {"mean": torch.as_tensor(saved["mean"], device="cuda"),
                   "components": torch.as_tensor(saved["components"], device="cuda")}
  keys = [s["key"] for s in manifest["states"]]
  start = time.monotonic()
  for i, state in enumerate(manifest["states"], 1):
    if i <= done:
      continue
    z = encoder.encode(load_frames(args.run, state, args.clip_frames, args.frame_stride))
    if z.ndim != 1 or not np.isfinite(z).all():
      raise ValueError(f"invalid vector at {state['key']}")
    if vectors is None:
      vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(keys), len(z)))
    vectors[i-1] = z
    if i % 50 == 0 or i == len(keys):
      vectors.flush()
      write_json(out / "progress.json", {**signature, "encoded": i})
      print(f"encoded {i}/{len(keys)}; {(time.monotonic()-start)/max(i-done, 1):.2f}s/clip; "
            f"peak CUDA {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
  vectors.flush()
  del vectors
  partial.replace(out / "latents.npy")
  write_json(out / "meta.json", {"manifest_sha256": digest(args.run / "manifest.json"),
             "latents_sha256": digest(out / "latents.npy"), "keys": keys, "model": args.model,
             "model_revision": encoder.revision, "pooling": encoder.name, "grid": GRID,
             "pca_sha256": digest(out / "pca.npz") if args.pooling != "mean_all" else None,
             "camera": manifest["camera"], "clip_frames": args.clip_frames, "frame_stride": args.frame_stride,
             "dtype": args.dtype, "spatial_pool": args.spatial_pool,
             "performance": {"session_seconds": time.monotonic()-start,
                             "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                             "gpu": torch.cuda.get_device_name()},
             "provenance": {**provenance(), "torch": torch.__version__}})
  (out / "progress.json").unlink()
  shape = np.load(out / "latents.npy", mmap_mode="r").shape
  print(f"saved {shape[0]} vectors of size {shape[1]} with pooling {encoder.name}")


if __name__ == "__main__":
  main()
