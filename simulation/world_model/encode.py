# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

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

from .common import load_run, check_manifest, write_json

# Turn every recorded state's camera clip into one latent vector z with the frozen V-JEPA 2 encoder (CUDA).
# The clip is the last --clip-frames static camera frames up to and including the state's own frame.
# --pooling decides what z is:
#   spatial_mean  (default) the 16x16 patch grid averaged over time, 2x2 pooled to 8x8 cells, each cell
#                 reduced by PCA to --pca-dims numbers -> z has 8*8*16 = 1024 values and keeps WHERE
#                 things are, which the cube (about 10 px) needs
#   spatial_last  the same grid from the last time slice only (jittery, kept for comparison)
#   mean_all      the mean over all tokens -> 1024 values, the original; the cube is lost in the arm
# The PCA basis is fitted on --pca-clips training clips and saved to features/pca.npz; --pca reuses another
# run's basis so several runs can be merged and trained on together (merge_runs.py). control_pipeline
# builds the same Encoder online from features/meta.json.
#   python -m world_model.encode --run data/full_test1
#   python -m world_model.encode --run data/obstacles_test1_fullpca --pca data/full_test1/features/pca.npz

MODEL = "facebook/vjepa2-vitl-fpc64-256"
# the weights revision the whole project was run with
PINNED_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
GRID = 16


# Pick clip_frames frames from a history (newest last), stride frames apart, repeating the oldest when short
def select_history(history, clip_frames, stride):
  picked = [history[max(len(history) - 1 - i * stride, 0)] for i in range(clip_frames)]
  return picked[::-1]


# Consecutive states share most of their clip, so decoded frames are cached
@functools.lru_cache(maxsize=2048)
def read_rgb(path):
  bgr = cv2.imread(path)
  if bgr is None:
    raise FileNotFoundError(path)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# The clip of one state as a (frames, 3, H, W) uint8 tensor
def load_frames(root, state, clip_frames, stride):
  frames = [read_rgb(str(pathlib.Path(root) / name)) for name in select_history(state["frames"], clip_frames, stride)]
  return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


# Frozen V-JEPA 2 plus the pooling that makes z, used offline here and online by the controller
class Encoder:
  DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

  def __init__(self, model, revision, pooling, pca=None, device="cuda", clip_frames=16, stride=1,
               dtype="fp16", spatial_pool=2):
    self.clip_frames, self.stride, self.dtype, self.spatial_pool = clip_frames, stride, dtype, spatial_pool
    self.pooling, self.device = pooling, device
    self.processor = AutoVideoProcessor.from_pretrained(model, revision=revision)
    self.model = AutoModel.from_pretrained(model, revision=revision, attn_implementation="sdpa")
    # fp16 is about 3% off fp32 per token, bf16 about 18%, at the same speed
    self.model.to(device, dtype=self.DTYPES[dtype]).eval()
    self.revision = getattr(self.model.config, "_commit_hash", None) or revision
    self.pca = None
    if pca is not None:
      self.pca = {"mean": torch.as_tensor(pca["mean"], device=device),
                  "components": torch.as_tensor(pca["components"], device=device)}

  # Same as the HF video processor (resize shortest edge 292, centre crop 256, normalise) but on the GPU,
  # the CPU processor took longer than the encoder itself
  def preprocess(self, video):
    x = video.to(self.device).float() / 255
    x = torch.nn.functional.interpolate(x, size=(292, 292), mode="bilinear", align_corners=False, antialias=True)
    x = x[:, :, 18:274, 18:274]
    mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
    std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)

  # All tokens of the clip, (time/2 * 16 * 16, 1024)
  @torch.inference_mode()
  def tokens(self, video):
    with torch.autocast(self.device, dtype=self.DTYPES[self.dtype], enabled=self.dtype != "fp32"):
      out = self.model(pixel_values_videos=self.preprocess(video), skip_predictor=True).last_hidden_state
    return out.float().squeeze(0)

  # The pooled patch grid before PCA, (16/n, 16/n, 1024), or the plain mean for mean_all
  def grid(self, video):
    out = self.tokens(video)
    if self.pooling == "mean_all":
      return out.mean(0)
    t = out.shape[0] // (GRID * GRID)
    out = out.reshape(t, GRID, GRID, -1)
    g = out[-1] if self.pooling == "spatial_last" else out.mean(0)
    n = self.spatial_pool
    if n > 1:
      g = g.reshape(GRID // n, n, GRID // n, n, -1).mean((1, 3))
    return g

  # The latent z of one clip
  def encode(self, video):
    g = self.grid(video)
    if self.pooling == "mean_all":
      return g.cpu().numpy()
    g = (g - self.pca["mean"]) @ self.pca["components"].T
    return g.reshape(-1).cpu().numpy()

  # Online use: a list of RGB uint8 frames, newest last
  def encode_frames(self, frames):
    picked = select_history(frames, self.clip_frames, self.stride)
    return self.encode(torch.from_numpy(np.stack(picked)).permute(0, 3, 1, 2))

  # Name written to meta.json, e.g. spatial_mean_pca16_grid8
  @property
  def name(self):
    if self.pooling == "mean_all":
      return "mean_all_encoder_tokens"
    return f"{self.pooling}_pca{self.pca['components'].shape[0]}_grid{GRID // self.spatial_pool}"


# Fit the per cell PCA on a random sample of training clips, label free
def fit_pca(encoder, root, manifest, clips, dims, seed):
  rng = np.random.default_rng(seed)
  train = [s for s in manifest["states"] if s["split"] == "train"]
  chosen = [train[i] for i in rng.choice(len(train), min(clips, len(train)), replace=False)]
  grids = []
  start = time.monotonic()
  for i, state in enumerate(chosen, 1):
    g = encoder.grid(load_frames(root, state, encoder.clip_frames, encoder.stride))
    grids.append(g.reshape(-1, encoder.model.config.hidden_size).cpu())
    if i % 20 == 0:
      print(f"pca fit: {i}/{len(chosen)} clips, {(time.monotonic() - start) / i:.2f}s/clip", flush=True)
  x = torch.cat(grids)
  mean = x.mean(0)
  _, s, v = torch.pca_lowrank(x - mean, q=dims, center=False, niter=4)
  explained = (s ** 2 / ((x - mean) ** 2).sum()).tolist()
  print(f"pca: {dims} components explain {sum(explained):.3f} of the patch variance", flush=True)
  return {"mean": mean.numpy(), "components": v.T.contiguous().numpy(), "explained": explained}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True, type=pathlib.Path)
  p.add_argument("--model", default=MODEL)
  p.add_argument("--revision", default=PINNED_REVISION)
  p.add_argument("--pooling", choices=("mean_all", "spatial_last", "spatial_mean"), default="spatial_mean")
  p.add_argument("--pca-dims", type=int, default=16)
  p.add_argument("--spatial-pool", type=int, default=2, help="average NxN patches before the PCA, 2 gives 8x8 cells")
  p.add_argument("--clip-frames", type=int, default=16, help="frames per clip, even; the original used 64")
  p.add_argument("--frame-stride", type=int, default=1, help="control steps between clip frames")
  p.add_argument("--pca-clips", type=int, default=300, help="training clips the PCA basis is fitted on")
  p.add_argument("--pca", type=pathlib.Path, help="reuse this pca.npz instead of fitting one")
  p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--resume", action="store_true", help="continue an interrupted encoding")
  args = p.parse_args()
  if not torch.cuda.is_available():
    p.error("encoding needs CUDA")
  if args.clip_frames < 2 or args.clip_frames % 2 or args.frame_stride < 1:
    p.error("clip-frames must be even and >= 2 (V-JEPA tubelets span 2 frames); frame-stride >= 1")

  manifest = load_run(args.run)
  problems = check_manifest(manifest)
  if problems:
    p.error("manifest problems: " + "; ".join(problems[:5]))

  # progress.json marks an encoding that is still running, meta.json a finished one
  out = args.run / "features"
  partial = out / "latents.partial.npy"
  done, vectors = 0, None
  if out.exists():
    if (out / "meta.json").exists():
      p.error(f"{out} is already encoded; delete it to encode again")
    if not args.resume:
      p.error(f"{out} holds an interrupted encoding; add --resume to continue it")
    done = json.loads((out / "progress.json").read_text())["encoded"]
    if done:
      vectors = np.load(partial, mmap_mode="r+")
    print(f"resuming after {done} clips", flush=True)
  else:
    out.mkdir()

  encoder = Encoder(args.model, args.revision, args.pooling, clip_frames=args.clip_frames, stride=args.frame_stride,
                    dtype=args.dtype, spatial_pool=args.spatial_pool)
  if args.pooling != "mean_all":
    basis = out / "pca.npz"
    if not basis.exists():
      if args.pca is not None:
        shutil.copyfile(args.pca, basis)
      else:
        pca = fit_pca(encoder, args.run, manifest, args.pca_clips, args.pca_dims, args.seed)
        np.savez(basis, mean=pca["mean"], components=pca["components"], explained=np.asarray(pca["explained"]))
    saved = np.load(basis)
    encoder.pca = {"mean": torch.as_tensor(saved["mean"], device="cuda"),
                   "components": torch.as_tensor(saved["components"], device="cuda")}

  # one vector per state, written into a memmap so an interrupted run can continue
  keys = [s["key"] for s in manifest["states"]]
  start = time.monotonic()
  for i, state in enumerate(manifest["states"], 1):
    if i <= done:
      continue
    z = encoder.encode(load_frames(args.run, state, args.clip_frames, args.frame_stride))
    if z.ndim != 1 or not np.isfinite(z).all():
      raise ValueError(f"invalid latent at {state['key']}")
    if vectors is None:
      vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(keys), len(z)))
    vectors[i - 1] = z
    if i % 50 == 0 or i == len(keys):
      vectors.flush()
      write_json(out / "progress.json", {"encoded": i})
      print(f"encoded {i}/{len(keys)}; {(time.monotonic() - start) / max(i - done, 1):.2f}s/clip", flush=True)

  vectors.flush()
  del vectors
  partial.replace(out / "latents.npy")
  write_json(out / "meta.json", {"keys": keys, "model": args.model, "model_revision": encoder.revision,
             "pooling": encoder.name, "grid": GRID, "camera": manifest["camera"], "clip_frames": args.clip_frames,
             "frame_stride": args.frame_stride, "dtype": args.dtype, "spatial_pool": args.spatial_pool,
             "seconds": time.monotonic() - start})
  (out / "progress.json").unlink()
  print(f"saved {len(keys)} latents of size {len(z)} with pooling {encoder.name}")


if __name__ == "__main__":
  main()
