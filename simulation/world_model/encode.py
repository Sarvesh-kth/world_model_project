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
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor

from .common import load_run, check_manifest, write_json

# Turn every recorded state into one latent vector z with the frozen V-JEPA 2 encoder (CUDA). z is the
# concatenation of one block per camera plus the table height map:
#   static     the last --clip-frames static frames up to the state's own frame, 16x16 patch grid averaged
#              over time, kept at 16x16 cells (--static-pool 1), each cell reduced by PCA to --static-pca
#              numbers -> 16 * 16 * 32 = 8192
#   wrist      the same from the wrist camera (--wrist-clip-frames), 2x2 pooled to 8x8 cells (--wrist-pool 2),
#              PCA to --wrist-pca numbers per cell -> 8 * 8 * 16 = 1024
#   heightmap  the 8x8 table height map from the static depth image, scaled to [0, 1] -> 64 (--no-heightmap drops it)
# The grid keeps WHERE things are, which the cube (about 15 px, one patch) needs: on the 8x8 / 16-per-cell grid
# the cube could not be placed better than 3-4 cm when the hand was away from it (the moment the planner picks
# its direction), on the full 16x16 grid with 32 per cell a spatial-softmax readout gets 0.7 cm from the same
# frames (train.py GridReadout). mean_all (the original, the mean over all tokens) loses the position entirely.
# One PCA basis per camera is fitted on --pca-clips training clips and saved to features/pca.npz; --pca reuses
# another run's basis so runs can be merged (merge_runs.py). control_pipeline builds the same Encoder online
# from features/meta.json. --recamera wrist redoes just that camera's block of a finished encoding (after a
# change of its clip length), refitting its basis unless --pca gives one.
#   python -m world_model.encode --run data/full_v2
#   python -m world_model.encode --run data/obstacles_v2 --pca data/full_v2/features/pca.npz
#   python -m world_model.encode --run data/full_v2 --recamera wrist

MODEL = "facebook/vjepa2-vitl-fpc64-256"
# the weights revision the whole project was run with
PINNED_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
GRID = 16
FRAME_KEYS = {"static": "frames", "wrist": "wrist_frames"}
# cells averaged before the PCA, per camera: the static view keeps every patch, the wrist view (close-up,
# the hand fills it) is pooled 2x2; and the PCA numbers kept per cell
SPATIAL_POOL = {"static": 1, "wrist": 2}
PCA_DIMS = {"static": 32, "wrist": 16}


# A per camera setting from meta.json: the dict as stored, or one number (old runs) for every camera
def per_camera(value, cameras, default):
  if isinstance(value, dict):
    return {c: int(value[c]) for c in cameras}
  return {c: int(default[c] if value is None else value) for c in cameras}


# Pick clip_frames frames from a history (newest last), stride frames apart, repeating the oldest when short
def select_history(history, clip_frames, stride):
  picked = [history[max(len(history) - 1 - i * stride, 0)] for i in range(clip_frames)]
  return picked[::-1]


# Consecutive states share most of their clip, so decoded frames are cached
@functools.lru_cache(maxsize=4096)
def read_rgb(path):
  bgr = cv2.imread(path)
  if bgr is None:
    raise FileNotFoundError(path)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# The clips of one state, camera -> (frames, 3, H, W) uint8 tensor
def load_clips(root, state, encoder, cameras=None):
  clips = {}
  for camera in cameras or encoder.cameras:
    names = select_history(state[FRAME_KEYS[camera]], encoder.clip_frames[camera], encoder.stride)
    frames = [read_rgb(str(pathlib.Path(root) / name)) for name in names]
    clips[camera] = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
  return clips


# Frozen V-JEPA 2 plus the pooling that makes z, used offline here and online by the controller.
# clip_frames and spatial_pool are dicts camera -> frames per clip / cells pooled, pca a dict
# camera -> {"mean", "components"} (None before fitting; the components per camera set the block sizes)
class Encoder:
  DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

  def __init__(self, model, revision, pooling, cameras=("static", "wrist"), clip_frames=None, pca=None,
               device="cuda", stride=1, dtype="fp16", spatial_pool=None, heightmap=True, heightmap_scale=0.25):
    self.cameras = tuple(cameras)
    self.clip_frames = dict(clip_frames or {"static": 16, "wrist": 16})
    self.stride, self.dtype = stride, dtype
    self.spatial_pool = per_camera(spatial_pool, self.cameras, SPATIAL_POOL)
    self.pooling, self.device = pooling, device
    self.heightmap, self.heightmap_scale = heightmap, heightmap_scale
    self.processor = AutoVideoProcessor.from_pretrained(model, revision=revision)
    self.model = AutoModel.from_pretrained(model, revision=revision, attn_implementation="sdpa")
    # fp16 is about 3% off fp32 per token, bf16 about 18%, at the same speed
    self.model.to(device, dtype=self.DTYPES[dtype]).eval()
    self.revision = getattr(self.model.config, "_commit_hash", None) or revision
    self.pca = {}
    if pca is not None:
      self.set_pca(pca)

  # The basis from pca.npz (arrays <camera>_mean, <camera>_components) or an already split dict
  def set_pca(self, pca):
    for camera in self.cameras:
      mean, comps = (pca[camera]["mean"], pca[camera]["components"]) if camera in pca else (pca[f"{camera}_mean"], pca[f"{camera}_components"])
      self.pca[camera] = {"mean": torch.as_tensor(np.asarray(mean), device=self.device),
                          "components": torch.as_tensor(np.asarray(comps), device=self.device)}

  # Same as the HF video processor (resize shortest edge 292, centre crop 256, normalise) but on the GPU,
  # the CPU processor took longer than the encoder itself
  def preprocess(self, video):
    x = video.to(self.device).float() / 255
    x = torch.nn.functional.interpolate(x, size=(292, 292), mode="bilinear", align_corners=False, antialias=True)
    x = x[:, :, 18:274, 18:274]
    mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
    std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)

  # All tokens of one clip, (time/2 * 16 * 16, 1024)
  @torch.inference_mode()
  def tokens(self, video):
    with torch.autocast(self.device, dtype=self.DTYPES[self.dtype], enabled=self.dtype != "fp32"):
      out = self.model(pixel_values_videos=self.preprocess(video), skip_predictor=True).last_hidden_state
    return out.float().squeeze(0)

  # The pooled patch grid of one camera's clip before PCA, (cells, cells, 1024), or the plain mean for mean_all
  def grid(self, video, camera):
    out = self.tokens(video)
    if self.pooling == "mean_all":
      return out.mean(0)
    t = out.shape[0] // (GRID * GRID)
    out = out.reshape(t, GRID, GRID, -1)
    g = out[-1] if self.pooling == "spatial_last" else out.mean(0)
    n = self.spatial_pool[camera]
    if n > 1:
      g = g.reshape(GRID // n, n, GRID // n, n, -1).mean((1, 3))
    return g

  # The latent z of one state: clips is camera -> video tensor, heightmap the (cells, cells) map in metres
  def encode(self, clips, heightmap=None):
    parts = []
    for camera in self.cameras:
      g = self.grid(clips[camera], camera)
      if self.pooling != "mean_all":
        g = ((g - self.pca[camera]["mean"]) @ self.pca[camera]["components"].T).reshape(-1)
      parts.append(g.cpu().numpy())
    if self.heightmap:
      if heightmap is None:
        raise ValueError("the encoder expects a height map")
      parts.append(np.asarray(heightmap, np.float32).reshape(-1) / self.heightmap_scale)
    return np.concatenate(parts).astype(np.float32)

  # Online use: histories is camera -> list of RGB uint8 frames (newest last)
  def encode_frames(self, histories, heightmap=None):
    clips = {}
    for camera in self.cameras:
      picked = select_history(histories[camera], self.clip_frames[camera], self.stride)
      clips[camera] = torch.from_numpy(np.stack(picked)).permute(0, 3, 1, 2)
    return self.encode(clips, heightmap)

  # Cells per side of each camera's grid, e.g. {"static": 16, "wrist": 8}
  def cells(self):
    return {camera: GRID // self.spatial_pool[camera] for camera in self.cameras}

  # PCA numbers per cell of each camera, from the bases
  def dims(self):
    return {camera: int(self.pca[camera]["components"].shape[0]) for camera in self.cameras}

  # Sizes of the blocks of z in order, e.g. {"static": 8192, "wrist": 1024, "heightmap": 64}
  def layout(self, cells=8):
    if self.pooling == "mean_all":
      out = {camera: 1024 for camera in self.cameras}
    else:
      out = {camera: self.cells()[camera] ** 2 * self.dims()[camera] for camera in self.cameras}
    if self.heightmap:
      out["heightmap"] = cells * cells
    return out

  # Name written to meta.json, e.g. spatial_mean_static16x32_wrist8x16
  @property
  def name(self):
    if self.pooling == "mean_all":
      return "mean_all_encoder_tokens"
    return self.pooling + "_" + "_".join(f"{c}{self.cells()[c]}x{self.dims()[c]}" for c in self.cameras)


# Fit the per cell PCA of every camera (or just the given ones) on a random sample of training clips, label
# free; dims is camera -> numbers kept per cell
def fit_pca(encoder, root, manifest, clips, dims, seed, cameras=None):
  cameras = cameras or encoder.cameras
  rng = np.random.default_rng(seed)
  train = [s for s in manifest["states"] if s["split"] == "train"]
  chosen = [train[i] for i in rng.choice(len(train), min(clips, len(train)), replace=False)]
  grids = {camera: [] for camera in cameras}
  for state in tqdm(chosen, desc="pca fit", unit="clip"):
    for camera, video in load_clips(root, state, encoder, cameras).items():
      grids[camera].append(encoder.grid(video, camera).reshape(-1, encoder.model.config.hidden_size).cpu())
  out = {}
  for camera in cameras:
    x = torch.cat(grids[camera])
    mean = x.mean(0)
    _, s, v = torch.pca_lowrank(x - mean, q=dims[camera], center=False, niter=4)
    explained = (s ** 2 / ((x - mean) ** 2).sum()).tolist()
    print(f"pca {camera}: {dims[camera]} components explain {sum(explained):.3f} of the patch variance", flush=True)
    out[f"{camera}_mean"] = mean.numpy()
    out[f"{camera}_components"] = v.T.contiguous().numpy()
    out[f"{camera}_explained"] = np.asarray(explained)
  return out


# Redo one camera's block of a finished encoding: the other blocks and the other camera's basis stay, so a
# changed clip length costs one forward per state instead of two. Writes into a copy of latents.npy with
# progress.json {"recamera": camera, "encoded": n}, so an interrupted pass continues with --resume. When
# the manifest has grown since the encoding, only the encoded states are redone; the new ones are then
# encoded in full by a later --resume, with the basis this pass leaves behind.
def recamera(args, manifest, cells, max_height):
  out = args.run / "features"
  camera = args.recamera
  meta = json.loads((out / "meta.json").read_text())
  keys = meta["keys"]
  if [s["key"] for s in manifest["states"][:len(keys)]] != keys:
    raise SystemExit("the manifest changed in front of the encoded states; delete features/ and encode again")
  if len(keys) < len(manifest["states"]):
    print(f"{len(manifest['states']) - len(keys)} states are not encoded yet, redoing the {camera} block of the {len(keys)} that are", flush=True)
  layout = meta["layout"]
  if camera not in layout:
    raise SystemExit(f"this encoding has no {camera} block")
  offset = sum(size for name, size in layout.items() if list(layout).index(name) < list(layout).index(camera))
  size = layout[camera]
  clip_frames = dict(meta["clip_frames"])
  clip_frames[camera] = args.clip_frames if camera == "static" else args.wrist_clip_frames
  encoder = Encoder(args.model, args.revision, args.pooling, cameras=meta["cameras"], clip_frames=clip_frames,
                    stride=meta.get("frame_stride", 1), dtype=meta.get("dtype", "fp16"), spatial_pool=meta.get("spatial_pool"),
                    heightmap=meta.get("heightmap", True), heightmap_scale=max_height)
  dims = {"static": args.static_pca, "wrist": args.wrist_pca}

  partial = out / "latents.partial.npy"
  progress = out / "progress.json"
  done = 0
  if progress.exists() and args.resume and json.loads(progress.read_text()).get("recamera") == camera:
    done = json.loads(progress.read_text())["encoded"]
    vectors = np.load(partial, mmap_mode="r+")
    basis = dict(np.load(out / "pca.recamera.npz"))
    print(f"resuming the {camera} block after {done} states", flush=True)
  else:
    vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(keys), sum(layout.values())))
    vectors[:] = np.load(out / "latents.npy", mmap_mode="r")
    # the new basis of this camera next to the untouched bases of the others
    basis = dict(np.load(out / "pca.npz"))
    if args.pca is not None:
      given = np.load(args.pca)
      basis.update({k: given[k] for k in given.files if k.startswith(camera + "_")})
    else:
      basis.update(fit_pca(encoder, args.run, manifest, args.pca_clips, dims, args.seed, [camera]))
    np.savez(out / "pca.recamera.npz", **basis)
  encoder.set_pca(basis)

  start = time.monotonic()
  bar = tqdm(total=len(keys), initial=done, desc=f"{camera} block", unit="state")
  for i, state in enumerate(manifest["states"][:len(keys)], 1):
    if i <= done:
      continue
    video = load_clips(args.run, state, encoder, [camera])[camera]
    g = encoder.grid(video, camera)
    block = ((g - encoder.pca[camera]["mean"]) @ encoder.pca[camera]["components"].T).reshape(-1).cpu().numpy()
    if block.shape != (size,) or not np.isfinite(block).all():
      raise ValueError(f"invalid {camera} block at {state['key']}")
    vectors[i - 1, offset:offset + size] = block
    bar.update(1)
    if i % 50 == 0 or i == len(keys):
      vectors.flush()
      write_json(progress, {"recamera": camera, "encoded": i})
  bar.close()

  vectors.flush()
  del vectors
  partial.replace(out / "latents.npy")
  (out / "pca.recamera.npz").replace(out / "pca.npz")
  meta.update(clip_frames=clip_frames, pca_dims=encoder.dims(), layout=encoder.layout(cells),
              seconds=meta.get("seconds", 0) + time.monotonic() - start)
  write_json(out / "meta.json", meta)
  progress.unlink()
  print(f"redid the {camera} block of {len(keys)} latents with {clip_frames[camera]} frames per clip")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True, type=pathlib.Path)
  p.add_argument("--model", default=MODEL)
  p.add_argument("--revision", default=PINNED_REVISION)
  p.add_argument("--cameras", nargs="+", default=["static", "wrist"], choices=("static", "wrist"))
  p.add_argument("--pooling", choices=("mean_all", "spatial_last", "spatial_mean"), default="spatial_mean")
  p.add_argument("--static-pool", type=int, default=SPATIAL_POOL["static"], help="patches averaged per cell of the static grid, 1 keeps all 16x16")
  p.add_argument("--wrist-pool", type=int, default=SPATIAL_POOL["wrist"], help="the same for the wrist grid, 2 gives 8x8 cells")
  p.add_argument("--static-pca", type=int, default=PCA_DIMS["static"], help="PCA numbers per cell of the static grid")
  p.add_argument("--wrist-pca", type=int, default=PCA_DIMS["wrist"], help="PCA numbers per cell of the wrist grid")
  p.add_argument("--clip-frames", type=int, default=16, help="static frames per clip, even; the original used 64")
  p.add_argument("--wrist-clip-frames", type=int, default=16, help="wrist frames per clip, even")
  p.add_argument("--frame-stride", type=int, default=1, help="control steps between clip frames")
  p.add_argument("--pca-clips", type=int, default=300, help="training clips the PCA basis is fitted on")
  p.add_argument("--pca", type=pathlib.Path, help="reuse this pca.npz instead of fitting one")
  p.add_argument("--no-heightmap", action="store_true", help="leave the table height map out of z")
  p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--resume", action="store_true", help="continue an interrupted encoding, or encode the states added since the last one")
  p.add_argument("--recamera", choices=("static", "wrist"), help="redo only this camera's block of a finished encoding (new clip length), its basis refitted unless --pca is given")
  args = p.parse_args()
  if not torch.cuda.is_available():
    p.error("encoding needs CUDA")
  for n in (args.clip_frames, args.wrist_clip_frames):
    if n < 2 or n % 2 or args.frame_stride < 1:
      p.error("clip frames must be even and >= 2 (V-JEPA tubelets span 2 frames); frame-stride >= 1")

  manifest = load_run(args.run)
  problems = check_manifest(manifest)
  if problems:
    p.error("manifest problems: " + "; ".join(problems[:5]))
  if "wrist" in args.cameras and not manifest["states"][0].get("wrist_frames"):
    p.error("this run has no wrist frames; encode it with --cameras static")
  if not args.no_heightmap and "heightmap" not in manifest["states"][0]:
    p.error("this run has no height maps; encode it with --no-heightmap")
  cells = manifest["config"]["cameras"]["heightmap"]["cells"]
  max_height = manifest["config"]["cameras"]["heightmap"]["max_height"]
  if args.recamera:
    return recamera(args, manifest, cells, max_height)

  # progress.json marks an encoding that is still running, meta.json a finished one; a finished one can be
  # extended when the manifest grew (new states appended, the old ones unchanged)
  out = args.run / "features"
  partial = out / "latents.partial.npy"
  keys = [s["key"] for s in manifest["states"]]
  done, vectors = 0, None
  if out.exists():
    if not args.resume:
      p.error(f"{out} is already encoded; add --resume to continue or extend it, or delete it to encode again")
    if (out / "meta.json").exists():
      old = json.loads((out / "meta.json").read_text())["keys"]
      if keys[:len(old)] != old:
        p.error("the manifest changed in front of the encoded states; delete features/ and encode again")
      if len(keys) == len(old):
        p.error(f"{out} is complete and the manifest has no new states")
      done = len(old)
      vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(keys), np.load(out / "latents.npy", mmap_mode="r").shape[1]))
      vectors[:done] = np.load(out / "latents.npy", mmap_mode="r")
      vectors.flush()
      # the progress marker goes first: if the encoder fails to load after this, --resume still finds the state
      write_json(out / "progress.json", {"encoded": done})
      (out / "meta.json").unlink()
      print(f"extending: {done} states encoded, {len(keys) - done} new", flush=True)
    else:
      done = json.loads((out / "progress.json").read_text())["encoded"]
      if done:
        vectors = np.load(partial, mmap_mode="r+")
      print(f"resuming after {done} clips", flush=True)
  else:
    out.mkdir()

  encoder = Encoder(args.model, args.revision, args.pooling, cameras=args.cameras,
                    clip_frames={"static": args.clip_frames, "wrist": args.wrist_clip_frames}, stride=args.frame_stride,
                    dtype=args.dtype, spatial_pool={"static": args.static_pool, "wrist": args.wrist_pool},
                    heightmap=not args.no_heightmap, heightmap_scale=max_height)
  if args.pooling != "mean_all":
    basis = out / "pca.npz"
    if not basis.exists():
      if args.pca is not None:
        shutil.copyfile(args.pca, basis)
      else:
        dims = {"static": args.static_pca, "wrist": args.wrist_pca}
        np.savez(basis, **fit_pca(encoder, args.run, manifest, args.pca_clips, dims, args.seed))
    encoder.set_pca(np.load(basis))

  # one vector per state, written into a memmap so an interrupted run can continue; the bar shows the rate
  # and the time left (about 0.27 s per state on an RTX 4060, 0.09 s on the cluster's GPU, GPU bound)
  start = time.monotonic()
  bar = tqdm(total=len(keys), initial=done, desc="encode", unit="state")
  for i, state in enumerate(manifest["states"], 1):
    if i <= done:
      continue
    z = encoder.encode(load_clips(args.run, state, encoder), state.get("heightmap"))
    if z.ndim != 1 or not np.isfinite(z).all():
      raise ValueError(f"invalid latent at {state['key']}")
    if vectors is None:
      vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(keys), len(z)))
    vectors[i - 1] = z
    bar.update(1)
    if i % 50 == 0 or i == len(keys):
      vectors.flush()
      write_json(out / "progress.json", {"encoded": i})
  bar.close()

  vectors.flush()
  del vectors
  partial.replace(out / "latents.npy")
  write_json(out / "meta.json", {"keys": keys, "model": args.model, "model_revision": encoder.revision,
             "pooling": encoder.name, "grid": GRID, "cameras": list(args.cameras), "clip_frames": encoder.clip_frames,
             "frame_stride": args.frame_stride, "dtype": args.dtype, "spatial_pool": encoder.spatial_pool,
             "cells": encoder.cells(), "pca_dims": encoder.dims() if args.pooling != "mean_all" else None,
             "heightmap": not args.no_heightmap, "heightmap_cells": cells, "heightmap_scale": max_height,
             "layout": encoder.layout(cells), "seconds": time.monotonic() - start})
  (out / "progress.json").unlink()
  print(f"saved {len(keys)} latents of size {len(z)} with pooling {encoder.name}, blocks {encoder.layout(cells)}")


if __name__ == "__main__":
  main()
