# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import pathlib

import numpy as np
import torch
from torch import nn

from .common import load_run, state_arrays, write_json
from .train import fit, normalizer, mlp, model_folder

# The obstacle model O(window of the obstacle map around the hand, hand height, gripper, held) -> proximity
# and P(contact). The map is env.obstacle_map(), the depth image with the robot and cube pixels dropped. A
# window around the hand's cell is the same geometry wherever the hand is, which is what makes it learnable
# from a few hundred episodes. The planners score the imagined hand with it.
#   python -m world_model.train_obstacle --run data/episodes --tag current --epochs 30


WINDOW = 9
CUBE_HALF = .0225   # half size of the unit cube, times the layout's scale


# Per-cell features of the WINDOW x WINDOW cells around the hand's cell of a batch of maps (n, cells, cells)
# in metres: the cell's height, its offset from the hand in metres (x, y), the hand's height above the table,
# the gripper width and command, held -> (n, WINDOW*WINDOW, 7). Cells outside the table have height 0.
FEATURES = 7


def cell_features(hm, p, held, table, centre, half, cells):
  n = len(hm)
  ee = np.asarray(p)[:, 14:17]
  size = 2 * np.asarray(half) / cells
  rel = (ee[:, :2] - (np.asarray(centre) - np.asarray(half))) / size
  cell = np.clip(np.floor(rel).astype(int), 0, cells - 1)
  r = WINDOW // 2
  padded = np.zeros((n, cells + WINDOW - 1, cells + WINDOW - 1), np.float32)
  padded[:, r:r + cells, r:r + cells] = hm
  rows = cell[:, 0, None] + np.arange(WINDOW)[None]
  cols = cell[:, 1, None] + np.arange(WINDOW)[None]
  heights = padded[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]].reshape(n, -1)
  # offset of each cell's centre from the hand
  grid = np.stack(np.meshgrid(np.arange(WINDOW) - r, np.arange(WINDOW) - r, indexing="ij"), -1).reshape(-1, 2)
  offset = (cell[:, None, :] + .5 + grid[None] - rel[:, None, :]) * size
  rest = np.concatenate((ee[:, 2:3] - table, np.asarray(p)[:, 18:20], np.asarray(held, np.float32)[:, None]), 1)
  x = np.concatenate((heights[:, :, None], offset, np.repeat(rest[:, None, :], WINDOW * WINDOW, 1)), 2)
  return x.astype(np.float32)


# The model's raw input, flat: (n, WINDOW*WINDOW*FEATURES)
def raw_inputs(hm, p, held, table, centre, half, cells):
  return cell_features(hm, p, held, table, centre, half, cells).reshape(len(hm), -1)


# One small network scores every cell from its own features (shared weights) and a max over the cells lets
# the nearest or tallest obstacle dominate, which keeps it from memorising scenes
class ObstacleModel(nn.Module):

  def __init__(self, inputs):
    super().__init__()
    self.cell = mlp(FEATURES, 2, width=64)

  def forward(self, x):
    y = self.cell(x.reshape(len(x), -1, FEATURES))
    return torch.stack((-y[:, :, 0].amax(1), y[:, :, 1].amax(1)), 1)


def load_obstacle_model(root, tag):
  ck = torch.load(model_folder(root, tag) / "models" / "obstacle.pt", map_location="cpu", weights_only=True)
  model = ObstacleModel(ck["inputs"])
  model.load_state_dict(ck["model"])
  return model.eval(), ck


# Proximity (clipped to [-1, 0]) and contact probability for one obstacle map (cells, cells) and a batch of
# robot states p (n, p_dim) and held readings (n,)
def obstacle_terms(model, ck, obstacle_map, p, held):
  hm = np.repeat(np.asarray(obstacle_map, np.float32)[None], len(p), 0)
  return terms_from_raw(model, ck, raw_inputs(hm, p, held, ck["table"], ck["centre"], ck["half"], ck["cells"]))


@torch.inference_mode()
def terms_from_raw(model, ck, raw):
  x = (torch.as_tensor(raw) - ck["x_mean"]) / ck["x_std"]
  y = model(x.to(next(model.parameters()).device)).cpu().numpy()
  return np.clip(y[:, 0], -1, 0), 1 / (1 + np.exp(-y[:, 1]))


# Distance from points (n, 3) to an obstacle's solid (a box with a yaw or a cylinder standing on the table)
def obstacle_distance(points, ob, table):
  d = np.asarray(points, np.float32) - np.array([ob["pos"][0], ob["pos"][1], table], np.float32)
  if ob["kind"] == "cylinder":
    radial = np.clip(np.hypot(d[:, 0], d[:, 1]) - ob["size"][0], 0, None)
    half_height = ob["size"][1]
  else:
    c, s = np.cos(ob["yaw"]), np.sin(ob["yaw"])
    local = np.stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]), 1)
    radial = np.linalg.norm(np.clip(np.abs(local) - np.asarray(ob["size"][:2]), 0, None), axis=1)
    half_height = ob["size"][2]
  vertical = np.clip(np.abs(d[:, 2] - half_height) - half_height, 0, None)
  return np.hypot(radial, vertical)


# The obstacle map of every scene of the run, rendered from its stored layout with the arm at home (cached)
def scene_maps(run, manifest):
  path = run / "features" / "obstacle_maps.npz"
  if path.exists():
    cached = np.load(path, allow_pickle=True)
    return dict(zip(cached["scenes"], cached["maps"]))
  from environment.config import Config
  from environment.env import PickPlaceEnv
  from environment.scene import EpisodeLayout
  env = PickPlaceEnv(Config.nested(manifest["config"]))
  maps = {}
  for i, setting in enumerate(manifest["scene_settings"]):
    env.reset(seed=setting["seed"], layout=EpisodeLayout.from_dict(setting["layout"]))
    maps[setting["rollout"]] = env.obstacle_map()
    if i % 50 == 0:
      print(f"rendered {i + 1}/{len(manifest['scene_settings'])} scene maps", flush=True)
  env.close()
  np.savez(path, scenes=np.asarray(list(maps)), maps=np.stack(list(maps.values())))
  return maps


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True, type=pathlib.Path)
  ap.add_argument("--tag", required=True)
  ap.add_argument("--epochs", type=int, default=30)
  ap.add_argument("--batch-size", type=int, default=256)
  ap.add_argument("--lr", type=float, default=0.001)
  ap.add_argument("--weight-decay", type=float, default=0.05)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--overwrite", action="store_true")
  args = ap.parse_args()
  torch.manual_seed(args.seed)

  manifest = load_run(args.run)
  states = manifest["states"]
  cfg = manifest["config"]
  table, centre, half = float(cfg["table"]["height"]), [float(v) for v in cfg["table"]["center"]], [float(v) for v in cfg["table"]["half_size"]]
  safe = float(cfg["rewards"]["safe_distance"])
  maps = scene_maps(args.run, manifest)
  cells = next(iter(maps.values())).shape[0]
  layouts = {s["rollout"]: s["layout"] for s in manifest["scene_settings"]}
  hm = np.stack([maps[s["scene"]] for s in states])
  p, xyz, held = state_arrays(manifest)
  # the recorded proximity measures the arm only; the carried cube is part of what must not touch an
  # obstacle, so the label is the closer of the two (contact already counts the cube)
  arm = safe * (1 + np.asarray([s["reward_components"].get("proximity", 0.0) for s in states], np.float32))
  cube = np.full(len(states), safe, np.float32)
  for scene, layout in layouts.items():
    rows = np.flatnonzero((np.asarray([s["scene"] for s in states]) == scene) & (held > .5))
    for ob in layout["obstacles"]:
      cube[rows] = np.minimum(cube[rows], obstacle_distance(xyz[rows], ob, table) - CUBE_HALF * layout["scale"])
  proximity = -np.clip(1 - np.minimum(arm, np.clip(cube, 0, None)) / safe, 0, 1).astype(np.float32)
  contact = np.asarray([bool(s.get("obstacle_contact")) or s["reward_components"].get("collision", 0.0) != 0 for s in states], np.float32)
  splits = np.asarray([s["split"] for s in states])
  ids = {s: np.flatnonzero(splits == s) for s in ("train", "val")}
  raw = raw_inputs(hm, p, held, table, centre, half, cells)
  x_mean, x_std = normalizer(raw[ids["train"]])
  # the same normalisation for every cell, and a floor on the heights' spread (most cells are empty)
  x_mean, x_std = x_mean.reshape(-1, FEATURES).mean(0).repeat(WINDOW * WINDOW), x_std.reshape(-1, FEATURES).mean(0).repeat(WINDOW * WINDOW)
  x_std[::FEATURES] = torch.maximum(x_std[::FEATURES], torch.tensor(.02))
  x = (torch.from_numpy(raw) - x_mean) / x_std
  targets = torch.from_numpy(np.stack((proximity, contact), 1))
  folder = args.run / "attempts" / args.tag
  pos_weight = torch.tensor(float((contact[ids["train"]] < .5).sum() / max(contact[ids["train"]].sum(), 1)) ** .5)
  print(f"obstacle: {len(states)} states, {int((proximity < 0).sum())} penalised ({int(((proximity < 0) & (cube < arm)).sum())} by the cube), "
        f"{int(contact.sum())} contacts, {cells}x{cells} maps, window {WINDOW}", flush=True)

  def loss_fn(model, batch):
    inputs, target = batch
    y = model(inputs)
    regression = (y[:, 0] - target[:, 0]).square().mean()
    classification = nn.functional.binary_cross_entropy_with_logits(y[:, 1], target[:, 1], pos_weight=pos_weight.to(y.device))
    return regression + classification, {"proximity_mse": regression.item(), "contact_bce": classification.item()}

  model = ObstacleModel(raw.shape[1])
  fit(folder, "obstacle", model, [x[ids["train"]], targets[ids["train"]]], [x[ids["val"]], targets[ids["val"]]], loss_fn,
      {"x_mean": x_mean, "x_std": x_std, "inputs": raw.shape[1], "cells": cells, "window": WINDOW, "table": table,
       "centre": centre, "half": half}, args)

  model, ck = load_obstacle_model(args.run, args.tag)
  v = ids["val"]
  prox, pc = terms_from_raw(model, ck, raw[v])
  penalised = proximity[v] != 0
  truth = contact[v] > .5
  guess = pc > .5
  tp, fp, fn = int((guess & truth).sum()), int((guess & ~truth).sum()), int((~guess & truth).sum())
  order = np.argsort(pc)
  ranks = np.empty(len(pc)); ranks[order] = np.arange(len(pc))
  auc = float((ranks[truth].sum() - truth.sum() * (truth.sum() - 1) / 2) / max(truth.sum() * (~truth).sum(), 1)) if truth.any() and (~truth).any() else None
  report = {"n": int(len(v)), "proximity_mae": float(np.abs(prox - proximity[v]).mean()),
            "proximity_mae_when_penalised": float(np.abs(prox - proximity[v])[penalised].mean()) if penalised.any() else None,
            "proximity_mae_when_free": float(np.abs(prox - proximity[v])[~penalised].mean()),
            "contact": {"positives": int(truth.sum()), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "auc": auc}}
  write_json(folder / "reports" / "obstacle_validation.json", report)
  print(f"obstacle validation: {report}")


if __name__ == "__main__":
  main()
