import argparse
import pathlib

import numpy as np
import torch
from torch import nn

from .common import load_run, write_json
from .train import fit, model_folder
from .train_obstacle import obstacle_distance

# The learned cost-to-go C(obstacle map, target) -> a GRID x GRID field of path length around the obstacles,
# used by the waypoint planner in place of the straight line (--cost-to-go) so a detour pays within the
# horizon. Trained on synthetic scenes, the labels come from a value iteration where blocked points are
# expensive but passable.
#   python -m world_model.train_cost_to_go --run data/episodes --tag current --scenes 20000

GRID = 32
RADIUS = .07
BLOCKING_HEIGHT = .12
BLOCKED = 20.0
CAP = 2.0
MAP_CELLS = 16


# The obstacle-only map of a layout's obstacles, (cells, cells) metres: a cell takes an obstacle's height
# when any of its 4 x 4 sample points lies inside the footprint
def rasterise(obstacles, table, centre, half, cells=MAP_CELLS):
  out = np.zeros((cells, cells), np.float32)
  if not obstacles:
    return out
  sub = (np.arange(cells * 4) + .5) / (cells * 4)
  xs = centre[0] - half[0] + sub * 2 * half[0]
  ys = centre[1] - half[1] + sub * 2 * half[1]
  pts = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
  pts = np.concatenate((pts, np.full((len(pts), 1), table + .01)), 1)
  for ob in obstacles:
    inside = obstacle_distance(pts, ob, table) <= 1e-6
    height = 2 * ob["size"][-1]
    hit = inside.reshape(cells, 4, cells, 4).any((1, 3))
    out[hit] = np.maximum(out[hit], height)
  return out


# Grid point coordinates (GRID, GRID, 2) and the step costs to the 8 neighbours
def grid_points(centre, half):
  xs = centre[0] - half[0] + (np.arange(GRID) + .5) * 2 * half[0] / GRID
  ys = centre[1] - half[1] + (np.arange(GRID) + .5) * 2 * half[1] / GRID
  return np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).astype(np.float32)


# Path length from every grid point to the target for a batch of maps (n, cells, cells) and targets (n, 2):
# value iteration over the 8-connected grid, crossing a blocked point costs BLOCKED times the step, (n, GRID, GRID)
def cost_fields(maps, targets, centre, half):
  n = len(maps)
  pts = grid_points(centre, half)
  step = 2 * np.asarray(half) / GRID
  # blocked: within RADIUS of a cell taller than BLOCKING_HEIGHT, the hand can carry the cube over the rest
  cells = maps.shape[1]
  mx = centre[0] - half[0] + (np.arange(cells) + .5) * 2 * half[0] / cells
  my = centre[1] - half[1] + (np.arange(cells) + .5) * 2 * half[1] / cells
  blocked = np.zeros((n, GRID, GRID), bool)
  for i in range(n):
    occ = np.argwhere(maps[i] > BLOCKING_HEIGHT)
    if len(occ):
      oc = np.stack((mx[occ[:, 0]], my[occ[:, 1]]), 1)
      blocked[i] = (np.linalg.norm(pts[:, :, None, :] - oc[None, None], axis=3) < RADIUS).any(2)
  cost = np.where(blocked, BLOCKED, 1.0).astype(np.float32)
  d = np.full((n, GRID, GRID), np.inf, np.float32)
  ti = np.clip(((targets - (np.asarray(centre) - np.asarray(half))) / (2 * np.asarray(half)) * GRID).astype(int), 0, GRID - 1)
  d[np.arange(n), ti[:, 0], ti[:, 1]] = 0
  shifts = [(dx, dy, float(np.hypot(dx * step[0], dy * step[1]))) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
  for _ in range(2 * GRID):
    best = d.copy()
    for dx, dy, length in shifts:
      moved = np.roll(d, (dx, dy), axis=(1, 2))
      if dx == 1: moved[:, 0, :] = np.inf
      if dx == -1: moved[:, -1, :] = np.inf
      if dy == 1: moved[:, :, 0] = np.inf
      if dy == -1: moved[:, :, -1] = np.inf
      best = np.minimum(best, moved + length * cost)
    if np.allclose(best, d, equal_nan=True):
      break
    d = best
  return np.minimum(d, CAP)


# Straight-line distance channel and a Gaussian blob at the target, (n, 2, GRID, GRID)
def target_channels(targets, centre, half):
  pts = grid_points(centre, half)
  diff = pts[None] - np.asarray(targets, np.float32)[:, None, None, :]
  dist = np.linalg.norm(diff, axis=3)
  blob = np.exp(-(dist / .03) ** 2)
  return np.stack((dist, blob), 1).astype(np.float32)


class CostToGo(nn.Module):

  def __init__(self, width=32, layers=6):
    super().__init__()
    net, channels = [], 3
    for _ in range(layers):
      net += [nn.Conv2d(channels, width, 5, padding=2), nn.LeakyReLU(.1)]
      channels = width
    net.append(nn.Conv2d(width, 1, 1))
    self.net = nn.Sequential(*net)

  # x = (map upsampled to the grid, straight distance, target blob); the net predicts the detour on top of
  # the straight line
  def forward(self, x):
    return x[:, 1] + self.net(x)[:, 0].clamp(min=0)


def model_inputs(maps, targets, centre, half):
  up = np.repeat(np.repeat(np.asarray(maps, np.float32), GRID // MAP_CELLS, 1), GRID // MAP_CELLS, 2)
  return np.concatenate((up[:, None] / .25, target_channels(targets, centre, half)), 1)


def load_cost_to_go(root, tag):
  ck = torch.load(model_folder(root, tag) / "models" / "cost_to_go.pt", map_location="cpu", weights_only=True)
  model = CostToGo()
  model.load_state_dict(ck["model"])
  return model.eval(), ck


# The field (GRID, GRID) for one obstacle map and target xy
@torch.inference_mode()
def cost_field(model, ck, obstacle_map, target):
  x = torch.as_tensor(model_inputs(np.asarray(obstacle_map)[None], np.asarray(target, np.float32)[None, :2], ck["centre"], ck["half"]))
  return model(x.to(next(model.parameters()).device))[0].cpu().numpy()


# Bilinear lookup of a field at points (n, 2), metres
def lookup(field, points, centre, half):
  g = (np.asarray(points, np.float32)[:, :2] - (np.asarray(centre) - np.asarray(half))) / (2 * np.asarray(half)) * GRID - .5
  g = np.clip(g, 0, GRID - 1 - 1e-4)
  i, j = g.astype(int), np.ceil(g).astype(int)
  f = g - i
  return ((1 - f[:, 0]) * (1 - f[:, 1]) * field[i[:, 0], i[:, 1]] + f[:, 0] * (1 - f[:, 1]) * field[j[:, 0], i[:, 1]]
          + (1 - f[:, 0]) * f[:, 1] * field[i[:, 0], j[:, 1]] + f[:, 0] * f[:, 1] * field[j[:, 0], j[:, 1]]).astype(np.float32)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True, type=pathlib.Path)
  ap.add_argument("--tag", required=True)
  ap.add_argument("--scenes", type=int, default=20000)
  ap.add_argument("--epochs", type=int, default=15)
  ap.add_argument("--batch-size", type=int, default=128)
  ap.add_argument("--lr", type=float, default=0.0003)
  ap.add_argument("--weight-decay", type=float, default=0.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--overwrite", action="store_true")
  args = ap.parse_args()
  torch.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)

  from environment.config import Config
  from environment.scene import sample_ends
  from environment.obstacles import sample_obstacles
  manifest = load_run(args.run)
  cfg = Config.nested(manifest["config"])
  table, centre, half = float(cfg.table.height), [float(v) for v in cfg.table.center], [float(v) for v in cfg.table.half_size]
  cache = args.run / "features" / f"cost_to_go_{args.scenes}_r{int(100 * RADIUS)}_h{int(100 * BLOCKING_HEIGHT)}.npz"
  if cache.exists():
    cached = np.load(cache)
    maps, targets, fields = cached["maps"], cached["targets"], cached["fields"]
  else:
    maps, targets = [], []
    while len(maps) < args.scenes:
      _, _, pick, place = sample_ends(cfg, rng)
      drawn = sample_obstacles(cfg, rng, pick, place)
      if drawn is None:
        continue
      obstacles = [o.to_dict() for o in drawn]
      m = rasterise(obstacles, table, centre, half)
      # both ends serve as targets: the goal of the carry and the cube of the reach
      maps += [m, m]
      targets += [place, pick]
    maps, targets = np.stack(maps), np.asarray(targets, np.float32)
    fields = np.concatenate([cost_fields(maps[i:i + 500], targets[i:i + 500], centre, half) for i in range(0, len(maps), 500)])
    np.savez(cache, maps=maps, targets=targets, fields=fields)
  print(f"cost_to_go: {len(maps)} synthetic (map, target) pairs, {int((maps > .01).any((1, 2)).sum())} with obstacles", flush=True)
  x = torch.from_numpy(model_inputs(maps, targets, centre, half))
  y = torch.from_numpy(fields)
  n_val = len(x) // 10
  ids = {"train": np.arange(n_val, len(x)), "val": np.arange(n_val)}

  # the loss counts the reachable points only (the capped interior of the blocked zones is not a path length)
  def loss_fn(model, batch):
    inputs, target = batch
    reachable = target < CAP
    err = ((model(inputs) - target).square() * reachable).sum() / reachable.sum().clamp(min=1)
    return err, {"field_mse": err.item()}

  model = CostToGo()
  folder = args.run / "attempts" / args.tag
  fit(folder, "cost_to_go", model, [x[ids["train"]], y[ids["train"]]], [x[ids["val"]], y[ids["val"]]], loss_fn,
      {"centre": centre, "half": half, "grid": GRID, "radius": RADIUS}, args)

  model, ck = load_cost_to_go(args.run, args.tag)
  with torch.inference_mode():
    pred = np.concatenate([model(x[ids["val"]][i:i + 256].to(next(model.parameters()).device)).cpu().numpy() for i in range(0, n_val, 256)])
  truth = fields[ids["val"]]
  free = truth < CAP
  detour = truth - x[ids["val"]][:, 1].numpy()
  report = {"n": int(n_val), "mae_m": float(np.abs(pred - truth)[free].mean()),
            "mae_m_where_detour_over_10cm": float(np.abs(pred - truth)[free & (detour > .1)].mean()) if (free & (detour > .1)).any() else None,
            "share_of_points_with_detour_over_10cm": float((free & (detour > .1)).mean())}
  write_json(folder / "reports" / "cost_to_go_validation.json", report)
  print(f"cost_to_go validation: {report}")


if __name__ == "__main__":
  main()
