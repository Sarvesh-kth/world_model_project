import argparse
import pathlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .common import load_run, load_features, state_arrays, check_manifest, write_json, outcome_metrics

# Train the three networks on a run's latents, one stage per call, same --tag for all:
#   readout   Q(z, gripper) -> cube xyz and held, convolutions on the camera grids with a spatial softmax so
#             the position is read off the image. readout_p is the blind twin on the robot state alone.
#   reward    R(z, p) -> reach, grasp, lift, transport, placed and the penalties
#   dynamics  D(z, p, a) -> next z, next p on windows of --rollout-steps steps; --task-weight makes the frozen
#             Q still read the recorded cube from the imagined latent, otherwise D drops the cube
# The best validation epoch is kept in <run>/attempts/<tag>/models/<stage>.pt. Commands in world_model/README.md.


# What R predicts, in output order: regressions for the shaped terms and proximity, logits for the flags
REWARD_OUTPUTS = ("reach", "grasp", "lift", "transport", "placed", "proximity", "collision", "table_hit", "empty_grasp")
REWARD_FLAGS = ("placed", "collision", "table_hit", "empty_grasp")
# the columns of p the readout Q gets: finger width and gripper command, nothing about where the arm is
P_GRIPPER = (18, 19)
# what Q predicts, in output order: absolute cube xyz, cube xyz relative to the grasp site, held logit
Q_OUTPUTS = 7


def mlp(inputs, outputs, width=128, dropout=0.1):
  return nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
                       nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, outputs))


# Q and R over the latent's grids: blocks is the latent layout (block -> size), cells the side of each camera
# grid, p_dim the robot numbers after the latent. Each camera grid gets a 1x1 and two 3x3 convolutions, a
# spatial softmax on the static features gives key points. For Q the absolute position has its own head on
# the static grid and the main head (relative position, held) sees those features detached, otherwise the
# key points learned to follow the hand.
class GridReadout(nn.Module):

  def __init__(self, blocks, cells, p_dim, outputs, width=64, keys=8, static_outputs=0):
    super().__init__()
    self.blocks, self.cells, self.p_dim, self.static_outputs = dict(blocks), dict(cells), p_dim, static_outputs
    self.cameras = [c for c in self.blocks if c != "heightmap"]
    extra = 1 if "heightmap" in self.blocks else 0
    self.nets = nn.ModuleDict()
    for i, c in enumerate(self.cameras):
      depth = self.blocks[c] // (self.cells[c] ** 2) + (extra if i == 0 else 0)
      self.nets[c] = nn.Sequential(nn.Conv2d(depth, width, 1), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.GELU())
    self.keys = nn.Conv2d(width, keys, 1)
    n = self.cells[self.cameras[0]]
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    self.register_buffer("grid", torch.stack((xs, ys)).reshape(2, n * n))
    self.head = mlp(width * len(self.cameras) + 2 * keys + p_dim, outputs - static_outputs)
    self.static_head = mlp(width + 2 * keys, static_outputs) if static_outputs else None

  def forward(self, x):
    n = x.shape[0]
    parts, offset = {}, 0
    for name, size in self.blocks.items():
      parts[name] = x[:, offset:offset + size]
      offset += size
    feats = []
    for i, camera in enumerate(self.cameras):
      c = self.cells[camera]
      g = parts[camera].reshape(n, c, c, -1).permute(0, 3, 1, 2)
      if i == 0 and "heightmap" in parts:
        h = parts["heightmap"]
        side = int(round(h.shape[1] ** .5))
        h = h.reshape(n, 1, side, side)
        if side != c:
          h = nn.functional.interpolate(h, size=(c, c), mode="nearest")
        g = torch.cat((g, h), 1)
      f = self.nets[camera](g)
      feats.append(f.mean((2, 3)))
      if i == 0:
        attention = torch.softmax(self.keys(f).reshape(n, -1, c * c), -1)
        feats.append((attention @ self.grid.T).reshape(n, -1))
    if self.static_head is None:
      return self.head(torch.cat(feats + [x[:, offset:]], -1))
    y = self.head(torch.cat([f.detach() for f in feats[:2]] + feats[2:] + [x[:, offset:]], -1))
    return torch.cat((self.static_head(torch.cat(feats[:2], -1)), y), -1)


# Latent dynamics in two heads: the robot head predicts the next robot state from (p, a), the visual head
# predicts the next latent from (z, p, a, next p). Both predict residuals. The robot inputs to the visual head
# are detached so the latent loss cannot bend the robot prediction.
class SplitDynamics(nn.Module):

  def __init__(self, z_dim, p_dim=20, a_dim=5, width=512, p_width=128):
    super().__init__()
    self.robot = nn.Sequential(nn.Linear(p_dim + a_dim, p_width), nn.LayerNorm(p_width), nn.GELU(),
                               nn.Linear(p_width, p_width), nn.GELU(), nn.Linear(p_width, p_dim))
    self.visual = nn.Sequential(nn.Linear(z_dim + 2 * p_dim + a_dim, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(width, width), nn.GELU(), nn.Dropout(0.1), nn.Linear(width, z_dim))

  def forward(self, z, p, action):
    next_p = p + self.robot(torch.cat((p, action), dim=-1))
    next_z = z + self.visual(torch.cat((z, p.detach(), action, next_p.detach()), dim=-1))
    return next_z, next_p


# Same two heads, but the robot head also sees z (detached), so the finger width it predicts can depend on
# whether a cube is between the fingers, which (p, a) alone cannot tell
class SplitDynamicsZ(SplitDynamics):

  def __init__(self, z_dim, p_dim=20, a_dim=5, width=512, p_width=128):
    super().__init__(z_dim, p_dim, a_dim, width, p_width)
    self.robot = nn.Sequential(nn.Linear(z_dim + p_dim + a_dim, p_width), nn.LayerNorm(p_width), nn.GELU(),
                               nn.Linear(p_width, p_width), nn.GELU(), nn.Linear(p_width, p_dim))

  def forward(self, z, p, action):
    next_p = p + self.robot(torch.cat((z.detach(), p, action), dim=-1))
    next_z = z + self.visual(torch.cat((z, p.detach(), action, next_p.detach()), dim=-1))
    return next_z, next_p


# Same two heads, but the robot head reads the frozen Q's reading (absolute cube position and held) instead
# of the whole latent, which made it lose the hand. Q is a frozen submodule saved with the checkpoint.
class SplitDynamicsQ(SplitDynamics):

  def __init__(self, z_dim, q, q_ck, p_dim=20, a_dim=5, width=512, p_width=128):
    super().__init__(z_dim, p_dim, a_dim, width, p_width)
    self.robot = nn.Sequential(nn.Linear(4 + p_dim + a_dim, p_width), nn.LayerNorm(p_width), nn.GELU(),
                               nn.Linear(p_width, p_width), nn.GELU(), nn.Linear(p_width, p_dim))
    self.q = q.requires_grad_(False)
    self.q_columns = list(q_ck["p_columns"])

  # Q's absolute position head and held probability on the normalised (z, p), no gradient into Q
  def reading(self, z, p):
    with torch.no_grad():
      y = self.q(torch.cat((z, p[:, self.q_columns]), -1))
    return torch.cat((y[:, :3], y[:, 6:7].sigmoid()), -1)

  def forward(self, z, p, action):
    next_p = p + self.robot(torch.cat((self.reading(z.detach(), p.detach()), p, action), dim=-1))
    next_z = z + self.visual(torch.cat((z, p.detach(), action, next_p.detach()), dim=-1))
    return next_z, next_p


# Mean and std over the training states, the std floored so constant columns do not blow up
def normalizer(a, floor=1e-3):
  return (torch.as_tensor(a.mean(0), dtype=torch.float32),
          torch.as_tensor(np.maximum(a.std(0), floor), dtype=torch.float32))


# Normalise with the statistics stored in a checkpoint, name is z, p or xyz
def normalized(a, stats, name):
  return (torch.as_tensor(np.asarray(a), dtype=torch.float32) - stats[name + "_mean"]) / stats[name + "_std"]


# Rebuild a network from its checkpoint dict
def build_model(ck):
  if ck["role"] == "dynamics":
    if ck.get("architecture", "SplitDynamics") == "SplitDynamicsZ":
      model = SplitDynamicsZ(ck["z_dim"])
    elif ck.get("architecture") == "SplitDynamicsQ":
      q_ck = ck["readout_ck"]
      q = GridReadout(q_ck["blocks"], q_ck["cells"], q_ck["p_dim"], q_ck["outputs"], static_outputs=q_ck.get("static_outputs", 0))
      model = SplitDynamicsQ(ck["z_dim"], q, q_ck)
    else:
      model = SplitDynamics(ck["z_dim"])
  elif ck.get("architecture") == "GridReadout":
    model = GridReadout(ck["blocks"], ck["cells"], ck["p_dim"], ck["outputs"], static_outputs=ck.get("static_outputs", 0))
  else:
    model = mlp(ck["input_dim"], ck["outputs"])
  model.load_state_dict(ck["model"])
  return model.eval()


# Where a tag's models, info.json and reports live: <run>/attempts/<tag>/, or the folder itself without a tag
# (a packaged model folder such as models/world_model, which holds models/, info.json and features/ side by side)
def model_folder(root, tag=""):
  root = pathlib.Path(root)
  return root / "attempts" / tag if tag else root


# Load <folder>/models/<role>.pt, returns the network and its checkpoint dict (statistics)
def load_model(root, role, tag=""):
  folder = model_folder(root, tag)
  ck = torch.load(folder / "models" / f"{role}.pt", map_location="cpu", weights_only=True)
  if ck["role"] != role:
    raise ValueError(f"{folder}/models/{role}.pt holds a {ck['role']} model")
  return build_model(ck), ck


# The device a model sits on
def device_of(model):
  return next(model.parameters()).device


# The input of Q or R: the normalised latent followed by the columns of p the checkpoint was trained with
def readout_inputs(ck, z, p):
  inputs = normalized(p, ck, "p")[:, list(ck["p_columns"])]
  if ck["role"] != "readout_p":
    inputs = torch.cat((normalized(z, ck, "z"), inputs), -1)
  return inputs


# Q's loss on a batch of raw outputs: the absolute head on states where the cube is not held, the relative
# head where it is (both normalised targets), BCE on the held logit
def readout_loss(y, absolute, relative, held):
  h = held >= .5
  pose = (torch.where(h[:, None], y[:, 3:6] - relative, y[:, :3] - absolute)).square().mean()
  contact = nn.functional.binary_cross_entropy_with_logits(y[:, 6], held)
  return pose + contact, {"xyz_mse": pose.item(), "held_bce": contact.item()}


# Q on raw (unnormalised) inputs: cube xyz in metres and held probability. The position is the absolute head,
# or the relative head plus the grasp site when the cube is read as held
@torch.inference_mode()
def readout(model, ck, z, p):
  output = model(readout_inputs(ck, z, p).to(device_of(model))).cpu()
  absolute = output[:, :3] * ck["xyz_std"] + ck["xyz_mean"]
  relative = output[:, 3:6] * ck["rel_std"] + ck["rel_mean"] + torch.as_tensor(np.asarray(p)[:, 14:17], dtype=torch.float32)
  held = output[:, 6].sigmoid()
  xyz = torch.where((held >= .5)[:, None], relative, absolute)
  return xyz.numpy(), held.numpy()


# R on raw (unnormalised) inputs: a dict name -> array over the batch, flags as probabilities, the shaped
# terms clipped to [0, 1] and proximity to [-1, 0]
@torch.inference_mode()
def reward_terms(model, ck, z, p):
  y = model(readout_inputs(ck, z, p).to(device_of(model))).cpu().numpy()
  out = {}
  for j, name in enumerate(ck["output_names"]):
    if name in REWARD_FLAGS:
      out[name] = 1 / (1 + np.exp(-y[:, j]))
    elif name == "proximity":
      out[name] = np.clip(y[:, j], -1, 0)
    else:
      out[name] = np.clip(y[:, j], 0, 1)
  return out


# Roll D forward from one raw (z, p) through a list of actions, returns raw z and p for every step
@torch.inference_mode()
def imagine(model, ck, z, p, actions):
  z, p = normalized(z, ck, "z"), normalized(p, ck, "p")
  # the model works on batches, a single state gets a batch dimension of one
  single = z.ndim == 1
  if single:
    z, p = z[None], p[None]
  zs, ps = [z], [p]
  for action in actions:
    a = torch.as_tensor(action, dtype=torch.float32)
    z, p = model(z, p, a[None] if single else a)
    zs.append(z)
    ps.append(p)
  zs, ps = torch.stack(zs), torch.stack(ps)
  if single:
    zs, ps = zs[:, 0], ps[:, 0]
  return zs.numpy() * ck["z_std"].numpy() + ck["z_mean"].numpy(), ps.numpy() * ck["p_std"].numpy() + ck["p_mean"].numpy()


# Train one network, keep the checkpoint with the lowest validation loss
def fit(folder, role, model, train, val, loss_fn, meta, args):
  path = folder / "models" / f"{role}.pt"
  if path.exists():
    if not args.overwrite:
      raise FileExistsError(f"{path} exists; delete it, train with another --tag or add --overwrite")
    path.unlink()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  print(f"{role}: {len(train[0])} training / {len(val[0])} validation examples on {device}, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  loader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True)
  # cosine decay to 1 % of the learning rate over the run, stepped per batch
  schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(loader), eta_min=args.lr * .01)
  best, history = float("inf"), []
  bar = tqdm(total=args.epochs, desc=role, unit="epoch")

  for epoch in range(1, args.epochs + 1):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
      batch = [x.to(device) for x in batch]
      loss, _ = loss_fn(model, batch)
      if not torch.isfinite(loss):
        raise RuntimeError(f"nonfinite {role} loss in epoch {epoch}")
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
      schedule.step()
      total += loss.item() * len(batch[0])
      count += len(batch[0])

    # validation in batches, the whole set does not fit on the GPU for the dynamics windows
    model.eval()
    with torch.inference_mode():
      v_total, v_count, parts = 0.0, 0, {}
      for offset in range(0, len(val[0]), args.batch_size):
        batch = [x[offset:offset + args.batch_size].to(device) for x in val]
        loss, terms = loss_fn(model, batch)
        n = len(batch[0])
        v_total += loss.item() * n
        v_count += n
        for name, value in terms.items():
          parts[name] = parts.get(name, 0.0) + float(value) * n
    value = v_total / v_count
    parts = {k: v / v_count for k, v in parts.items()}
    history.append({"epoch": epoch, "train_loss": total / count, "val_loss": value, **parts})

    if value < best:
      best = value
      path.parent.mkdir(parents=True, exist_ok=True)
      torch.save({**meta, "role": role, "epoch": epoch, "val_loss": value,
                  "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, path)
    bar.write(f"{role} epoch {epoch:03d} train={total / count:.5f} val={value:.5f} "
              + " ".join(f"{k}={v:.5f}" for k, v in parts.items()))
    bar.set_postfix(train=f"{total / count:.4f}", val=f"{value:.4f}", best=f"{best:.4f}")
    bar.update(1)

  bar.close()
  write_json(folder / "reports" / f"{role}_training.json",
             {"device": device, "history": history, "best_validation_loss": best,
              "training_examples": len(train[0]), "validation_examples": len(val[0]),
              "settings": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}})
  print(f"{role}: saved the best epoch ({min(history, key=lambda h: h['val_loss'])['epoch']}) to {path}", flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("stage", choices=("readout", "reward", "dynamics"))
  p.add_argument("--run", required=True, type=pathlib.Path)
  p.add_argument("--tag", default="", help="models go to <run>/attempts/<tag>/models")
  p.add_argument("--epochs", type=int, default=30)
  p.add_argument("--batch-size", type=int, default=256)
  p.add_argument("--lr", type=float, default=0.001)
  p.add_argument("--weight-decay", type=float, default=0.05)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--rollout-steps", type=int, default=8, help="dynamics: steps per training window")
  p.add_argument("--task-weight", type=float, default=1.0, help="dynamics: weight of the frozen Q term, 0 turns it off")
  p.add_argument("--robot-sees-z", action="store_true", help="dynamics: SplitDynamicsZ, the robot head reads the whole latent")
  p.add_argument("--robot-sees-q", action="store_true", help="dynamics: SplitDynamicsQ, the robot head reads the frozen Q's cube and held reading (needs a GridReadout Q of the same tag)")
  p.add_argument("--mlp-readout", action="store_true", help="readout / reward: the old MLP on (z, p) instead of GridReadout; see the note at the top")
  p.add_argument("--z-noise", type=float, default=0.3, help="readout / reward: Gaussian noise on the normalised latent while training, against scene memorisation")
  p.add_argument("--overwrite", action="store_true", help="replace an existing model of this tag (after the data grew)")
  args = p.parse_args()
  if min(args.epochs, args.batch_size, args.rollout_steps) < 1 or args.lr <= 0:
    p.error("epochs, batch-size, rollout-steps and lr must be positive")
  torch.manual_seed(args.seed)
  torch.set_num_threads(min(torch.get_num_threads(), 4))

  # data: latents, robot states, cube positions and held labels, split by the manifest's train / val
  manifest = load_run(args.run)
  problems = check_manifest(manifest)
  if problems:
    p.error("manifest problems: " + "; ".join(problems[:5]))
  z, feature_meta = load_features(args.run, manifest)
  robot, xyz, held = state_arrays(manifest)
  splits = np.asarray([s["split"] for s in manifest["states"]])
  ids = {s: np.flatnonzero(splits == s) for s in ("train", "val")}
  folder = args.run / "attempts" / args.tag if args.tag else args.run

  # normalisation from the training states only, stored in every checkpoint: "xyz" is the absolute cube
  # position, "rel" the cube relative to the grasp site (the relative statistics from the held states, where
  # that head is used)
  relative = xyz - robot[:, 14:17]
  stats = {}
  for name, array, rows in (("z", z, ids["train"]), ("p", robot, ids["train"]), ("xyz", xyz, ids["train"]),
                            ("rel", relative, ids["train"][held[ids["train"]] >= .5])):
    stats[name + "_mean"], stats[name + "_std"] = normalizer(array[rows] if len(rows) else array[ids["train"]])
  # a height map cell nothing ever stood on in training has std ~0 and would blow up a test-time obstacle,
  # so those cells get a floor
  blocks = feature_meta.get("layout", {})
  if "heightmap" in blocks:
    n = blocks["heightmap"]
    stats["z_std"][-n:] = torch.maximum(stats["z_std"][-n:], torch.tensor(0.1))
  nz, np_, nxyz, nrel = (normalized(a, stats, n) for n, a in (("z", z), ("p", robot), ("xyz", xyz), ("rel", relative)))
  nheld = torch.from_numpy(held)
  meta = {**stats, "encoder": feature_meta["model"], "z_dim": z.shape[1], "seed": args.seed}

  # Q and R train on a noisy latent: a different draw every batch, nothing at validation / inference
  def noisy(x, model):
    if model.training and args.z_noise > 0:
      x = x.clone()
      x[:, :z.shape[1]] += args.z_noise * torch.randn_like(x[:, :z.shape[1]])
    return x

  # Q and R: the grid network when the latent has grids (Q with the gripper columns of p only, R with all of
  # p), otherwise (or with --mlp-readout) the MLP on (z, p)
  cells = feature_meta.get("cells") or {c: 8 for c in blocks if c != "heightmap"}
  def readout_network(outputs, p_columns, static_outputs=0):
    if args.mlp_readout or not blocks or sum(blocks.values()) != z.shape[1]:
      p_columns = tuple(range(robot.shape[1]))
      return mlp(z.shape[1] + len(p_columns), outputs), {"input_dim": z.shape[1] + len(p_columns), "outputs": outputs, "p_columns": p_columns}
    return (GridReadout(blocks, cells, len(p_columns), outputs, static_outputs=static_outputs),
            {"architecture": "GridReadout", "blocks": blocks, "cells": cells, "p_dim": len(p_columns), "outputs": outputs,
             "p_columns": tuple(p_columns), "static_outputs": static_outputs})

  # what the controller needs besides the checkpoints (the environment config and the resting cube height),
  # so a clone that has only attempts/<tag>/ and features/{pca.npz,meta.json} can run without the manifest
  rest = [s["object_xyz"][2] for s in manifest["states"] if s["split"] == "train" and not s["held"]
          and abs(s["object_xyz"][2] - manifest["config"]["table"]["height"]) < .06]
  write_json(folder / "info.json", {"config": manifest["config"], "rest_z": float(np.median(rest)),
                                    "encoder": {k: v for k, v in feature_meta.items() if k != "keys"}})

  if args.stage == "readout":
    # Q from (z, gripper), and readout_p from the whole p alone as the blind control
    def outcome_loss(model, batch):
      x, absolute, rel, g = batch
      return readout_loss(model(noisy(x, model) if x.shape[1] > z.shape[1] else x), absolute, rel, g)

    reports = {}
    q, extra = readout_network(Q_OUTPUTS, P_GRIPPER, static_outputs=3)
    for role, x, model, ck in (("readout", torch.cat((nz, np_[:, list(extra["p_columns"])]), -1), q, extra),
                               ("readout_p", np_, mlp(np_.shape[1], Q_OUTPUTS),
                                {"input_dim": np_.shape[1], "outputs": Q_OUTPUTS, "p_columns": tuple(range(np_.shape[1]))})):
      arrays = {s: [a[i] for a in (x, nxyz, nrel, nheld)] for s, i in ids.items()}
      fit(folder, role, model, arrays["train"], arrays["val"], outcome_loss, {**meta, **ck}, args)
      q, ck = load_model(args.run, role, args.tag)
      v = ids["val"]
      predicted, probability = readout(q, ck, z[v], robot[v])
      reports[role] = outcome_metrics(predicted, probability, xyz[v], held[v])
      print(f"{role} validation: {reports[role]}")
    write_json(folder / "reports/readout_validation.json", reports)

  elif args.stage == "reward":
    # targets are the recorded unweighted reward components, regression for the continuous ones and BCE
    # for the flags
    comps = {k: np.asarray([s["reward_components"].get(k, 0.0) for s in manifest["states"]], np.float32) for k in REWARD_OUTPUTS if k != "placed"}
    comps["placed"] = np.asarray([s["reward_stage"] == "done" for s in manifest["states"]], np.float32)
    # runs recorded before the empty_grasp penalty existed get it from p, the environment's own definition
    missing = np.asarray(["empty_grasp" not in s["reward_components"] for s in manifest["states"]])
    comps["empty_grasp"][missing] = -((robot[missing, 19] < 0) & (robot[missing, 18] < .03)).astype(np.float32)
    columns = [comps[k] != 0 if k in REWARD_FLAGS else comps[k] for k in REWARD_OUTPUTS]
    targets = torch.from_numpy(np.stack(columns, 1).astype(np.float32))
    flags = [j for j, k in enumerate(REWARD_OUTPUTS) if k in REWARD_FLAGS]
    values = [j for j, k in enumerate(REWARD_OUTPUTS) if k not in REWARD_FLAGS]
    model, extra = readout_network(len(REWARD_OUTPUTS), tuple(range(robot.shape[1])))
    x = torch.cat((nz, np_[:, list(extra["p_columns"])]), -1)

    def reward_loss(model, batch):
      inputs, target = batch
      y = model(noisy(inputs, model))
      regression = (y[:, values] - target[:, values]).square().mean()
      classification = nn.functional.binary_cross_entropy_with_logits(y[:, flags], target[:, flags])
      return regression + classification, {"terms_mse": regression.item(), "flags_bce": classification.item()}

    fit(folder, "reward", model, [x[ids["train"]], targets[ids["train"]]],
        [x[ids["val"]], targets[ids["val"]]], reward_loss,
        {**meta, "output_names": list(REWARD_OUTPUTS), **extra}, args)
    r, ck = load_model(args.run, "reward", args.tag)
    v = ids["val"]
    terms = reward_terms(r, ck, z[v], robot[v])
    t = targets[v].numpy()
    report = {"n": len(v), "penalised_states": int((t[:, REWARD_OUTPUTS.index("proximity")] != 0).sum())}
    for j, name in enumerate(REWARD_OUTPUTS):
      if name in REWARD_FLAGS:
        guess, truth = terms[name] > .5, t[:, j] > .5
        tp, fp, fn = int((guess & truth).sum()), int((guess & ~truth).sum()), int((~guess & truth).sum())
        report[name] = {"positives": int(truth.sum()), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
      else:
        report[f"{name}_mae"] = float(np.abs(terms[name] - t[:, j]).mean())
    write_json(folder / "reports/reward_validation.json", report)
    print(f"reward validation: {report}")

  else:
    # every window of rollout_steps consecutive actions inside a rollout is one training example
    w = args.rollout_steps
    arrays = {}
    for split in ("train", "val"):
      windows, actions = [], []
      for r in manifest["rollouts"]:
        if r["split"] != split:
          continue
        for start in range(len(r["actions"]) - w + 1):
          windows.append(r["states"][start:start + w + 1])
          actions.append(r["actions"][start:start + w])
      if not windows:
        p.error("rollout-steps is longer than the recorded sequences")
      windows = np.asarray(windows)
      arrays[split] = [nz[windows], np_[windows], torch.tensor(actions, dtype=torch.float32),
                       nxyz[windows], nrel[windows], nheld[windows]]

    # the frozen Q that reads every imagined state
    q_frozen = None
    if args.task_weight > 0:
      q_frozen, q_ck = load_model(args.run, "readout", args.tag)
      q_columns = list(q_ck["p_columns"])
      for name in stats:
        if not torch.allclose(q_ck[name], stats[name]):
          raise ValueError("the readout was trained with other normalisation; train readout and dynamics with the same tag")
      q_frozen.requires_grad_(False).to("cuda" if torch.cuda.is_available() else "cpu")
      print(f"dynamics: task-consistency through the frozen Q, weight {args.task_weight}", flush=True)

    def dynamics_loss(model, batch):
      zz, pp, aa, xx, rr, gg = batch
      current_z, current_p = zz[:, 0], pp[:, 0]
      z_loss = p_loss = task_loss = 0
      for t in range(w):
        current_z, current_p = model(current_z, current_p, aa[:, t])
        z_loss += (current_z - zz[:, t + 1]).square().mean() / w
        p_loss += (current_p - pp[:, t + 1]).square().mean() / w
        if q_frozen is not None:
          y = q_frozen(torch.cat((current_z, current_p[:, q_columns]), -1))
          task_loss += readout_loss(y, xx[:, t + 1], rr[:, t + 1], gg[:, t + 1])[0] / w
      terms = {"z_mse": z_loss.item(), "p_mse": p_loss.item()}
      if q_frozen is None:
        return z_loss + p_loss, terms
      return z_loss + p_loss + args.task_weight * task_loss, {**terms, "task": task_loss.item()}

    extra = {}
    if args.robot_sees_q:
      q_model, q_ck = load_model(args.run, "readout", args.tag)
      if q_ck.get("architecture") != "GridReadout":
        raise ValueError("--robot-sees-q needs a GridReadout readout of the same tag")
      model = SplitDynamicsQ(z.shape[1], q_model, q_ck)
      extra = {"architecture": "SplitDynamicsQ", "readout_ck": {k: q_ck[k] for k in ("blocks", "cells", "p_dim", "outputs", "p_columns", "static_outputs")}}
    else:
      model = SplitDynamicsZ(z.shape[1]) if args.robot_sees_z else SplitDynamics(z.shape[1])
      extra = {"architecture": "SplitDynamicsZ" if args.robot_sees_z else "SplitDynamics"}
    fit(folder, "dynamics", model, arrays["train"], arrays["val"], dynamics_loss, {**meta, "rollout_steps": w, "task_weight": args.task_weight, **extra}, args)


if __name__ == "__main__":
  main()
