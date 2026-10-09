"""Train the first one-step action-conditioned latent and proprio dynamics MLP."""

import argparse
import hashlib
import json
import pathlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Dynamics(nn.Module):
  def __init__(self, z_dim, p_dim, a_dim, width=512):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(z_dim + p_dim + a_dim, width), nn.LayerNorm(width), nn.GELU(),
                             nn.Linear(width, width), nn.GELU(), nn.Linear(width, z_dim + p_dim))
    self.z_dim = z_dim

  def forward(self, z, p, action):
    delta = self.net(torch.cat((z, p, action), dim=-1))
    return z + delta[..., :self.z_dim], p + delta[..., self.z_dim:]


class SplitDynamics(nn.Module):
  """Predict robot motion separately, then predict the next V-JEPA visual vector."""

  def __init__(self, z_dim, p_dim, a_dim, width=512, p_width=128):
    super().__init__()
    self.robot = nn.Sequential(nn.Linear(p_dim + a_dim, p_width), nn.LayerNorm(p_width),
                               nn.GELU(), nn.Linear(p_width, p_width), nn.GELU(),
                               nn.Linear(p_width, p_dim))
    self.visual = nn.Sequential(nn.Linear(z_dim + 2 * p_dim + a_dim, width),
                                nn.LayerNorm(width), nn.GELU(),
                                nn.Linear(width, width), nn.GELU(),
                                nn.Linear(width, z_dim))

  def forward(self, z, p, action):
    next_p = p + self.robot(torch.cat((p, action), dim=-1))
    # In a rollout, p can itself be a robot-head prediction from the previous step.
    # Detach both robot contexts so visual loss cannot override the proprio targets.
    next_z = z + self.visual(torch.cat((z, p.detach(), action, next_p.detach()), dim=-1))
    return next_z, next_p


class SplitDynamicsZ(SplitDynamics):
  """Same two heads, but the robot head also sees the visual vector, so the finger width it
  predicts can depend on whether a cube is between the fingers (the plain robot head cannot know)."""

  def __init__(self, z_dim, p_dim, a_dim, width=512, p_width=128):
    super().__init__(z_dim, p_dim, a_dim, width=width, p_width=p_width)
    self.robot = nn.Sequential(nn.Linear(z_dim + p_dim + a_dim, p_width), nn.LayerNorm(p_width),
                               nn.GELU(), nn.Linear(p_width, p_width), nn.GELU(),
                               nn.Linear(p_width, p_dim))

  def forward(self, z, p, action):
    # z is detached here so the robot loss does not reshape the visual head
    next_p = p + self.robot(torch.cat((z.detach(), p, action), dim=-1))
    next_z = z + self.visual(torch.cat((z, p.detach(), action, next_p.detach()), dim=-1))
    return next_z, next_p


def model_from_checkpoint(checkpoint):
  """Load either dynamics architecture without changing old checkpoints."""
  architecture = checkpoint.get("architecture", "Dynamics")
  if architecture == "Dynamics":
    model = Dynamics(checkpoint["z_dim"], checkpoint["p_dim"],
                     checkpoint["a_dim"], width=checkpoint["width"])
  elif architecture == "SplitDynamics":
    model = SplitDynamics(checkpoint["z_dim"], checkpoint["p_dim"],
                          checkpoint["a_dim"], width=checkpoint["width"],
                          p_width=checkpoint["p_width"])
  elif architecture == "SplitDynamicsZ":
    model = SplitDynamicsZ(checkpoint["z_dim"], checkpoint["p_dim"],
                           checkpoint["a_dim"], width=checkpoint["width"],
                           p_width=checkpoint["p_width"])
  else:
    raise ValueError(f"unknown dynamics architecture: {architecture}")
  model.load_state_dict(checkpoint["model"])
  return model


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=pathlib.Path)
  parser.add_argument("--features", required=True, type=pathlib.Path)
  parser.add_argument("--out", required=True, type=pathlib.Path)
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--architecture", choices=("joint", "split"), default="joint",
                      help="split trains a separate robot-motion head and V-JEPA visual head")
  args = parser.parse_args()
  if args.epochs < 1 or args.batch_size < 1:
    parser.error("epochs and batch-size must be positive")
  torch.manual_seed(args.seed)
  manifest = json.loads(args.manifest.read_text())
  feature_meta = json.loads((args.features / "meta.json").read_text())
  if (feature_meta["manifest"] != str(args.manifest.resolve())
      or feature_meta["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest()
      or feature_meta["camera"] != manifest["camera"]
      or feature_meta["clip_frames"] != manifest["clip_frames"]):
    parser.error("feature cache was made from a different manifest or camera setting")
  latents = np.load(args.features / "latents.npy")
  index = feature_meta["index"]
  samples = manifest["samples"]
  z = np.stack([latents[index[f"{s['episode']}:{s['source']}"]] for s in samples])
  next_z = np.stack([latents[index[s.get('target_key', f"{s['episode']}:{s['target']}")]]
                     for s in samples])
  proprio = np.asarray([s["p"] for s in samples], dtype=np.float32)
  next_p = np.asarray([s["next_p"] for s in samples], dtype=np.float32)
  action = np.asarray([s["action"] for s in samples], dtype=np.float32)
  train_ids = np.asarray([i for i, s in enumerate(samples) if s["split"] == "train"])
  val_ids = np.asarray([i for i, s in enumerate(samples) if s["split"] == "val"])
  if not len(train_ids) or not len(val_ids):
    parser.error("manifest needs both train and validation transitions")
  z_mean = np.concatenate((z[train_ids], next_z[train_ids])).mean(axis=0)
  z_std = np.maximum(np.concatenate((z[train_ids], next_z[train_ids])).std(axis=0), 1e-3)
  p_mean = np.concatenate((proprio[train_ids], next_p[train_ids])).mean(axis=0)
  p_std = np.maximum(np.concatenate((proprio[train_ids], next_p[train_ids])).std(axis=0), 1e-3)
  arrays = ((z - z_mean) / z_std, (proprio - p_mean) / p_std, action,
            (next_z - z_mean) / z_std, (next_p - p_mean) / p_std)
  arrays = [torch.as_tensor(x, dtype=torch.float32) for x in arrays]
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model_type = Dynamics if args.architecture == "joint" else SplitDynamics
  model = model_type(z.shape[1], proprio.shape[1], action.shape[1]).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  train = TensorDataset(*(a[train_ids] for a in arrays))
  loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
  val = [a[val_ids].to(device) for a in arrays]
  branch_pairs = {}
  for i in train_ids:
    sample = samples[i]
    if sample.get("branch_action") in ("plus_x", "minus_x"):
      branch_pairs.setdefault((sample["episode"], sample["source"]), {})[
        sample["branch_action"]] = int(i)
  paired = [v for v in branch_pairs.values() if len(v) == 2]
  paired_plus = [v["plus_x"] for v in paired]
  paired_minus = [v["minus_x"] for v in paired]
  args.out.mkdir(parents=True, exist_ok=True)
  best = float("inf")

  def save_checkpoint(path, epoch, z_loss, p_loss):
    checkpoint = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "z_dim": z.shape[1],
                  "p_dim": proprio.shape[1], "a_dim": action.shape[1],
                  "z_mean": torch.from_numpy(z_mean.copy()),
                  "z_std": torch.from_numpy(z_std.copy()),
                  "p_mean": torch.from_numpy(p_mean.copy()),
                  "p_std": torch.from_numpy(p_std.copy()),
                  "architecture": "Dynamics" if args.architecture == "joint" else "SplitDynamics",
                  "width": 512,
                  "proprio_columns": manifest["proprio_columns"],
                  "action_columns": manifest["action_columns"],
                  "encoder_model": feature_meta["model"],
                  "pooling": feature_meta["pooling"], "camera": feature_meta["camera"],
                  "clip_frames": feature_meta["clip_frames"], "epoch": epoch,
                  "val_z_mse": z_loss, "val_p_mse": p_loss}
    if args.architecture == "split":
      checkpoint["p_width"] = 128
    torch.save(checkpoint, path)

  for epoch in range(1, args.epochs + 1):
    model.train()
    for z_t, p_t, a_t, z_next, p_next in loader:
      z_t, p_t, a_t, z_next, p_next = [x.to(device) for x in (z_t, p_t, a_t, z_next, p_next)]
      pred_z, pred_p = model(z_t, p_t, a_t)
      loss = (pred_z - z_next).square().mean() + (pred_p - p_next).square().mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    model.eval()
    with torch.inference_mode():
      vz, vp, va, vnz, vnp = val
      pz, pp = model(vz, vp, va)
      z_loss = (pz - vnz).square().mean().item()
      p_loss = (pp - vnp).square().mean().item()
      persistence_z = (vz - vnz).square().mean().item()
      persistence_p = (vp - vnp).square().mean().item()
      shuffled_z, shuffled_p = model(vz, vp, va.roll(1, 0))
      shuffled_loss = ((shuffled_z - vnz).square().mean()
                       + (shuffled_p - vnp).square().mean()).item()
      contrast = ""
      if paired:
        train_z, train_p, train_a = [a.to(device) for a in arrays[:3]]
        _, train_pred_p = model(train_z, train_p, train_a)
        predicted_effect = ((train_pred_p[paired_plus, 14] - train_pred_p[paired_minus, 14])
                            * float(p_std[14]) * 100).cpu().numpy()
        real_effect = (next_p[paired_plus, 14] - next_p[paired_minus, 14]) * 100
        contrast = (f" train x effect={predicted_effect.mean():+.3f} cm"
                    f" (real {real_effect.mean():+.3f},"
                    f" MAE {np.abs(predicted_effect - real_effect).mean():.3f})")
    if z_loss + p_loss < best:
      best = z_loss + p_loss
      save_checkpoint(args.out / "best.pt", epoch, z_loss, p_loss)
    if args.architecture == "split" and epoch == args.epochs:
      save_checkpoint(args.out / "last.pt", epoch, z_loss, p_loss)
    print(f"epoch {epoch:03d} val z={z_loss:.4f} p={p_loss:.4f} "
          f"persistence z={persistence_z:.4f} p={persistence_p:.4f} "
          f"shuffled-action total={shuffled_loss:.4f}{contrast}", flush=True)
  print(f"best checkpoint: {args.out / 'best.pt'} (normalized MSE sum {best:.4f})")
  print("Compare validation errors with persistence before calling this a useful world model.")


if __name__ == "__main__":
  main()
