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


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=pathlib.Path)
  parser.add_argument("--features", required=True, type=pathlib.Path)
  parser.add_argument("--out", required=True, type=pathlib.Path)
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--seed", type=int, default=0)
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
  model = Dynamics(z.shape[1], proprio.shape[1], action.shape[1]).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  train = TensorDataset(*(a[train_ids] for a in arrays))
  loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
  val = [a[val_ids].to(device) for a in arrays]
  args.out.mkdir(parents=True, exist_ok=True)
  best = float("inf")
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
    if z_loss + p_loss < best:
      best = z_loss + p_loss
      torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "z_dim": z.shape[1],
                  "p_dim": proprio.shape[1], "a_dim": action.shape[1],
                  "z_mean": torch.from_numpy(z_mean.copy()),
                  "z_std": torch.from_numpy(z_std.copy()),
                  "p_mean": torch.from_numpy(p_mean.copy()),
                  "p_std": torch.from_numpy(p_std.copy()),
                  "architecture": "Dynamics", "width": 512,
                  "proprio_columns": manifest["proprio_columns"],
                  "action_columns": manifest["action_columns"],
                  "encoder_model": feature_meta["model"],
                  "pooling": feature_meta["pooling"], "camera": feature_meta["camera"],
                  "clip_frames": feature_meta["clip_frames"], "epoch": epoch,
                  "val_z_mse": z_loss, "val_p_mse": p_loss}, args.out / "best.pt")
    print(f"epoch {epoch:03d} val z={z_loss:.4f} p={p_loss:.4f} "
          f"persistence z={persistence_z:.4f} p={persistence_p:.4f} "
          f"shuffled-action total={shuffled_loss:.4f}", flush=True)
  print(f"best checkpoint: {args.out / 'best.pt'} (normalized MSE sum {best:.4f})")
  print("Compare validation errors with persistence before calling this a useful world model.")


if __name__ == "__main__":
  main()
