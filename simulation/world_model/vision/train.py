"""Train Q on real clips, split D on short rollouts, or the no-vision outcome baseline."""
import argparse
import pathlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from world_model.train_dynamics import SplitDynamics
from .data import load_run, load_features, state_arrays, digest, write_json, outcome_metrics, provenance
from .audit import audit


def mlp(inputs, outputs, width=128):
  return nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.GELU(),
                       nn.Linear(width, width), nn.GELU(), nn.Linear(width, outputs))


def normalizer(a):
  return (torch.as_tensor(a.mean(0), dtype=torch.float32),
          torch.as_tensor(np.maximum(a.std(0), 1e-3), dtype=torch.float32))


def normalized(a, stats, name):
  return (torch.tensor(a, dtype=torch.float32)-stats[name+"_mean"])/stats[name+"_std"]


def sequence_input(p, actions, maximum):
  # Prefix actions + mask: no actual future robot values or object labels are inputs.
  padded = np.zeros((maximum, 5), np.float32)
  mask = np.zeros(maximum, np.float32)
  padded[:len(actions)], mask[:len(actions)] = actions, 1
  return np.concatenate((p, padded.ravel(), mask))


def build_model(ck):
  if ck["role"] == "dynamics":
    model = SplitDynamics(ck["z_dim"], 20, 5, width=512, p_width=128)
  else:
    model = mlp(ck["input_dim"], 4)
  model.load_state_dict(ck["model"])
  return model.eval()


def load_model(root, role, tag=""):
  root = pathlib.Path(root)
  output = root if not tag else root / "attempts" / tag
  ck = torch.load(output / "models" / f"{role}.pt", map_location="cpu", weights_only=True)
  if ck["role"] != role or ck["manifest_sha256"] != digest(root / "manifest.json"):
    raise ValueError(f"{role} checkpoint belongs to a different run")
  if ck["features_sha256"] != digest(root / "features/meta.json"):
    raise ValueError(f"{role} checkpoint uses a different encoder/cache")
  return build_model(ck), ck


@torch.inference_mode()
def readout(model, ck, z, p):
  inputs = normalized(p, ck, "p")
  if ck["role"] == "readout":
    inputs = torch.cat((normalized(z, ck, "z"), inputs), -1)
  output = model(inputs)
  xyz = output[:, :3]*ck["xyz_std"]+ck["xyz_mean"]
  return xyz.numpy(), output[:, 3].sigmoid().numpy()


@torch.inference_mode()
def imagine(model, ck, z, p, actions, robot_states=None):
  """Roll out z recursively; optional recorded p is an offline diagnostic only."""
  if robot_states is not None:
    robot_states = np.asarray(robot_states)
    if robot_states.shape != (len(actions)+1, len(p)) or not np.isfinite(robot_states).all():
      raise ValueError("recorded robot trajectory must contain one finite p per state")
    if not np.allclose(robot_states[0], p, rtol=0, atol=1e-6):
      raise ValueError("recorded robot trajectory does not start at the source p")
    robot_states = normalized(robot_states, ck, "p")
  z, p = normalized(z, ck, "z"), normalized(p, ck, "p")
  zs, ps = [z], [p]
  for t, action in enumerate(actions):
    action = torch.as_tensor(action, dtype=torch.float32)
    if robot_states is None:
      z, p = model(z, p, action)
    else:
      # D_z consumes BOTH current and next robot state: override both inputs.
      current_p, p = robot_states[t], robot_states[t+1]
      z = z + model.visual(torch.cat((z, current_p, action, p), dim=-1))
    zs.append(z)
    ps.append(p)
  return (torch.stack(zs).numpy()*ck["z_std"].numpy()+ck["z_mean"].numpy(),
          torch.stack(ps).numpy()*ck["p_std"].numpy()+ck["p_mean"].numpy())


@torch.inference_mode()
def baseline_predict(model, ck, p, actions):
  p = normalized(p, ck, "p").numpy()
  x = torch.as_tensor(sequence_input(p, actions, ck["max_horizon"])[None], dtype=torch.float32)
  y = model(x)
  return (y[:, :3]*ck["xyz_std"]+ck["xyz_mean"]).numpy(), y[:, 3].sigmoid().numpy()


def q_gate(metrics, limits):
  return (max(metrics["xyz_mae_cm"]) <= limits["xyz_cm"]
          and metrics["height_mae_cm"] <= limits["height_cm"]
          and metrics["grasp_f1"] >= limits["f1"]
          and metrics["grasp_brier"] <= limits["brier"]
          and min(metrics["positive_labels"], metrics["negative_labels"]) > 0)


def fit(root, role, model, train, val, loss_fn, meta, args):
  path = root / "models" / f"{role}.pt"
  if path.exists():
    raise ValueError(f"{path} already exists; use --tag for a new training attempt")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  print(f"{role}: {len(train[0])} training / {len(val[0])} validation examples; "
        f"device={device}; parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  loader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True)
  best, best_epoch, history = float("inf"), None, []
  for epoch in range(1, args.epochs+1):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
      batch = [x.to(device) for x in batch]
      loss, _ = loss_fn(model, batch)
      if not torch.isfinite(loss):
        raise RuntimeError(f"nonfinite {role} loss, epoch {epoch}; inspect inputs/scales")
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
      total += loss.item()*len(batch[0]); count += len(batch[0])
    model.eval()
    with torch.inference_mode():
      # Bounded batches allow larger datasets without allocating all validation tensors on GPU.
      v_total, v_count, parts = 0.0, 0, {}
      for offset in range(0, len(val[0]), args.batch_size):
        batch = [x[offset:offset+args.batch_size].to(device) for x in val]
        loss, terms = loss_fn(model, batch)
        n = len(batch[0]); v_total += loss.item()*n; v_count += n
        for name, value in terms.items():
          parts[name] = parts.get(name, 0.0)+float(value)*n
      value = v_total/v_count
      parts = {k: v/v_count for k, v in parts.items()}
    row = {"epoch": epoch, "train_loss": total/count, "val_loss": value, **parts}
    history.append(row)
    if not np.isfinite(value):
      raise RuntimeError(f"nonfinite {role} validation loss")
    if value < best:
      best = value
      best_epoch = epoch
      path.parent.mkdir(parents=True, exist_ok=True)
      torch.save({**meta, "role": role, "epoch": epoch, "val_loss": value,
                  "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, path)
    print(f"{role} epoch {epoch:03d} train={total/count:.5f} val={value:.5f} "
          + " ".join(f"{k}={v:.5f}" for k, v in parts.items()), flush=True)
  write_json(root / "reports" / f"{role}_training.json", {"device": device, "history": history,
                         "best_validation_loss": best, "best_epoch": best_epoch, "checkpoint": str(path),
                         "checkpoint_sha256": digest(path), "seed": args.seed,
                         "settings": meta["training_settings"], "provenance": meta["provenance"]})
  print(f"{role}: best checkpoint {path} selected using validation only", flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("stage", choices=("readout", "dynamics", "baseline"))
  p.add_argument("--run", required=True, type=pathlib.Path)
  p.add_argument("--epochs", type=int, default=60)
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--lr", type=float, default=0.001)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--rollout-steps", type=int, default=4)
  p.add_argument("--tag", default="", help="optional new model directory, e.g. retry_1; pass the same tag to tests")
  p.add_argument("--q-xyz-cm", type=float, default=2.0)
  p.add_argument("--q-height-cm", type=float, default=1.0)
  p.add_argument("--q-f1", type=float, default=0.8)
  p.add_argument("--q-brier", type=float, default=0.15)
  args = p.parse_args()
  if min(args.epochs, args.batch_size, args.rollout_steps) < 1 or args.lr <= 0:
    p.error("epochs, batch-size, rollout-steps and lr must be positive")
  if min(args.q_xyz_cm, args.q_height_cm) <= 0 or not 0 <= args.q_f1 <= 1 or not 0 <= args.q_brier <= 1:
    p.error("Q distance limits must be positive; F1 and Brier limits must be in [0,1]")
  if args.tag and (pathlib.Path(args.tag).name != args.tag or args.tag in (".", "..")):
    p.error("tag must be a single directory name")
  torch.manual_seed(args.seed)
  torch.set_num_threads(min(torch.get_num_threads(), 4))
  manifest = load_run(args.run)
  issues = audit(args.run, manifest, images=False)["problems"]
  if issues:
    p.error("dataset audit failed: " + "; ".join(issues[:5]))
  z, feature_meta = load_features(args.run, manifest)
  robot, xyz, grasp = state_arrays(manifest)
  splits = np.asarray([s["split"] for s in manifest["states"]])
  ids = {s: np.flatnonzero(splits == s) for s in ("train", "val")}
  if any(len(i) == 0 for i in ids.values()):
    p.error("training and validation states are required")
  stats = {}
  for name, array in (("z", z), ("p", robot), ("xyz", xyz)):
    stats[name+"_mean"], stats[name+"_std"] = normalizer(array[ids["train"]])
  nz, np_, nxyz = (normalized(a, stats, n) for n, a in (("z", z), ("p", robot), ("xyz", xyz)))
  ng = torch.from_numpy(grasp)
  meta = {**stats, "manifest_sha256": digest(args.run / "manifest.json"),
          "features_sha256": digest(args.run / "features/meta.json"),
          "encoder": feature_meta["model"], "z_dim": z.shape[1], "seed": args.seed,
          "training_settings": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()},
          "provenance": {**provenance(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
  output = args.run if not args.tag else args.run / "attempts" / args.tag
  # Checkpoints remain tied to original data even when saved under an attempt directory.
  def train_role(role, model, arrays, loss, extra=None):
    fit(output, role, model, arrays["train"], arrays["val"], loss, {**meta, **(extra or {})}, args)

  def outcome_loss(model, batch):
    x, target, g = batch
    y = model(x)
    pose = (y[:, :3]-target).square().mean()
    contact = nn.functional.binary_cross_entropy_with_logits(y[:, 3], g)
    return pose+contact, {"xyz_mse": pose.item(), "grasp_bce": contact.item()}

  if args.stage == "readout":
    limits = {"xyz_cm": args.q_xyz_cm, "height_cm": args.q_height_cm,
              "f1": args.q_f1, "brier": args.q_brier}
    reports = {}
    for role, x in (("readout", torch.cat((nz, np_), -1)), ("readout_p", np_)):
      arrays = {s: [a[i] for a in (x, nxyz, ng)] for s, i in ids.items()}
      train_role(role, mlp(x.shape[1], 4), arrays, outcome_loss,
                 {"input_dim": x.shape[1], "q_limits": limits})
      ck = torch.load(output / "models" / f"{role}.pt", weights_only=True)
      q = build_model(ck)
      v = ids["val"]
      predicted, prob = readout(q, ck, z[v], robot[v])
      reports[role] = outcome_metrics(predicted, prob, xyz[v], grasp[v])
      print(f"{role} real validation: {reports[role]}")
    reports["q_gate_passed"] = q_gate(reports["readout"], limits)
    reports["limits"] = limits
    write_json(output / "reports/readout_validation.json", reports)
    print(f"Q real-validation gate: {'PASS' if reports['q_gate_passed'] else 'FAIL'}; "
          "a failed gate makes imagined object metrics diagnostic only")
  elif args.stage == "dynamics":
    w = args.rollout_steps
    arrays = {}
    for split in ("train", "val"):
      windows, actions = [], []
      for r in manifest["rollouts"]:
        if r["split"] != split:
          continue
        for start in range(len(r["actions"])-w+1):
          windows.append(r["states"][start:start+w+1])
          actions.append(r["actions"][start:start+w])
      if not windows:
        p.error("rollout-steps is longer than collected sequences")
      windows = np.asarray(windows)
      arrays[split] = [nz[windows], np_[windows], torch.tensor(actions, dtype=torch.float32)]

    def dynamics_loss(model, batch):
      zz, pp, aa = batch
      current_z, current_p = zz[:, 0], pp[:, 0]
      z_loss, p_loss = 0, 0
      for t in range(w):
        current_z, current_p = model(current_z, current_p, aa[:, t])
        z_loss += (current_z-zz[:, t+1]).square().mean()/w
        p_loss += (current_p-pp[:, t+1]).square().mean()/w
      return z_loss+p_loss, {"z_mse": z_loss.item(), "p_mse": p_loss.item()}

    train_role("dynamics", SplitDynamics(z.shape[1], 20, 5), arrays, dynamics_loss,
               {"rollout_steps": w})
  else:
    maximum = max(len(r["actions"]) for r in manifest["rollouts"])
    arrays = {}
    for split in ("train", "val"):
      inputs, targets = [], []
      for r in manifest["rollouts"]:
        if r["split"] == split:
          for h in range(1, len(r["states"])):
            inputs.append(sequence_input(np_[r["states"][0]].numpy(), r["actions"][:h], maximum))
            targets.append(r["states"][h])
      arrays[split] = [torch.tensor(np.asarray(inputs), dtype=torch.float32), nxyz[targets], ng[targets]]
    dim = arrays["train"][0].shape[1]
    train_role("no_vision", mlp(dim, 4), arrays, outcome_loss,
               {"input_dim": dim, "max_horizon": maximum})
    print("No-vision baseline predicts outcomes directly from starting p and action sequence. "
          "It never sees z, object coordinates, or measured future p as inputs.")


if __name__ == "__main__":
  main()
