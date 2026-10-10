import argparse
import pathlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import load_run, load_features, state_arrays, check_manifest, write_json, outcome_metrics

# Train the three small networks on a run's latents, one stage per call, same --tag for all three:
#   readout   Q(z, p) -> cube xyz and held logit (and readout_p, the same from p alone, a control for
#             what the image contributes)
#   reward    R(z, p) -> proximity penalty, collision logit, table-hit logit (the penalty head)
#   dynamics  D(z, p, a) -> next z, next p, trained on windows of --rollout-steps steps. --task-weight adds
#             a task-consistency term: the frozen Q (trained first, same tag) must still read the recorded
#             cube xyz and held label from the imagined (z, p), otherwise plain latent MSE lets D drop the
#             cube because it is 1 of 64 grid cells and the arm dominates the latent. --robot-sees-z lets the
#             robot head read z too, so the predicted finger width can depend on whether a cube is between
#             the fingers.
# Models land in <run>/attempts/<tag>/models/<stage>.pt with their normalisation statistics.
#   python -m world_model.train readout  --run data/combined_test1 --tag combined --epochs 30
#   python -m world_model.train reward   --run data/combined_test1 --tag combined --epochs 30
#   python -m world_model.train dynamics --run data/combined_test1 --tag combined --epochs 30 --rollout-steps 8 --task-weight 1.0 --robot-sees-z


def mlp(inputs, outputs, width=128):
  return nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.GELU(),
                       nn.Linear(width, width), nn.GELU(), nn.Linear(width, outputs))


# Latent dynamics in two heads: the robot head predicts the next robot state from (p, a), the visual head
# predicts the next latent from (z, p, a, next p). Both predict residuals. The robot inputs to the visual head
# are detached so the latent loss cannot bend the robot prediction.
class SplitDynamics(nn.Module):

  def __init__(self, z_dim, p_dim=20, a_dim=5, width=512, p_width=128):
    super().__init__()
    self.robot = nn.Sequential(nn.Linear(p_dim + a_dim, p_width), nn.LayerNorm(p_width), nn.GELU(),
                               nn.Linear(p_width, p_width), nn.GELU(), nn.Linear(p_width, p_dim))
    self.visual = nn.Sequential(nn.Linear(z_dim + 2 * p_dim + a_dim, width), nn.LayerNorm(width), nn.GELU(),
                                nn.Linear(width, width), nn.GELU(), nn.Linear(width, z_dim))

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


# Mean and std over the training states, the std floored so constant columns do not blow up
def normalizer(a):
  return (torch.as_tensor(a.mean(0), dtype=torch.float32),
          torch.as_tensor(np.maximum(a.std(0), 1e-3), dtype=torch.float32))


# Normalise with the statistics stored in a checkpoint, name is z, p or xyz
def normalized(a, stats, name):
  return (torch.as_tensor(np.asarray(a), dtype=torch.float32) - stats[name + "_mean"]) / stats[name + "_std"]


# Rebuild a network from its checkpoint dict
def build_model(ck):
  if ck["role"] == "dynamics":
    if ck.get("architecture", "SplitDynamics") == "SplitDynamicsZ":
      model = SplitDynamicsZ(ck["z_dim"])
    else:
      model = SplitDynamics(ck["z_dim"])
  else:
    model = mlp(ck["input_dim"], 4)
  model.load_state_dict(ck["model"])
  return model.eval()


# Load <run>/attempts/<tag>/models/<role>.pt, returns the network and its checkpoint dict (statistics)
def load_model(root, role, tag=""):
  root = pathlib.Path(root)
  folder = root / "attempts" / tag if tag else root
  ck = torch.load(folder / "models" / f"{role}.pt", map_location="cpu", weights_only=True)
  if ck["role"] != role:
    raise ValueError(f"{folder}/models/{role}.pt holds a {ck['role']} model")
  return build_model(ck), ck


# Q on raw (unnormalised) inputs: cube xyz in metres and held probability
@torch.inference_mode()
def readout(model, ck, z, p):
  inputs = normalized(p, ck, "p")
  if ck["role"] == "readout":
    inputs = torch.cat((normalized(z, ck, "z"), inputs), -1)
  output = model(inputs)
  xyz = output[:, :3] * ck["xyz_std"] + ck["xyz_mean"]
  return xyz.numpy(), output[:, 3].sigmoid().numpy()


# Roll D forward from one raw (z, p) through a list of actions, returns raw z and p for every step
@torch.inference_mode()
def imagine(model, ck, z, p, actions):
  z, p = normalized(z, ck, "z"), normalized(p, ck, "p")
  zs, ps = [z], [p]
  for action in actions:
    z, p = model(z, p, torch.as_tensor(action, dtype=torch.float32))
    zs.append(z)
    ps.append(p)
  return (torch.stack(zs).numpy() * ck["z_std"].numpy() + ck["z_mean"].numpy(),
          torch.stack(ps).numpy() * ck["p_std"].numpy() + ck["p_mean"].numpy())


# Train one network, keep the checkpoint with the lowest validation loss
def fit(folder, role, model, train, val, loss_fn, meta, args):
  path = folder / "models" / f"{role}.pt"
  if path.exists():
    raise FileExistsError(f"{path} exists; delete it or train with another --tag")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  print(f"{role}: {len(train[0])} training / {len(val[0])} validation examples on {device}, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  loader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True)
  best, history = float("inf"), []

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
    print(f"{role} epoch {epoch:03d} train={total / count:.5f} val={value:.5f} "
          + " ".join(f"{k}={v:.5f}" for k, v in parts.items()), flush=True)

  write_json(folder / "reports" / f"{role}_training.json",
             {"device": device, "history": history, "best_validation_loss": best,
              "training_examples": len(train[0]), "validation_examples": len(val[0]),
              "settings": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}})
  print(f"{role}: saved the best epoch to {path}", flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("stage", choices=("readout", "reward", "dynamics"))
  p.add_argument("--run", required=True, type=pathlib.Path)
  p.add_argument("--tag", default="", help="models go to <run>/attempts/<tag>/models")
  p.add_argument("--epochs", type=int, default=30)
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--lr", type=float, default=0.001)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--rollout-steps", type=int, default=8, help="dynamics: steps per training window")
  p.add_argument("--task-weight", type=float, default=1.0, help="dynamics: weight of the frozen Q term, 0 turns it off")
  p.add_argument("--robot-sees-z", action="store_true", help="dynamics: SplitDynamicsZ instead of SplitDynamics")
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

  # normalisation from the training states only, stored in every checkpoint
  stats = {}
  for name, array in (("z", z), ("p", robot), ("xyz", xyz)):
    stats[name + "_mean"], stats[name + "_std"] = normalizer(array[ids["train"]])
  nz, np_, nxyz = (normalized(a, stats, n) for n, a in (("z", z), ("p", robot), ("xyz", xyz)))
  nheld = torch.from_numpy(held)
  meta = {**stats, "encoder": feature_meta["model"], "z_dim": z.shape[1], "seed": args.seed}

  # what the controller needs besides the checkpoints (the environment config and the resting cube height),
  # so a clone that has only attempts/<tag>/ and features/{pca.npz,meta.json} can run without the manifest
  rest = [s["object_xyz"][2] for s in manifest["states"] if s["split"] == "train" and not s["held"]
          and abs(s["object_xyz"][2] - manifest["config"]["table"]["height"]) < .06]
  write_json(folder / "info.json", {"config": manifest["config"], "rest_z": float(np.median(rest)),
                                    "encoder": {k: v for k, v in feature_meta.items() if k != "keys"}})

  if args.stage == "readout":
    # Q from (z, p), and readout_p from p alone as the blind control
    def outcome_loss(model, batch):
      x, target, g = batch
      y = model(x)
      pose = (y[:, :3] - target).square().mean()
      contact = nn.functional.binary_cross_entropy_with_logits(y[:, 3], g)
      return pose + contact, {"xyz_mse": pose.item(), "held_bce": contact.item()}

    reports = {}
    for role, x in (("readout", torch.cat((nz, np_), -1)), ("readout_p", np_)):
      arrays = {s: [a[i] for a in (x, nxyz, nheld)] for s, i in ids.items()}
      fit(folder, role, mlp(x.shape[1], 4), arrays["train"], arrays["val"], outcome_loss,
          {**meta, "input_dim": x.shape[1]}, args)
      q, ck = load_model(args.run, role, args.tag)
      v = ids["val"]
      predicted, probability = readout(q, ck, z[v], robot[v])
      reports[role] = outcome_metrics(predicted, probability, xyz[v], held[v])
      print(f"{role} validation: {reports[role]}")
    write_json(folder / "reports/readout_validation.json", reports)

  elif args.stage == "reward":
    # targets are the recorded unweighted reward components: proximity (continuous), collision and table hit (0/1)
    comps = np.asarray([[s["reward_components"].get(k, 0.0) for k in ("proximity", "collision", "table_hit")]
                        for s in manifest["states"]], np.float32)
    targets = torch.from_numpy(np.c_[comps[:, 0], comps[:, 1] != 0, comps[:, 2] != 0].astype(np.float32))
    x = torch.cat((nz, np_), -1)

    def penalty_loss(model, batch):
      inputs, target = batch
      y = model(inputs)
      proximity = (y[:, 0] - target[:, 0]).square().mean()
      collision = nn.functional.binary_cross_entropy_with_logits(y[:, 1], target[:, 1])
      table = nn.functional.binary_cross_entropy_with_logits(y[:, 2], target[:, 2])
      return proximity + collision + table, {"proximity_mse": proximity.item(), "collision_bce": collision.item(),
                                             "table_bce": table.item()}

    fit(folder, "reward", mlp(x.shape[1], 4), [x[ids["train"]], targets[ids["train"]]],
        [x[ids["val"]], targets[ids["val"]]], penalty_loss, {**meta, "input_dim": x.shape[1]}, args)
    r, ck = load_model(args.run, "reward", args.tag)
    v = ids["val"]
    with torch.inference_mode():
      y = r(x[v]).numpy()
    t = targets[v].numpy()
    report = {"n": len(v), "proximity_mae": float(np.abs(y[:, 0] - t[:, 0]).mean()),
              "penalised_states": int((t[:, 0] != 0).sum())}
    for j, name in ((1, "collision"), (2, "table_hit")):
      guess, truth = y[:, j] > 0, t[:, j] > .5
      tp, fp, fn = int((guess & truth).sum()), int((guess & ~truth).sum()), int((~guess & truth).sum())
      report[name] = {"positives": int(truth.sum()), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
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
                       nxyz[windows], nheld[windows]]

    # the frozen Q that reads every imagined state
    q_frozen = None
    if args.task_weight > 0:
      q_frozen, q_ck = load_model(args.run, "readout", args.tag)
      for name in stats:
        if not torch.allclose(q_ck[name], stats[name]):
          raise ValueError("the readout was trained with other normalisation; train readout and dynamics with the same tag")
      q_frozen.requires_grad_(False).to("cuda" if torch.cuda.is_available() else "cpu")
      print(f"dynamics: task-consistency through the frozen Q, weight {args.task_weight}", flush=True)

    def dynamics_loss(model, batch):
      zz, pp, aa, xx, gg = batch
      current_z, current_p = zz[:, 0], pp[:, 0]
      z_loss = p_loss = task_loss = 0
      for t in range(w):
        current_z, current_p = model(current_z, current_p, aa[:, t])
        z_loss += (current_z - zz[:, t + 1]).square().mean() / w
        p_loss += (current_p - pp[:, t + 1]).square().mean() / w
        if q_frozen is not None:
          y = q_frozen(torch.cat((current_z, current_p), -1))
          task_loss += ((y[:, :3] - xx[:, t + 1]).square().mean()
                        + nn.functional.binary_cross_entropy_with_logits(y[:, 3], gg[:, t + 1])) / w
      terms = {"z_mse": z_loss.item(), "p_mse": p_loss.item()}
      if q_frozen is None:
        return z_loss + p_loss, terms
      return z_loss + p_loss + args.task_weight * task_loss, {**terms, "task": task_loss.item()}

    model = SplitDynamicsZ(z.shape[1]) if args.robot_sees_z else SplitDynamics(z.shape[1])
    fit(folder, "dynamics", model, arrays["train"], arrays["val"], dynamics_loss,
        {**meta, "rollout_steps": w, "task_weight": args.task_weight,
         "architecture": "SplitDynamicsZ" if args.robot_sees_z else "SplitDynamics"}, args)


if __name__ == "__main__":
  main()
