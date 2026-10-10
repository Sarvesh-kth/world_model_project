import argparse
import json
import pathlib

import numpy as np
import torch

from .common import load_run, load_features, state_arrays, write_json
from .train import load_model, imagine, normalized

# Can the penalty head R see obstacles? Scored on the held-out split twice: on the REAL latents, and on
# latents IMAGINED by D over 1 / 4 / 8 steps from the recorded actions, which is what the planner sees.
#   python -m world_model.eval_reward --run data/combined_test1 --tag combined --split test


# R's three outputs on raw (z, p): proximity in [-1, 0], collision and table-hit probabilities
def scores(r, rc, z, p):
  with torch.inference_mode():
    y = r(torch.cat((normalized(z, rc, "z"), normalized(p, rc, "p")), -1)).numpy()
  return np.clip(y[:, 0], -1, 0), 1 / (1 + np.exp(-y[:, 1])), 1 / (1 + np.exp(-y[:, 2]))


def metrics(prox, col, tab, truth):
  out = {"n": len(prox), "penalised_states": int((truth[:, 0] != 0).sum()), "contact_states": int((truth[:, 1] != 0).sum()),
         "proximity_mae": float(np.abs(prox - truth[:, 0]).mean())}
  near = truth[:, 0] != 0
  if near.any():
    out["proximity_mae_when_penalised"] = float(np.abs(prox[near] - truth[near, 0]).mean())
  for name, prob, column in (("collision", col, 1), ("table_hit", tab, 2)):
    t = truth[:, column] != 0
    g = prob > .5
    tp, fp, fn = int((g & t).sum()), int((g & ~t).sum()), int((~g & t).sum())
    # AUC without a threshold: how often a contact state is ranked above a clear one
    auc = float((prob[t][:, None] > prob[~t][None, :]).mean()) if t.any() and (~t).any() else None
    out[name] = {"positives": int(t.sum()), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "auc": auc}
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", type=pathlib.Path, default=pathlib.Path("data/combined_test1"))
  p.add_argument("--tag", default="combined")
  p.add_argument("--split", default="test")
  args = p.parse_args()
  manifest = load_run(args.run)
  z, _ = load_features(args.run, manifest)
  robot, _, _ = state_arrays(manifest)
  truth = np.asarray([[s["reward_components"].get(k, 0.0) for k in ("proximity", "collision", "table_hit")]
                      for s in manifest["states"]], np.float32)
  r, rc = load_model(args.run, "reward", args.tag)
  d, dc = load_model(args.run, "dynamics", args.tag)
  report = {"split": args.split, "tag": args.tag}

  # real latents, empty scenes and obstacle scenes separately
  for view in ("empty", "obstacles"):
    ids = np.asarray([i for i, s in enumerate(manifest["states"]) if s["split"] == args.split and s.get("view") == view])
    if len(ids):
      report[f"real_{view}"] = metrics(*scores(r, rc, z[ids], robot[ids]), truth[ids])
      print(f"REAL latents, {view}: {json.dumps(report[f'real_{view}'])}", flush=True)

  # imagined latents: 20 windows per obstacle rollout, D rolled forward with the recorded actions
  for horizon in (1, 4, 8):
    pred, true = [], []
    for roll in manifest["rollouts"]:
      if roll["split"] != args.split or roll.get("view") != "obstacles" or len(roll["actions"]) < horizon:
        continue
      for start in np.unique(np.linspace(0, len(roll["actions"]) - horizon, 20, dtype=int)):
        s0, s1 = roll["states"][start], roll["states"][start + horizon]
        zz, pp = imagine(d, dc, z[s0], robot[s0], roll["actions"][start:start + horizon])
        pred.append((zz[-1], pp[-1]))
        true.append(truth[s1])
    if pred:
      zz = np.stack([a for a, _ in pred])
      pp = np.stack([b for _, b in pred])
      report[f"imagined_h{horizon}_obstacles"] = metrics(*scores(r, rc, zz, pp), np.stack(true))
      print(f"IMAGINED h={horizon}, obstacles: {json.dumps(report[f'imagined_h{horizon}_obstacles'])}", flush=True)

  out = args.run / "attempts" / args.tag / "reports" / f"reward_{args.split}.json"
  write_json(out, report)
  print(f"saved {out}")


if __name__ == "__main__":
  main()
