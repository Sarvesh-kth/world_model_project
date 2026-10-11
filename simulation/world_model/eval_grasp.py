import argparse
import json
import pathlib

import numpy as np
import torch

from .common import load_run, load_features, state_arrays, write_json
from .train import load_model, readout, reward_terms, normalized
from controller import planner

# Does the world model know a grasp from a miss? From recorded states of a split where the open hand hovers
# near the cube, D imagines the planner's go-to-and-close rule aimed at the cube and at points 2 / 4 / 6 cm
# beside it and 3 / 5 cm above it over the planner's horizon; Q then reads held on the imagined end state,
# D's finger width is checked and R gives its grasp term and empty-grasp probability. A model that has only
# seen clean grasps calls every close a hold (held ~0.9 at every offset, width stuck at the cube's 4.5 cm).
# The second part replays the RECORDED closes of the split through D (the window starts 3 steps before the
# fingers close, the recorded actions follow) and compares Q's held on the imagined states with the recorded
# label, separately for closes that held and closes that missed: hallucination on real action sequences.
#   python -m world_model.eval_grasp --run data/combined_v2 --tag combined --split test

OFFSETS = (("at the cube", 0, 0), ("2 cm beside", .02, 0), ("4 cm beside", .04, 0), ("6 cm beside", .06, 0),
           ("3 cm above", 0, .03), ("5 cm above", 0, .05))


# D one step on raw units for a batch
def predict(d, dc, z, p, actions):
  with torch.inference_mode():
    zz, pp = d(normalized(z, dc, "z"), normalized(p, dc, "p"), torch.as_tensor(actions, dtype=torch.float32))
  return zz.numpy() * dc["z_std"].numpy() + dc["z_mean"].numpy(), pp.numpy() * dc["p_std"].numpy() + dc["p_mean"].numpy()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", type=pathlib.Path, default=pathlib.Path("data/combined_v2"))
  ap.add_argument("--tag", default="combined")
  ap.add_argument("--split", default="test")
  ap.add_argument("--states", type=int, default=200, help="hover states to start from")
  ap.add_argument("--horizon", type=int, default=12)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  manifest = load_run(args.run)
  z, _ = load_features(args.run, manifest)
  p, xyz, held = state_arrays(manifest)
  split = np.asarray([s["split"] for s in manifest["states"]])
  table = manifest["config"]["table"]["height"]
  rest_z = float(np.median([x[2] for x, h, s in zip(xyz, held, split) if not h and abs(x[2] - table) < .06]))
  d, dc = load_model(args.run, "dynamics", args.tag)
  q, qc = load_model(args.run, "readout", args.tag)
  r, rc = load_model(args.run, "reward", args.tag)
  max_delta = manifest["config"]["control"]["max_delta"]

  # open hand within 4 cm beside the cube and 4-14 cm above it, the cube resting on the table
  ee = p[:, 14:17]
  above = ee[:, 2] - xyz[:, 2]
  ok = ((split == args.split) & (held < .5) & (p[:, 19] > 0) & (np.linalg.norm(ee[:, :2] - xyz[:, :2], axis=1) < .04)
        & (above > .04) & (above < .14) & (np.abs(xyz[:, 2] - rest_z) < .02))
  rng = np.random.default_rng(args.seed)
  starts = rng.choice(np.flatnonzero(ok), min(args.states, int(ok.sum())), replace=False)
  print(f"{len(starts)} hover states of the {args.split} split, rest height {rest_z:.4f}", flush=True)

  report = {"split": args.split, "tag": args.tag, "states": int(len(starts)), "offsets": {}}
  for name, beside, up in OFFSETS:
    n = len(starts)
    angle = rng.uniform(0, 2 * np.pi, n)
    waypoints = xyz[starts].copy()
    waypoints[:, 0] += beside * np.cos(angle)
    waypoints[:, 1] += beside * np.sin(angle)
    waypoints[:, 2] += up - .015
    switch = np.full(n, planner.ON_ARRIVAL)
    grip_after = np.full(n, -1.0, np.float32)
    grip = np.ones(n, np.float32)
    zs, ps = z[starts].copy(), p[starts].copy()
    for t in range(args.horizon):
      actions = planner.goto_actions(waypoints.astype(np.float32), switch, grip_after, grip, ps[:, 14:17], t, max_delta, rest_z + .08)
      grip = actions[:, 4]
      zs, ps = predict(d, dc, zs, ps, actions)
    closed = grip < 0
    _, held_read = readout(q, qc, zs, ps)
    terms = reward_terms(r, rc, zs, ps)
    row = {"closed": float(closed.mean())}
    if closed.any():
      row.update({"held": float((held_read[closed] >= .5).mean()), "width_cm": float(100 * ps[closed, 18].mean()),
                  "r_grasp": float(terms["grasp"][closed].mean()),
                  "r_empty_grasp": float(terms["empty_grasp"][closed].mean()) if "empty_grasp" in terms else None})
    report["offsets"][name] = row
    print(f"{name:12s}: closed within the horizon {row['closed']:.2f}" + (
      f"; of those: held {row['held']:.2f}, finger width {row['width_cm']:.1f} cm, R grasp {row['r_grasp']:.2f}, "
      f"R empty grasp {row['r_empty_grasp'] if row['r_empty_grasp'] is None else round(row['r_empty_grasp'], 2)}" if closed.any() else ""), flush=True)

  # recorded closes: imagined held and width against the truth, at 1, 4 and 8 steps after the close
  closes = {"held": [], "missed": []}
  for roll in manifest["rollouts"]:
    if roll["split"] != args.split:
      continue
    ids = roll["states"]
    for j in range(1, len(ids) - 8):
      if p[ids[j], 19] < 0 and p[ids[j - 1], 19] > 0:
        outcome = "held" if held[ids[j:j + 8]].any() else "missed"
        closes[outcome].append((ids[max(j - 3, 0)], roll["actions"][max(j - 3, 0):j + 8], j - max(j - 3, 0)))
  for outcome, windows in closes.items():
    if not windows:
      continue
    row = {"closes": len(windows)}
    for after in (1, 4, 8):
      imagined_held, true_held, width_error = [], [], []
      for start, actions, close_at in windows:
        zs, ps = z[start][None].copy(), p[start][None].copy()
        for action in actions[:close_at + after]:
          zs, ps = predict(d, dc, zs, ps, np.asarray(action, np.float32)[None])
        k = start + close_at + after
        _, h = readout(q, qc, zs, ps)
        imagined_held.append(float(h[0] >= .5))
        true_held.append(float(held[k]))
        width_error.append(abs(float(ps[0, 18]) - float(p[k, 18])))
      row[f"after_{after}"] = {"imagined_held": float(np.mean(imagined_held)), "true_held": float(np.mean(true_held)),
                               "width_error_cm": float(100 * np.mean(width_error))}
    report[f"recorded_{outcome}"] = row
    print(f"recorded closes that {outcome:6s} ({len(windows)}): " + "; ".join(
      f"{a} steps after: imagined held {row[f'after_{a}']['imagined_held']:.2f} vs true {row[f'after_{a}']['true_held']:.2f}, width off {row[f'after_{a}']['width_error_cm']:.1f} cm"
      for a in (1, 4, 8)), flush=True)

  out = args.run / "attempts" / args.tag / "reports" / f"grasp_{args.split}.json"
  write_json(out, report)
  print(f"saved {out}")


if __name__ == "__main__":
  main()
