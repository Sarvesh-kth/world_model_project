import argparse
import csv
import json
import pathlib

import numpy as np

# What happened in one recorded controller episode (any method), from the files the episode already wrote:
# a short text diagnosis and one picture, diagnosis.png, with six camera frames and six plots. Needs no GPU,
# no models and no rendering, so it takes a second.
#   python -m controller.diagnose data/<run>/episodes/mpc/seed_20494010
# To look at it in JupyterLab: double-click diagnosis.png (or actual.gif, the camera view) in the file
# browser, or in a notebook cell (any kernel):
#   from IPython.display import Image, display
#   display(Image("data/<run>/episodes/mpc/seed_20494010/diagnosis.png"))


def read_steps(path):
  rows = list(csv.DictReader(open(path)))
  def column(name):
    values = [r.get(name, "") for r in rows]
    if all(v in ("", None) for v in values):
      return None
    return np.array([{"True": 1.0, "False": 0.0}.get(v, np.nan if v in ("", None) else v) for v in values], float)
  return column


def load(folder):
  folder = pathlib.Path(folder)
  t = np.load(folder / "trajectory.npz")
  ep = {"folder": folder, "result": json.loads((folder / "result.json").read_text()), "column": read_steps(folder / "steps.csv"),
        "gripper": t["p"][:, 14:17], "width": t["p"][:, 18], "cube": t["object_xyz"], "held": t["held"].astype(float),
        "actions": t["actions"], "plans": dict(np.load(folder / "plans.npz")) if (folder / "plans.npz").exists() else None,
        "candidates": None}
  if (folder / "candidates.csv").exists():
    rows = list(csv.DictReader(open(folder / "candidates.csv")))
    ep["candidates"] = {k: np.array([float(r[k]) if k != "valid" else r[k] == "True" for r in rows]) for k in ("step", "score", "valid")}
  layout = ep["result"]["layout"]
  ep["goal"] = np.array([*layout["place"][:2], ep["cube"][0, 2]])
  return ep


# Per step spread of the candidate scores (best minus median of the valid ones): near zero means the
# planner had nothing to choose between. Steps with a single candidate (mpc's retreat after placing) are skipped.
def score_spread(candidates):
  steps = np.unique(candidates["step"])
  best, spread = [], []
  for s in steps:
    scores = candidates["score"][(candidates["step"] == s) & candidates["valid"]]
    best.append(scores.max() if len(scores) > 1 else np.nan)
    spread.append(scores.max() - np.median(scores) if len(scores) > 1 else np.nan)
  return steps, np.array(best), np.array(spread)


def report(ep):
  r, col, g, cube = ep["result"], ep["column"], ep["gripper"], ep["cube"]
  lines = [f"{ep['folder'].parent.name} {ep['folder'].name}: placed={r['task_success']}, final distance to B "
           f"{r['final_goal_distance_cm']:.1f} cm, {len(ep['actions'])} steps"]
  to_cube = 100 * np.linalg.norm(g - cube, axis=1)
  path = 100 * np.linalg.norm(np.diff(g, axis=0), axis=1).sum()
  lines.append(f"arm: moved {path:.0f} cm in total, ended {100 * np.linalg.norm(g[-1] - g[0]):.0f} cm from where it started; "
               f"closest to the cube {to_cube.min():.1f} cm (step {to_cube.argmin()}), started {to_cube[0]:.1f} cm away")
  closed = ep["actions"][:, -1] <= 0
  if closed.any():
    first = int(np.argmax(closed))
    far = int((closed & (to_cube[:-1] > 5)).sum())
    lines.append(f"gripper: commanded closed on {closed.sum()} of {len(closed)} steps, first at step {first} with the cube "
                 f"{to_cube[first]:.1f} cm away; closed while more than 5 cm from the cube on {far} steps")
  else:
    lines.append("gripper: never commanded closed")
  lines.append(f"cube: really held on {int(ep['held'].sum())} steps, moved {100 * np.linalg.norm(cube[-1] - cube[0]):.1f} cm")

  findings = []
  if path < 5:
    findings.append("the arm barely moved")
  elif to_cube.min() > 5:
    findings.append("the arm moved but never got within 5 cm of the cube")
  q_held, q_error = col("Q_held_probability"), col("Q_xyz_mae_cm")
  if q_held is not None:
    false_held = int(((q_held > .5) & (ep["held"][1:] < .5)).sum())
    lines.append(f"Q (camera): cube position off by {np.nanmean(q_error):.1f} cm on average; said held while not held on "
                 f"{false_held} steps, said not held while held on {int(((q_held <= .5) & (ep['held'][1:] > .5)).sum())}")
    if np.nanmean(q_error) > 3:
      findings.append(f"Q's cube estimate is poor ({np.nanmean(q_error):.1f} cm average error)")
  d_ee, d_cube = col("D_one_step_ee_mae_cm"), col("D_Q_one_step_xyz_mae_cm")
  if d_ee is not None:
    lines.append(f"D (one step ahead, the chosen plan): gripper off by {np.nanmean(d_ee):.2f} cm, cube by {np.nanmean(d_cube):.2f} cm")
  valid, fallback = col("valid_candidates"), col("planner_fallback")
  if valid is not None:
    lines.append(f"planner: valid candidates per step {np.nanmean(valid):.0f} on average (min {np.nanmin(valid):.0f}), "
                 f"fallback (none valid, hold still) on {int(np.nansum(fallback))} steps, "
                 f"{np.nanmean(col('decision_seconds')):.2f} s per decision")
    if np.nanmean(fallback) > .2:
      findings.append("the planner fell back to holding still on many steps (every candidate left the valid range)")
  if ep["candidates"] is not None:
    _, best, spread = score_spread(ep["candidates"])
    single = int(np.isnan(spread).sum())
    lines.append(f"scores: best candidate {np.nanmean(best):.3f} on average, best minus median {np.nanmean(spread):.4f}"
                 + (f"; no search (retreat after letting go at B) on {single} steps" if single else ""))
    if np.nanmean(spread) < .01:
      findings.append("all candidates scored almost the same: the score gives the planner nothing to choose by")
  plans = ep["plans"]
  if plans is not None:
    imagined_move = 100 * np.nanmax(np.linalg.norm(plans["chosen_cube"] - plans["chosen_cube"][:, :1], axis=2), axis=1)
    reach = np.linalg.norm(plans["chosen_gripper"][:, -1] - plans["chosen_gripper"][:, 0], axis=1)
    lines.append(f"imagination: the chosen plan moved the gripper {100 * np.nanmean(reach):.1f} cm over the horizon on "
                 f"average; it imagined the cube moving more than 2 cm on {int((imagined_move > 2).sum())} steps")
    if "chosen_held" in plans:
      imagined_held = np.nanmax(plans["chosen_held"], axis=1) > .5
      fooled = int((imagined_held & (ep["held"][:-1] < .5)).sum())
      lines.append(f"imagination: the chosen plan expected to hold the cube on {int(imagined_held.sum())} steps, "
                   f"{fooled} of them while the cube was not held")
      if fooled > .2 * len(imagined_held):
        findings.append("the planner kept expecting a grasp that never happened (the model is fooled by its own plans)")
  lines.append("findings: " + ("; ".join(findings) if findings else "nothing obviously wrong in these numbers"))
  return lines


def figure(ep, out):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from PIL import Image
  col, g, cube, goal = ep["column"], ep["gripper"], ep["cube"], ep["goal"]
  steps = np.arange(len(g))
  fig = plt.figure(figsize=(16, 11), constrained_layout=True)
  grid = fig.add_gridspec(3, 6, height_ratios=[1, 1.15, 1.15])
  fig.suptitle(report(ep)[0], fontsize=12)

  # six camera frames, evenly spread
  frames = sorted((ep["folder"] / "frames").glob("*.jpg"))
  for k, i in enumerate(np.linspace(0, len(frames) - 1, 6).astype(int) if frames else []):
    ax = fig.add_subplot(grid[0, k])
    ax.imshow(Image.open(frames[i]))
    ax.set_title(f"step {i}", fontsize=9)
    ax.axis("off")

  # top view, turned like the camera image (robot at the top, table left to right): what happened (blue)
  # and what the planner imagined (orange) every few steps
  ax = fig.add_subplot(grid[1, 0:2])
  plans = ep["plans"]
  if plans is not None:
    for t in range(0, len(plans["chosen_gripper"]), 5):
      path = plans["chosen_gripper"][t]
      ax.plot(path[:, 1], path[:, 0], color="tab:orange", alpha=.45, lw=1)
    ax.plot([], [], color="tab:orange", label="imagined plan (every 5th step)")
  ax.plot(g[:, 1], g[:, 0], color="tab:blue", lw=2, label="gripper, actual")
  ax.plot(g[0, 1], g[0, 0], "o", color="tab:blue")
  ax.plot(cube[:, 1], cube[:, 0], color="tab:purple", lw=2)
  ax.plot(cube[0, 1], cube[0, 0], "s", color="tab:purple", ms=9, label="cube")
  ax.add_patch(plt.Circle((goal[1], goal[0]), .07, color="tab:green", alpha=.3, label="target B"))
  ax.set_xlim(-.45, .45)
  ax.set_ylim(.5, -.15)
  ax.set_aspect("equal")
  ax.set_xlabel("y (m)")
  ax.set_ylabel("x (m)")
  ax.set_title("seen from above, turned like the camera image")
  ax.legend(fontsize=7, loc="lower left")

  ax = fig.add_subplot(grid[1, 2:4])
  ax.plot(steps, 100 * np.linalg.norm(g - cube, axis=1), label="gripper to cube")
  ax.plot(steps, 100 * np.linalg.norm(cube[:, :2] - goal[:2], axis=1), label="cube to B")
  ax.plot(steps, 100 * (g[:, 2] - cube[0, 2]), label="gripper height over the cube's rest", alpha=.7)
  ax.set_title("distances (cm)")
  ax.set_xlabel("step")
  ax.legend(fontsize=8)

  ax = fig.add_subplot(grid[1, 4:6])
  ax.step(steps[1:], ep["actions"][:, -1] > 0, where="post", label="command open")
  ax.plot(steps, ep["width"] / .085, label="finger opening (1 = fully open)")
  ax.plot(steps, ep["held"], label="really held", lw=2)
  if col("Q_held_probability") is not None:
    ax.plot(steps[1:], col("Q_held_probability"), "--", label="Q says held")
  if plans is not None and "chosen_held" in plans:
    ax.plot(steps[:-1], np.nanmax(plans["chosen_held"], axis=1), ":", label="plan expects held")
  ax.set_ylim(-.05, 1.1)
  ax.set_title("gripper and grasp")
  ax.set_xlabel("step")
  ax.legend(fontsize=8)

  ax = fig.add_subplot(grid[2, 0:2])
  ax.plot(steps[1:], ep["actions"][:, :3] * 4, lw=1)
  ax.legend(["dx", "dy", "dz"], fontsize=8)
  ax.set_title("executed motion (cm per step)")
  ax.set_xlabel("step")

  ax = fig.add_subplot(grid[2, 2:4])
  if ep["candidates"] is not None:
    s, best, spread = score_spread(ep["candidates"])
    ax.plot(s, best, label="best candidate")
    ax.plot(s, best - spread, label="median candidate")
    ax2 = ax.twinx()
    ax2.plot(steps[1:], col("valid_candidates"), color="grey", alpha=.4, label="valid candidates")
    ax2.set_ylabel("valid candidates", color="grey")
    ax.legend(fontsize=8, loc="upper left")
  else:
    ax.text(.5, .5, "no planner in this episode", ha="center", transform=ax.transAxes)
  ax.set_title("planner scores")
  ax.set_xlabel("step")

  ax = fig.add_subplot(grid[2, 4:6])
  for name, label in (("Q_xyz_mae_cm", "Q: cube position error"), ("D_one_step_ee_mae_cm", "D: gripper one step ahead"),
                      ("D_Q_one_step_xyz_mae_cm", "D+Q: cube one step ahead")):
    if col(name) is not None:
      ax.plot(steps[1:], col(name), label=label, lw=1)
  ax.set_title("model errors (cm)")
  ax.set_xlabel("step")
  ax.legend(fontsize=8)

  fig.savefig(out, dpi=90)
  plt.close(fig)


def main():
  p = argparse.ArgumentParser(description="text diagnosis and a one-picture summary of a recorded episode")
  p.add_argument("episode", type=pathlib.Path, help="data/<run>/episodes/<method>/seed_<seed>")
  p.add_argument("--no-figure", action="store_true")
  args = p.parse_args()
  ep = load(args.episode)
  print("\n".join(report(ep)))
  if not args.no_figure:
    figure(ep, args.episode / "diagnosis.png")
    print(f"wrote {args.episode / 'diagnosis.png'} (camera view: {args.episode / 'actual.gif'})")


if __name__ == "__main__":
  main()
