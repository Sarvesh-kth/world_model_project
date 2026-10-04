"""Plot saved choices/forecasts and animate real camera frames; no GPU or model loading."""
import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from .data import digest


def rows(path):
  with path.open(newline="") as stream:
    return list(csv.DictReader(stream))


def overview(report, output):
  fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True, layout="constrained")
  versions = ("original", "robot", "visual", "real_Q_diagnostic")
  names = ("Original D", "Robot correction", "Visual correction", "Q on real future")
  for ax, horizon in zip(axes, (4, 8, 16, 30)):
    selected = [next(m for m in report["metrics"] if m["split"] == "test"
                and m["source_phase"] == "all" and m["horizon"] == horizon and m["version"] == v)
                for v in versions]
    for i, m in enumerate(selected):
      total, success, failed = m["decisions"], m["chosen_successes"], m["false_successes"]
      abstained = round(total*m["abstention_rate"])
      assert success+failed+abstained == total
      left = 0
      for count, color, label in ((success, "#22855c", "Successful choice"),
                                  (failed, "#c44949", "Chosen failure"),
                                  (abstained, "#dadde3", "Abstained")):
        width = 100*count/total
        ax.barh(i, width, left=left, color=color, label=label if i == 0 else None)
        if width >= 10:
          ax.text(left+width/2, i, str(count), ha="center", va="center", fontsize=10)
        left += width
    m = selected[0]
    available = round(m["decisions"]*m["candidate_success_coverage"])
    ax.set_title(f"{horizon} steps ({horizon/10:g} s)\nSuccess available: {available}/{m['decisions']}")
    ax.set_xlim(0, 100); ax.set_xlabel("Decision sets (%)")
    ax.set_yticks(range(4), names)
  axes[0].invert_yaxis()
  gates = "/".join("PASS" if report["Q_gates"][s]["passed"] else "FAIL" for s in ("val", "test"))
  fig.suptitle(f"Fresh test decisions — seed {report['seed']} — Q validation/test: {gates}")
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="outside lower center", ncol=3)
  fig.savefig(output, dpi=160); plt.close(fig)


def forecast(report, candidates, steps, args, output):
  # One recorded source/horizon; no imagined frames are decoded from JEPA vectors.
  suffix = f"{args.scene}/{args.placement}/s00/h30/"
  candidates = [r for r in candidates if r["decision"].startswith(suffix) and r["candidate"] == args.branch]
  if not candidates:
    raise ValueError(f"no candidate {suffix}{args.branch}; check the saved candidate CSV")
  decision = candidates[0]["decision"]
  series = {v: sorted([r for r in steps if r["decision"] == decision and r["candidate"] == args.branch
                      and r["version"] == v], key=lambda r: int(r["step"])) for v in
            ("original", "robot", "visual", "real_Q_diagnostic")}
  actual = series["original"]
  real_height = np.array([float(r["real_height_cm"]) for r in actual])
  reference = real_height[-report["stable_steps"]:].min()-float(candidates[0]["real_min_lift_cm"])
  x = np.array([int(r["step"])/10 for r in actual])
  fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
  for ax, column, label in zip(axes, (None, "real_held", "real_width_cm"),
                               ("Cube rise (cm)", "Held label / probability", "Total finger width (cm)")):
    values = real_height-reference if column is None else [float(r[column] == "True") if column == "real_held"
                                                        else float(r[column]) for r in actual]
    ax.plot(x, values, color="#252525", linewidth=2.5, label="Real simulator")
    ax.set_xlabel("Time after candidate start (s)"); ax.set_ylabel(label); ax.grid(alpha=.2)
  for version, name, color in (("original", "Original D → Q", "#3774c2"),
                              ("robot", "Robot correction → Q", "#22855c"),
                              ("visual", "Visual correction → Q", "#d38324"),
                              ("real_Q_diagnostic", "Q on real future (diagnostic)", "#9255a7")):
    data = series[version]
    assert [r["step"] for r in data] == [r["step"] for r in actual]
    row = next(r for r in candidates if r["version"] == version)
    height = np.array([float(r["predicted_height_cm"]) for r in data])
    q_reference = height[-report["stable_steps"]:].min()-float(row["min_lift_cm"])
    for ax, values in zip(axes, (height-q_reference, [float(r["held_proxy"]) for r in data],
                                [float(r["predicted_width_cm"]) for r in data])):
      ax.plot(x, values, color=color, label=name, linestyle="--" if version == "real_Q_diagnostic" else "-")
  axes[0].axhline(report["goal_cm"], color="#777777", linestyle=":", label="5 cm goal")
  threshold = report["calibrations"]["30"]["threshold"]
  axes[1].axhline(threshold, color="#777777", linestyle=":")
  axes[1].set_ylim(-.05, 1.08 if threshold <= 1 else 1.15)
  axes[2].axhline(0, color="#777777", linestyle=":"); axes[2].axhline(8, color="#777777", linestyle=":")
  fig.suptitle(f"{args.scene} / {args.placement} / {args.branch} — seed {report['seed']} — actual success: {candidates[0]['real_success']}")
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="outside lower center", ncol=3)
  fig.savefig(output, dpi=160); plt.close(fig)


def animation(root, scene, placement, output, fps):
  manifest = json.loads((root / "manifest.json").read_text())
  branches = ("close_lift", "open_lift", "close_lift_release")
  rollouts = [next(r for r in manifest["rollouts"] if r["scene"] == scene
                  and r["placement"] == placement and r["branch"] == b) for b in branches]
  frames = []
  for t in range(min(len(r["states"]) for r in rollouts)):
    tiles = []
    for branch, rollout in zip(branches, rollouts):
      state = manifest["states"][rollout["states"][t]]
      with Image.open(root / state["frames"][-1]) as image:
        image = image.convert("RGB"); image.thumbnail((400, 400))
      tile = Image.new("RGB", (image.width, image.height+42), "white")
      tile.paste(image, (0, 42))
      ImageDraw.Draw(tile).text((6, 5), f"{branch} | t={t/manifest['control_hz']:.1f}s\nReal held: {state['held']}", fill="black")
      tiles.append(tile)
    canvas = Image.new("RGB", (sum(i.width for i in tiles), max(i.height for i in tiles)), "white")
    left = 0
    for tile in tiles:
      canvas.paste(tile, (left, 0)); left += tile.width
    frames.append(canvas)
  frames[0].save(output, save_all=True, append_images=frames[1:], duration=round(1000/fps), loop=0)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--results", type=pathlib.Path, default=pathlib.Path("../results/decision_v1"))
  parser.add_argument("--observations", type=pathlib.Path, default=pathlib.Path("data/decision_v1/observations"))
  parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/decision_v1/visualizations"))
  parser.add_argument("--scene", default="scene_0011")
  parser.add_argument("--placement", choices=("under", "offset"), default="under")
  parser.add_argument("--branch", default="close_lift")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--fps", type=int, default=10, help="GIF playback speed; source data stays at 10 Hz")
  args = parser.parse_args()
  if args.fps < 1:
    parser.error("fps must be positive")
  folder = args.results / "reports"
  report = json.loads((folder / f"seed_{args.seed}.json").read_text())
  if ((args.observations / "manifest.json").is_file()
      and digest(args.observations / "manifest.json") != report["manifest_sha256"]):
    raise ValueError("camera recordings belong to a different experiment; do not mix reports and frames")
  args.out.mkdir(parents=True, exist_ok=True)
  overview(report, args.out / f"choices_seed_{args.seed}.png")
  forecast(report, rows(folder / f"seed_{args.seed}_candidates.csv"),
           rows(folder / f"seed_{args.seed}_forecast_steps.csv"), args,
           args.out / f"{args.scene}_{args.placement}_{args.branch}_seed_{args.seed}.png")
  if (args.observations / "manifest.json").is_file():
    animation(args.observations, args.scene, args.placement,
              args.out / f"{args.scene}_{args.placement}_real_branches.gif", args.fps)
  else:
    print("Real GIF skipped: camera frames/manifest are on the notebook, outside the Git export.")
  print(f"Saved charts and available real video to {args.out.resolve()}")


if __name__ == "__main__":
  main()
