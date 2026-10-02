"""Check scene splits, alignment, restored sources, images and label/outcome balance."""
import argparse
from collections import Counter, defaultdict
import pathlib

import numpy as np
from PIL import Image, ImageDraw

from .data import load_run, write_json, clip_paths


def audit(root, manifest, images=True):
  problems, scenes, referenced = [], defaultdict(set), set()
  states, rollouts = manifest["states"], manifest["rollouts"]
  if not manifest.get("complete"):
    problems.append("collection is incomplete; inspect the collection traceback and use a new run directory")
  if len({s["key"] for s in states}) != len(states):
    problems.append("duplicate state keys")
  files = set()
  for s in states:
    scenes[s["scene"]].add(s["split"])
    if (len(s["p"]) != 20 or len(s["object_xyz"]) != 3
        or not np.isfinite(s["p"] + s["object_xyz"]).all()):
      problems.append(f"bad/nonfinite state {s['key']}")
    if not s["frames"] or len(s["frames"]) > 64:
      problems.append(f"bad camera history {s['key']}")
    files.update(s["frames"])
  if any(len(v) != 1 for v in scenes.values()):
    problems.append("a scene group appears in multiple splits: leakage")
  pairs = defaultdict(dict)
  for r in rollouts:
    if any(not isinstance(i, int) or i < 0 or i >= len(states) for i in r["states"]):
      problems.append(f"invalid state index {r['id']}")
      continue
    referenced.update(r["states"])
    source = states[r["states"][0]]
    if source["scene"] != r["scene"] or source["split"] != r["split"]:
      problems.append(f"rollout source split/scene mismatch {r['id']}")
    pairs[(r["scene"], r["placement"])][r["branch"]] = r
    if len(r["states"]) != len(r["actions"]) + 1:
      problems.append(f"unaligned rollout {r['id']}")
      continue
    if max(r["restore_p_error"], r["restore_integration_error"]) > 1e-5:
      problems.append(f"restore mismatch {r['id']}")
    for j, action in enumerate(r["actions"], 1):
      s, prev = states[r["states"][j]], states[r["states"][j-1]]
      if s["scene"] != r["scene"] or s["split"] != r["split"]:
        problems.append(f"source/target split mismatch {r['id']} step {j}")
      if len(action) != 5 or not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
        problems.append(f"invalid action {r['id']} step {j}")
      if s["action_from_previous"] != action:
        problems.append(f"action not aligned to next state {s['key']}")
      if abs(s["time"]-prev["time"]-1/manifest["control_hz"]) > 1e-6:
        problems.append(f"time gap {s['key']}")
      if s["frames"][:-1] != (prev["frames"] + s["frames"][-1:])[-64:][:-1]:
        problems.append(f"camera history does not follow branch {s['key']}")
  if referenced != set(range(len(states))):
    problems.append("orphan or invalid state indices")
  contrasts, p_errors = [], []
  for (scene, placement), pair in pairs.items():
    if not {"close_lift", "open_lift"}.issubset(pair):
      problems.append(f"missing action branch {scene}/{placement}")
      continue
    a, b = pair["close_lift"], pair["open_lift"]
    if len({r["states"][0] for r in pair.values()}) != 1:
      problems.append(f"action branches do not share source {scene}/{placement}")
    contrast = states[a["states"][-1]]["object_xyz"][2]-states[b["states"][-1]]["object_xyz"][2]
    contrasts.append({"scene": scene, "placement": placement, "split": a["split"],
                      "height_effect_cm": 100*contrast})
  for scene in scenes:
    near, far = pairs.get((scene, "under"), {}), pairs.get((scene, "offset"), {})
    if not {"close_lift", "open_lift"}.issubset(near) or set(near) != set(far):
      problems.append(f"missing position pair {scene}")
      continue
    p1 = states[near["close_lift"]["states"][0]]["p"]
    p2 = states[far["close_lift"]["states"][0]]["p"]
    error = float(np.max(np.abs(np.asarray(p1)-p2)))
    p_errors.append(error)
    if error > manifest["settings"]["pair_tolerance"]:
      problems.append(f"position pair robot state mismatch {scene}: {error}")
    for branch in near:
      if near[branch]["actions"] != far[branch]["actions"]:
        problems.append(f"position pair actions differ {scene}/{branch}")
  for split in ("train", "val", "test"):
    labels = [s["held"] for s in states if s["split"] == split]
    if not labels or not any(labels) or all(labels):
      problems.append(f"{split} needs both held and not-held labels; grasp setup may have failed")
    eligible = [c for c in contrasts if c["split"] == split and c["placement"] == "under"
                and abs(c["height_effect_cm"]) >= 2]
    if not eligible:
      problems.append(f"{split} has no under-gripper A/B height effect >= 2 cm")
  if images:
    for filename in sorted(files):
      try:
        with Image.open(root / filename) as img:
          img.verify()
      except (OSError, ValueError) as e:
        problems.append(f"missing/corrupt frame {filename}: {e}")
  return {"problems": problems, "states": len(states), "rollouts": len(rollouts),
          "scene_groups": len(scenes), "scenes_by_split": dict(Counter(next(iter(v)) for v in scenes.values())),
          "held_labels_by_split": {split: dict(Counter(str(s['held']) for s in states if s['split'] == split))
                                   for split in ("train", "val", "test")},
          "distinct_frames": len(files), "max_position_pair_p_error": max(p_errors, default=0),
          "branches": dict(Counter(r["branch"] for r in rollouts)),
          "rejected_attempts": len(manifest.get("rejected_attempts", [])),
          "contrasts": contrasts}


def previews(root, manifest):
  # ponytail: one contact sheet per scene; videos can be added when these are insufficient.
  states = manifest["states"]
  for scene in sorted({r["scene"] for r in manifest["rollouts"]}):
    rows = [r for r in manifest["rollouts"] if r["scene"] == scene]
    sheet = Image.new("RGB", (4*256, len(rows)*284), "white")
    draw = ImageDraw.Draw(sheet)
    for row, r in enumerate(rows):
      h = len(r["actions"])
      for column, step in enumerate((0, h//3, 2*h//3, h)):
        s = states[r["states"][step]]
        with Image.open(clip_paths(root, s)[-1]) as img:
          sheet.paste(img.resize((256, 256)), (column*256, row*284))
        draw.text((column*256+4, row*284+258), f"{r['placement']} {r['branch']} t={step} held={s['held']}", fill="black")
    sheet.save(root / "reports" / f"{scene}.jpg")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--run", required=True, type=pathlib.Path)
  args = p.parse_args()
  manifest = load_run(args.run)
  report = audit(args.run, manifest)
  write_json(args.run / "reports/audit.json", report)
  print(f"scenes by split: {report['scenes_by_split']}")
  print(f"states={report['states']} rollouts={report['rollouts']} frames={report['distinct_frames']}")
  print(f"held labels: {report['held_labels_by_split']}")
  print(f"position-pair maximum robot mismatch: {report['max_position_pair_p_error']:.3g}")
  for c in report["contrasts"]:
    if c["placement"] == "under":
      print(f"  {c['scene']} {c['split']}: real close/open object-height effect {c['height_effect_cm']:+.2f} cm")
  if report["problems"]:
    for problem in report["problems"]:
      print(f"ERROR: {problem}")
    raise SystemExit("Audit failed; do not encode/train this dataset.")
  previews(args.run, manifest)
  print(f"PASS: alignment/splits/restores/images/labels checked. Inspect {args.run}/reports/scene_*.jpg")


if __name__ == "__main__":
  main()
