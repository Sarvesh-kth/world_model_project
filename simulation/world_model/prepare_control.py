import argparse
import csv
import json
import pathlib
import shutil

import numpy as np

from environment.config import Config
from environment.rewards import Rewards, COMPONENTS
from .common import new_manifest, write_json

# Turn the planner's own closed-loop episodes (controller/control_pipeline, data/control_*/episodes/jepa_mpc)
# into a run folder, so Q, R and D also train on the states the planner itself produces: the hand roaming
# low beside the cube, closes that miss, drops and the cube at odd places afterwards. Every state has the
# true cube position and held flag from the simulator, both camera frames, and the latent the controller
# computed online (identical to the offline encoder, so nothing has to be encoded). Reward components are
# recomputed from the recorded state with the environment's own Rewards. Only episodes without obstacles
# for now: the proximity term needs the obstacle distances, which are not recorded.
#   python -m world_model.prepare_control --control data/control_v4_empty data/control_v3_empty --run data/planner_v1
#   python -m world_model.merge_runs --runs data/full_v2 data/obstacles_v2 data/episodes_v2 data/planner_v1 --out data/combined_v2 --overwrite


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--control", nargs="+", type=pathlib.Path, required=True, help="control run folders")
  p.add_argument("--run", type=pathlib.Path, required=True, help="run folder to create")
  p.add_argument("--method", default="jepa_mpc")
  p.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("data/combined_v2"),
                 help="the run whose encoder settings and PCA basis the control episodes were made with")
  args = p.parse_args()
  folders = [f for root in args.control for f in sorted((root / "episodes" / args.method).glob("seed_*")) if (f / "trajectory.npz").exists()]
  if not folders:
    p.error("no episodes found")
  info = json.loads((args.models_run / "attempts/combined/info.json").read_text())
  cfg = Config.nested(info["config"])
  layout_file = json.loads(pathlib.Path("configs/grade_e_layout.json").read_text())
  manifest = new_manifest("planner", info["config"], {k: str(v) for k, v in vars(args).items()})
  latents = []
  args.run.mkdir(parents=True, exist_ok=True)
  (args.run / "episodes").mkdir(exist_ok=True)

  for index, folder in enumerate(folders):
    result = json.loads((folder / "result.json").read_text())
    layout = result.get("layout", layout_file)
    if layout.get("obstacles"):
      print(f"skipping {folder}: obstacle episodes are not supported yet")
      continue
    t = np.load(folder / "trajectory.npz")
    rows = list(csv.DictReader((folder / "steps.csv").open()))
    n = len(t["p"])
    if len(rows) != n - 1 or len(t["z"]) != n:
      raise ValueError(f"{folder}: {n} states, {len(rows)} steps, {len(t['z'])} latents")
    scene = f"{folder.parent.parent.parent.name}_{folder.name}"
    # train and val only: the planner's episodes are training material, the recorded runs keep the test split
    split = "val" if index % 5 == 4 else "train"
    link = args.run / "episodes" / scene
    if not link.exists():
      link.symlink_to(folder.resolve())
    start = len(manifest["states"])
    rest_z = float(info["rest_z"])
    place = np.array([*layout["place"], cfg.table.height])
    rewards = Rewards(cfg)
    history, wrist_history, actions = [], [], []
    for j in range(n):
      p_, xyz, held = t["p"][j], t["object_xyz"][j], bool(t["held"][j])
      for camera, kept in (("static", history), ("wrist", wrist_history)):
        kept.append(f"episodes/{scene}/frames/{camera}_{j:04d}.jpg")
      action = [float(v) for v in t["actions"][j - 1]] if j else None
      if j:
        actions.append(action)
      row = rows[j - 1] if j else None
      # the environment's components from the recorded state; an empty table has no obstacle terms
      state = {"object_pos": xyz, "ee_pos": p_[14:17], "place_pos": place, "object_rest_z": rest_z, "grasped": held,
               "gripper_closed": bool(p_[19] < 0), "gripper_width": float(p_[18]),
               "obstacle_contacts": 0.0, "table_contacts": float(row["table_contact"] == "True") if row else 0.0,
               "obstacle_distance": cfg.rewards.safe_distance, "failed": j == n - 1 and not result["task_success"]}
      total, comps, _ = rewards.compute(state, np.asarray(action if action else [0, 0, 0, 0, 1.0]))
      manifest["states"].append({
        "key": f"{scene}:{j:04d}", "scene": scene, "split": split, "frames": history[-64:],
        "wrist_frames": wrist_history[-64:], "heightmap": (t["z"][j, 2048:] * cfg.cameras.heightmap.max_height).tolist(),
        "p": p_.tolist(), "object_xyz": xyz.tolist(), "object_quat": [1.0, 0.0, 0.0, 0.0], "held": held,
        "obstacle_contact": False, "table_contact": bool(state["table_contacts"]),
        "reward_stage": rewards.stage, "reward": float(total), "reward_components": {k: float(comps[k]) for k in COMPONENTS},
        "action_from_previous": action, "time": j / cfg.control.hz,
        "view": "empty", "case": "planner", "task_success": bool(result["task_success"])})
    manifest["rollouts"].append({"id": scene, "scene": scene, "split": split, "view": "empty", "case": "planner",
                                 "states": list(range(start, start + n)), "actions": actions,
                                 "result": {"task_success": bool(result["task_success"]), "steps": n, "obstacle_contact_steps": 0}})
    manifest["scene_settings"].append({"rollout": scene, "seed": result["seed"], "layout": layout, "policy": "planner"})
    latents.append(np.asarray(t["z"], np.float32))

  manifest["complete"] = True
  write_json(args.run / "manifest.json", manifest)
  write_json(args.run / "scene_settings.json", manifest["scene_settings"])
  features = args.run / "features"
  features.mkdir(exist_ok=True)
  np.save(features / "latents.npy", np.concatenate(latents))
  shutil.copyfile(args.models_run / "features/pca.npz", features / "pca.npz")
  meta = json.loads((args.models_run / "features/meta.json").read_text())
  write_json(features / "meta.json", {**{k: v for k, v in meta.items() if k not in ("keys", "seconds", "merged_from")},
                                      "keys": [s["key"] for s in manifest["states"]]})
  held = sum(s["held"] for s in manifest["states"])
  closed_on_air = sum(1 for s in manifest["states"] if s["reward_components"]["empty_grasp"])
  print(f"{len(manifest['rollouts'])} planner episodes -> {len(manifest['states'])} states; held {held}; fingers shut on nothing {closed_on_air}")


if __name__ == "__main__":
  main()
