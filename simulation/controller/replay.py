import sys

# offscreen rendering (EGL on Linux) only for --video; a window needs GLFW instead, so this is decided
# before mujoco is imported. Without EGL export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa first.
if "--video" in sys.argv:
  import offscreen  # noqa: F401

import argparse
import json
import pathlib
import time

import cv2
import mujoco
import numpy as np

from environment import EpisodeLayout, scene
from environment.config import Config

# Replay a recorded episode from its saved simulator states with the mpc's imagined plans drawn on top, no
# GPU and no world model. Green = the chosen imagined hand path (faint for runner-ups), orange to red = the
# other candidates with --all-paths, purple = the imagined cube, blue = the real hand trail.
#   python -m controller.replay data/<run>/episodes/mpc/seed_20494010            # window (mjpython on a Mac)
#   python -m controller.replay data/<run>/episodes/mpc/seed_20494010 --video    # gif + mp4
# Window keys: space pause, left / right one step, up / down speed, R restart.

CHOSEN = (0.05, 0.85, 0.15, 0.95)
RUNNER_UP = (0.2, 0.85, 0.35, 0.25)
CUBE = (0.6, 0.2, 0.9, 0.8)
TRAIL = (0.15, 0.45, 1.0, 0.9)
IDENTITY = np.eye(3).flatten()


# The episode's scene, recorded states and (mpc only) the planner's imagined paths
def load_episode(folder, models_run=None, tag=None):
  folder = pathlib.Path(folder)
  result = json.loads((folder / "result.json").read_text())
  trajectory = np.load(folder / "trajectory.npz")
  if "qpos" not in trajectory:
    raise SystemExit(f"{folder}/trajectory.npz has no simulator states (qpos): it was recorded before replay "
                     "support, rerun the episode")
  # the environment config: from the models run named in the run's settings.json, as the controller loaded it
  settings = folder.parents[2] / "settings.json"
  if settings.exists() and (models_run is None or tag is None):
    run = json.loads(settings.read_text())
    models_run, tag = models_run or run["models_run"], run["tag"] if tag is None else tag
  # a packaged model folder (models/world_model, no tag) or a run folder with attempts/<tag>/
  models_run, tag = pathlib.Path(models_run or "models/world_model"), tag or ""
  info = models_run / "attempts" / tag / "info.json" if tag else models_run / "info.json"
  cfg = Config.nested(json.loads(info.read_text())["config"])
  model = scene.build_scene(cfg, EpisodeLayout.from_dict(result["layout"]))
  plans = dict(np.load(folder / "plans.npz")) if (folder / "plans.npz").exists() else None
  return {"folder": folder, "result": result, "model": model, "qpos": trajectory["qpos"],
          "gripper": trajectory["p"][:, 14:17], "plans": plans}


def add_sphere(scn, pos, radius, rgba):
  if scn.ngeom >= scn.maxgeom or not np.isfinite(pos).all():
    return
  mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                      np.asarray(pos, float), IDENTITY, np.asarray(rgba, np.float32))
  scn.ngeom += 1


def add_path(scn, points, width, rgba, dots=True):
  points = [q for q in points if np.isfinite(q).all()]
  for a, b in zip(points[:-1], points[1:]):
    if scn.ngeom >= scn.maxgeom or np.linalg.norm(b - a) < 1e-5:
      continue
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), IDENTITY, np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1
  if dots:
    for q in points[1:]:
      add_sphere(scn, q, 1.6 * width, rgba)


# Everything drawn for step t: the trail up to t and the plan decided at t (plan t produced action t)
def draw(scn, ep, t):
  plans = ep["plans"]
  if plans is None or t >= len(plans["chosen_gripper"]):
    for q in ep["gripper"][:t + 1:2]:
      add_sphere(scn, q, .006, TRAIL)
    return
  chosen = plans["chosen_gripper"][t]
  fallback = bool(plans["fallback"][t]) if "fallback" in plans else False
  if ep.get("show_all_paths", False):
    paths = plans["all_gripper"][t]
    distances = []
    for path in paths:
      common = np.isfinite(path).all(1) & np.isfinite(chosen).all(1)
      distances.append(float(np.linalg.norm(path[common] - chosen[common], axis=1).mean()) if common.any() else 0.)
    # colour is how far a path strays from the chosen one, nothing to do with reward
    maximum = max(max(distances), 1e-6)
    highlighted = set(plans["runner_indices"][t].tolist())
    if not fallback:
      highlighted.add(int(plans["chosen_index"][t]))
    for index, (path, distance) in enumerate(zip(paths, distances)):
      if index in highlighted or plans["all_round"][t, index] < 0:
        continue
      fraction = min(distance / maximum, 1.)
      rgba = (1.0, .55 * (1. - fraction), 0.0, .16 if plans["all_valid"][t, index] else .08)
      add_path(scn, path, .0015, rgba, dots=False)
  for path in plans["runner_gripper"][t]:
    add_path(scn, path, .003, RUNNER_UP, dots=False)
  if not fallback:
    add_path(scn, chosen, .007, CHOSEN)
    for q in plans["chosen_cube"][t][1:]:
      add_sphere(scn, q, .008, CUBE)
  for q in ep["gripper"][:t + 1:2]:
    add_sphere(scn, q, .006, TRAIL)


def caption(ep, t):
  r, last = ep["result"], len(ep["qpos"]) - 1
  lines = [f"{ep['folder'].parent.name} {ep['folder'].name}  step {t}/{last}",
           f"placed={r['task_success']}  final distance to B {r['final_goal_distance_cm']:.1f} cm"]
  if ep.get("label"):
    lines[0] = f"{ep['label']}  step {t}/{last}"
  plans = ep["plans"]
  if plans is not None and t < len(plans["score"]):
    lines.append(f"chosen plan score {plans['score'][t]:.3f}  valid candidates {int(plans['valid'][t])}")
    if "requested_rank" in plans:
      lines.append(f"requested rank {int(plans['requested_rank'][t])} within winning CEM round")
    if "fallback" in plans and plans["fallback"][t]:
      lines.append("REQUESTED PLAN INVALID: hold still, keep gripper command")
    if ep.get("show_all_paths", False):
      n = int((plans["all_round"][t] >= 0).sum())
      valid = int(plans["all_valid"][t].sum())
      lines.append(f"all CEM rounds: {n} proposals ({valid} valid); finite paths drawn")
      lines.append("green: chosen + top 5; orange/red: farther from chosen")
      lines.append("purple: imagined cube; blue: actual hand history")
  return lines


# A free camera above the table's front corner: the arm does not hide the imagined paths
def overview_camera():
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = [0.1, 0.0, 0.8]
  cam.distance, cam.azimuth, cam.elevation = 1.3, 160.0, -45.0
  return cam


def window(ep, speed):
  import mujoco.viewer
  data = mujoco.MjData(ep["model"])
  last = len(ep["qpos"]) - 1
  state = {"t": 0, "paused": False, "speed": speed}

  def on_key(key):
    if key == 32:
      state["paused"] = not state["paused"]
    elif key in (262, 263):
      state["t"] = int(np.clip(state["t"] + (1 if key == 262 else -1), 0, last))
      state["paused"] = True
    elif key in (265, 264):
      state["speed"] *= 2 if key == 265 else .5
    elif key == 82:
      state["t"], state["paused"] = 0, False

  try:
    viewer = mujoco.viewer.launch_passive(ep["model"], data, key_callback=on_key)
  except RuntimeError as e:
    raise SystemExit(f"{e}\nOn a Mac start the window with mjpython: ../.venv/bin/mjpython -m controller.replay ...")
  with viewer:
    while viewer.is_running():
      t = state["t"]
      data.qpos[:] = ep["qpos"][t]
      mujoco.mj_forward(ep["model"], data)
      with viewer.lock():
        viewer.user_scn.ngeom = 0
        draw(viewer.user_scn, ep, t)
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          "\n".join(caption(ep, t) + [f"speed x{state['speed']:g}" + ("  PAUSED" if state["paused"] else "")]), None))
      viewer.sync()
      time.sleep(.1 / state["speed"])
      if not state["paused"]:
        if t < last:
          state["t"] += 1
        else:
          state["paused"] = True


def video(ep, size, out, camera, every, speed=1.0):
  from PIL import Image
  # the saved model may default to a small framebuffer
  ep["model"].vis.global_.offheight = max(ep["model"].vis.global_.offheight, size)
  ep["model"].vis.global_.offwidth = max(ep["model"].vis.global_.offwidth, size)
  data = mujoco.MjData(ep["model"])
  renderer = mujoco.Renderer(ep["model"], size, size, max_geom=30000)
  cam = overview_camera() if camera == "overview" else camera
  frames = []
  # software rendering (OSMesa) is slow, so only every few steps are drawn
  steps = list(range(0, len(ep["qpos"]), every))
  if steps[-1] != len(ep["qpos"]) - 1:
    steps.append(len(ep["qpos"]) - 1)
  for k, t in enumerate(steps):
    if k % 25 == 0:
      print(f"rendering step {t} of {len(ep['qpos']) - 1}", flush=True)
    data.qpos[:] = ep["qpos"][t]
    mujoco.mj_forward(ep["model"], data)
    renderer.update_scene(data, camera=cam)
    draw(renderer.scene, ep, t)
    frame = np.ascontiguousarray(renderer.render())
    # the caption on a dark band, readable at small sizes
    lines = caption(ep, t)
    frame[:16 * len(lines) + 6] = (frame[:16 * len(lines) + 6] * .35).astype(np.uint8)
    for i, line in enumerate(lines):
      cv2.putText(frame, line, (6, 15 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    frames.append(frame)
  renderer.close()
  out = pathlib.Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  # speed only changes the playback, not the recorded actions
  gif = [Image.fromarray(f) for f in frames]
  gif[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif[1:], duration=max(10, round(100 * every / speed)), loop=0)
  writer = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 10 * speed / every, (size, size))
  for f in frames:
    writer.write(f[..., ::-1])
  writer.release()
  cv2.imwrite(str(out.with_suffix(".png")), frames[-1][..., ::-1])
  print(f"wrote {out.with_suffix('.gif')}, {out.with_suffix('.mp4')} and the last frame {out.with_suffix('.png')}")


def main():
  p = argparse.ArgumentParser(description="replay a recorded controller episode, with the planner's imagined plans")
  p.add_argument("episode", type=pathlib.Path, help="data/<run>/episodes/<method>/seed_<seed>")
  p.add_argument("--video", action="store_true", help="render a gif + mp4 instead of opening a window")
  p.add_argument("--out", type=pathlib.Path, default=None, help="video path without suffix (default <episode>/replay)")
  p.add_argument("--camera", default="overview", help="'overview' (free camera above the table) or 'static'")
  p.add_argument("--size", type=int, default=320, help="video frame size in pixels")
  p.add_argument("--every", type=int, default=2, help="video: draw every n-th step (1 = all, slower)")
  p.add_argument("--speed", type=float, default=1.0, help="window/video playback speed: 0.5 is half speed; does not change simulation")
  p.add_argument("--all-paths", action="store_true", help="draw every saved candidate from every CEM round; requires --save-all-plans at collection")
  p.add_argument("--models-run", default=None, help="only if the run's settings.json is missing")
  p.add_argument("--tag", default=None)
  args = p.parse_args()
  if not np.isfinite(args.speed) or args.speed <= 0 or args.every < 1:
    p.error("speed must be positive and finite; every must be at least 1")
  ep = load_episode(args.episode, args.models_run, args.tag)
  if args.all_paths and (ep["plans"] is None or "all_gripper" not in ep["plans"]):
    p.error("This recording saved only the chosen path and runner-ups. Run a NEW episode with "
            "controller.control_pipeline --save-all-plans in a separate --out folder first.")
  ep["show_all_paths"] = args.all_paths
  print(f"{len(ep['qpos']) - 1} steps, placed={ep['result']['task_success']}, "
        f"{'with' if ep['plans'] is not None else 'without'} imagined plans")
  if args.video:
    video(ep, args.size, args.out or args.episode / "replay", args.camera, args.every, args.speed)
  else:
    window(ep, args.speed)


if __name__ == "__main__":
  main()
