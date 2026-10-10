import sys

# offscreen rendering (EGL on Linux) only for --video; a window needs GLFW instead, so this is decided
# before mujoco is imported. On the KTH notebook export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa first.
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

# Replay a recorded controller episode (any method) from its folder, with what the planner imagined drawn on
# top for mpc episodes. Needs no GPU and no world model: it rebuilds the scene from the episode's layout and
# sets the saved simulator state of every step (trajectory.npz "qpos", recorded since mpc was added).
#   window, on a Mac:  ../.venv/bin/mjpython -m controller.replay data/<run>/episodes/mpc/seed_20494010
#   window, on Linux:  python -m controller.replay data/<run>/episodes/mpc/seed_20494010
#   video, anywhere:   python -m controller.replay data/<run>/episodes/mpc/seed_20494010 --video
# Drawn: orange thick = the plan the planner chose at this step (imagined gripper path), orange faint = the
# runner-up plans, purple = where the planner imagined the cube along its chosen plan, blue = where the
# gripper actually went. The green disc on the table is the target B. Window keys: space pause / resume,
# right / left one step (pauses), up / down twice / half the speed, R from the start.

CHOSEN = (1.0, 0.45, 0.0, 0.95)
RUNNER_UP = (1.0, 0.7, 0.25, 0.35)
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
    models_run, tag = models_run or run["models_run"], tag or run["tag"]
  models_run, tag = pathlib.Path(models_run or "data/combined_test1"), tag or "combined"
  cfg = Config.nested(json.loads((models_run / "attempts" / tag / "info.json").read_text())["config"])
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
  for q in ep["gripper"][:t + 1:2]:
    add_sphere(scn, q, .006, TRAIL)
  plans = ep["plans"]
  if plans is None or t >= len(plans["chosen_gripper"]):
    return
  for path in plans["runner_gripper"][t]:
    add_path(scn, path, .004, RUNNER_UP, dots=False)
  add_path(scn, plans["chosen_gripper"][t], .007, CHOSEN)
  for q in plans["chosen_cube"][t][1:]:
    add_sphere(scn, q, .008, CUBE)


def caption(ep, t):
  r, last = ep["result"], len(ep["qpos"]) - 1
  lines = [f"{ep['folder'].parent.name} {ep['folder'].name}  step {t}/{last}",
           f"placed={r['task_success']}  final distance to B {r['final_goal_distance_cm']:.1f} cm"]
  plans = ep["plans"]
  if plans is not None and t < len(plans["score"]):
    lines.append(f"chosen plan score {plans['score'][t]:.3f}  valid candidates {int(plans['valid'][t])}")
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


def video(ep, size, out, camera, every):
  from PIL import Image
  data = mujoco.MjData(ep["model"])
  renderer = mujoco.Renderer(ep["model"], size, size)
  cam = overview_camera() if camera == "overview" else camera
  frames = []
  # software rendering on the notebook (OSMesa) is slow, so only every few steps are drawn
  steps = range(0, len(ep["qpos"]), every)
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
  # a gif plays inline in Jupyter; the mp4 (MPEG-4) plays in QuickTime / VLC after download; both in real time
  gif = [Image.fromarray(f) for f in frames]
  gif[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif[1:], duration=100 * every, loop=0)
  writer = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 10 / every, (size, size))
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
  p.add_argument("--speed", type=float, default=1.0, help="window playback speed, 1 = real time (10 steps per second)")
  p.add_argument("--models-run", default=None, help="only if the run's settings.json is missing")
  p.add_argument("--tag", default=None)
  args = p.parse_args()
  ep = load_episode(args.episode, args.models_run, args.tag)
  print(f"{len(ep['qpos']) - 1} steps, placed={ep['result']['task_success']}, "
        f"{'with' if ep['plans'] is not None else 'without'} imagined plans")
  if args.video:
    video(ep, args.size, args.out or args.episode / "replay", args.camera, max(1, args.every))
  else:
    window(ep, args.speed)


if __name__ == "__main__":
  main()
