"""Watch an agent work in M1's MuJoCo scene, with what its planner imagines drawn on top.

Live viewer (like M1's play.py; on macOS it needs mjpython):
    .venv/bin/mjpython -m controller.experiments.visualize --agent oracle --viewer
Video instead (any Python; written to data/runs/visualize/):
    .venv/bin/python -m controller.experiments.visualize --agent jepa --max-steps 120

Agents:
    scripted    M1's expert (reads the true sim state); nothing to imagine, so no plan drawn
    oracle      CEM, planning with the simulator itself as the world model (slow: 2-6 s per plan)
    state_mlp   CEM with the learned state model (needs: python -m controller.experiments.state_mlp)
    jepa        CEM with V-JEPA 2 + the JEPA dynamics model: the Level E deliverable (~1 s per plan)
    random      random actions

Drawn on the scene (see controller/visual.py):
    orange, thick  the plan CEM chose: imagined gripper path      orange, faint  runner-up (elite) plans
    blue dots      where the gripper actually went                green ball     goal gripper position (jepa)
"""

import argparse
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from controller.adapters.m1_adapter import grade_e_adapter
from controller.config import Paths
from controller.costs import STAGE_COSTS
from controller.eval import task_success
from controller.visual import PlanOverlay, put_text


def make_agent(name, task, seed):
    from controller import agents
    from controller.experiments.oracle_cem import ORACLE_CFG

    if name == "random":
        return agents.RandomAgent(seed)
    if name == "scripted":
        return agents.ScriptedAgent()
    if name == "oracle":
        return agents.OracleCEMAgent(STAGE_COSTS[task], ORACLE_CFG, seed=seed)
    if name == "state_mlp":
        from controller.experiments.state_mlp import MODEL_PATH

        return agents.StateMLPAgent(str(MODEL_PATH), STAGE_COSTS[task], ORACLE_CFG, seed=seed)
    if name == "jepa":
        from controller.adapters.m2_adapter import JEPAEncoder
        from controller.experiments.jepa_cem import CHECKPOINT, JEPA_CFG

        return agents.JEPACEMAgent(str(CHECKPOINT), JEPAEncoder(), JEPA_CFG, seed=seed)
    raise ValueError(name)


def overview_camera():
    """A free camera above the table's front corner: the arm doesn't hide the imagined paths."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.1, 0.0, 0.8]
    cam.distance, cam.azimuth, cam.elevation = 1.3, 160.0, -45.0
    return cam


def status_lines(args, t, f, info, plan_s, overlay):
    lines = [f"{args.agent} | task {args.task} | seed {args.seed} | step {t}",
             f"stage {info.get('stage', '-')} | grasped {int(f[9])} | gripper {'open' if f[10] else 'closed'}"]
    if plan_s is not None:
        lines.append(f"last plan {plan_s:.1f} s" + (" | orange = imagined plan" if overlay.plan is not None else ""))
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="oracle", choices=["scripted", "oracle", "state_mlp", "jepa", "random"])
    p.add_argument("--task", default="place", choices=list(STAGE_COSTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--viewer", action="store_true", help="live MuJoCo viewer (macOS: run with .venv/bin/mjpython)")
    p.add_argument("--speed", type=float, default=1.0, help="live viewer pace, 1.0 = real time (10 steps/s)")
    p.add_argument("--view", default="overview", choices=["overview", "static"], help="video camera")
    p.add_argument("--size", type=int, default=480, help="video frame size in pixels")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-overlay", action="store_true")
    args = p.parse_args()

    adapter = grade_e_adapter()
    obs, reset_info = adapter.reset(seed=args.seed)
    agent = make_agent(args.agent, args.task, args.seed)
    try:
        agent.reset(adapter, obs)
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(f"could not load the {args.agent} model ({type(e).__name__}). For state_mlp run "
                         f"`python -m controller.experiments.state_mlp`, for jepa `... jepa_pipeline` first.") from e
    overlay = PlanOverlay()
    print(f"{args.agent} on '{args.task}', seed {args.seed}: {reset_info['layout']}")
    print("orange thick = chosen plan (imagined gripper path), orange faint = runner-up plans, "
          "blue = executed path, green = goal gripper position")

    viewer = None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(adapter.env.model, adapter.env.data)
    renderer = None if args.no_video else mujoco.Renderer(adapter.env.model, args.size, args.size)
    camera = overview_camera() if args.view == "overview" else "static"
    frames, info, plan_s, success = [], {}, None, False

    def show(t):
        f = adapter.task_features()
        lines = status_lines(args, t, f, info, plan_s, overlay)
        if viewer is not None:
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if not args.no_overlay:
                    overlay.draw(viewer.user_scn)
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "\n".join(lines), None))
            viewer.sync()
        if renderer is not None:
            renderer.update_scene(adapter.env.data, camera=camera)
            if not args.no_overlay:
                overlay.draw(renderer.scene)
            frames.append(put_text(renderer.render(), lines))

    overlay.update(agent, adapter.task_features()[0:3])
    show(0)
    t0 = time.time()
    for t in range(1, args.max_steps + 1):
        if viewer is not None and not viewer.is_running():
            break
        step_start = time.time()
        action = agent.act(adapter, obs)
        took = time.time() - step_start
        if took > 0.05:
            plan_s = took  # a replan happened in this step
        obs, _, terminated, truncated, info = adapter.step(action)
        f = adapter.task_features()
        overlay.update(agent, f[0:3])
        show(t)
        if t % 10 == 0 or info["success"] or terminated or truncated:
            print(f"step {t:3d}  gripper {np.round(f[0:3], 3)}  cube {np.round(f[3:6], 3)}  grasped {int(f[9])}  "
                  f"stage {info['stage']:9s}  [{time.time() - t0:5.0f} s]", flush=True)
        if task_success(args.task, f, info):
            success = True
            break
        if terminated or truncated:
            break
        if viewer is not None:
            time.sleep(max(0.0, 0.1 / args.speed - (time.time() - step_start)))
    print(f"{'SUCCESS' if success else 'no success'} after {t} steps")

    if frames:
        path = Paths().runs_dir / "visualize" / f"{args.agent}_{args.task}_seed{args.seed}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (args.size, args.size))
        for fr in frames:
            video.write(fr[..., ::-1])
        video.release()
        cv2.imwrite(str(path.with_suffix(".png")), frames[-1][..., ::-1])
        print(f"video: {path}  (last frame: {path.with_suffix('.png').name})")
    if viewer is not None:
        print("episode over; close the viewer window to exit")
        while viewer.is_running():
            time.sleep(0.1)
        viewer.close()
    adapter.close()


if __name__ == "__main__":
    main()
