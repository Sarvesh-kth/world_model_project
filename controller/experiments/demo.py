"""Watch one agent do one episode on the Grade E scene: the manual test for M3 Level E.

    .venv/bin/python -m controller.experiments.demo --agent oracle --task place --seed 0
    .venv/bin/mjpython -m controller.experiments.demo --agent oracle --viewer      # macOS live viewer
    .venv/bin/python -m controller.experiments.demo --agent jepa --max-steps 100   # needs the JEPA pipeline

Agents: random, scripted (M1's expert), oracle (CEM + simulator), state_mlp (CEM + learned state
model), jepa (CEM + V-JEPA 2 + D, the Level E deliverable). Prints progress every few steps and
saves an MP4 of the static camera to data/runs/demo/<agent>_<task>_seed<seed>.mp4.
"""

import argparse
import time

import cv2
import numpy as np

from controller.adapters.m1_adapter import grade_e_adapter
from controller.config import Paths
from controller.costs import STAGE_COSTS
from controller.eval import task_success


def make_agent(name, task, seed):
    from controller import agents

    if name == "random":
        return agents.RandomAgent(seed)
    if name == "scripted":
        return agents.ScriptedAgent()
    if name == "oracle":
        from controller.experiments.oracle_cem import ORACLE_CFG

        return agents.OracleCEMAgent(STAGE_COSTS[task], ORACLE_CFG, seed=seed)
    if name == "state_mlp":
        from controller.experiments.oracle_cem import ORACLE_CFG
        from controller.experiments.state_mlp import MODEL_PATH

        return agents.StateMLPAgent(str(MODEL_PATH), STAGE_COSTS[task], ORACLE_CFG, seed=seed)
    if name == "jepa":
        from controller.adapters.m2_adapter import JEPAEncoder
        from controller.experiments.jepa_cem import CHECKPOINT, JEPA_CFG

        return agents.JEPACEMAgent(str(CHECKPOINT), JEPAEncoder(), JEPA_CFG, seed=seed)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="oracle", choices=["random", "scripted", "oracle", "state_mlp", "jepa"])
    p.add_argument("--task", default="place", choices=list(STAGE_COSTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--viewer", action="store_true", help="live MuJoCo viewer (on macOS run with .venv/bin/mjpython)")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    adapter = grade_e_adapter()
    obs, info = adapter.reset(seed=args.seed)
    agent = make_agent(args.agent, args.task, args.seed)
    agent.reset(adapter, obs)
    print(f"{args.agent} on '{args.task}', seed {args.seed}: {info['layout']}")

    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(adapter.env.model, adapter.env.data)
    frames = [] if args.no_video else [adapter.render("static")]
    t0, success = time.time(), False
    for t in range(1, args.max_steps + 1):
        obs, _, terminated, truncated, info = adapter.step(agent.act(adapter, obs))
        f = adapter.task_features()
        if frames is not None and not args.no_video:
            frames.append(adapter.render("static"))
        if viewer is not None:
            viewer.sync()
        if t % 10 == 0 or info["success"] or terminated or truncated:
            print(f"step {t:3d}  gripper {np.round(f[0:3], 3)}  cube {np.round(f[3:6], 3)}  "
                  f"grasped {int(f[9])}  stage {info['stage']:9s}  [{time.time() - t0:5.0f} s]", flush=True)
        if task_success(args.task, f, info):
            success = True
            break
        if terminated or truncated:
            break
    print(f"{'SUCCESS' if success else 'no success'} after {t} steps")
    if frames and not args.no_video:
        path = Paths().runs_dir / "demo" / f"{args.agent}_{args.task}_seed{args.seed}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for fr in frames:
            video.write(fr[..., ::-1])
        video.release()
        print(f"video: {path}")
    if viewer is not None:
        viewer.close()
    adapter.close()


if __name__ == "__main__":
    main()
