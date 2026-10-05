"""Exact-state A-to-B SAC baseline; train, compare BC, evaluate, save videos, export."""
import argparse
import json
import pathlib
import shutil
import sys
from types import SimpleNamespace

import numpy as np

from environment.config import load_config
from .data import digest, provenance, write_json
from .pipeline import SIMULATION, run_command
from .rl_control import train, validate


def export(root, target, state):
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "run_info.json", state)
    for name in ("summary.json", "evaluations.json", "demos.json", "rl_monitor.csv",
                 "models/rl_training.json"):
        source = root / name
        if source.exists():
            shutil.copyfile(source, target / source.name)
    for folder in ("logs", "episodes"):
        for source in (root / folder).rglob("*"):
            if source.is_file() and source.suffix in (".csv", ".json", ".txt"):
                destination = target / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    write_json(target / "files.json", {str(p.relative_to(target)): digest(p)
        for p in sorted(target.rglob("*")) if p.is_file() and p.name != "files.json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/grade_e.yml"))
    parser.add_argument("--layout", type=pathlib.Path, default=pathlib.Path("configs/grade_e_layout.json"))
    parser.add_argument("--seed", type=int, default=20263005)
    parser.add_argument("--demos", type=int, default=100)
    parser.add_argument("--bc-epochs", type=int, default=240)
    parser.add_argument("--bc-weight", type=float, default=100)
    parser.add_argument("--critic-warmup", type=int, default=2000)
    parser.add_argument("--rl-steps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--ent-coef", default="auto_0.005")
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--video-episodes", type=int, default=5)
    parser.add_argument("--position-jitter", type=float, default=.01)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("train",), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (min(args.demos, args.bc_epochs, args.rl_steps, args.eval_every, args.threads,
            args.validation_episodes, args.test_episodes) < 1 or args.seed < 0
            or not 0 <= args.position_jitter <= .03 or args.critic_warmup < 0
            or not 0 <= args.bc_weight <= 1000 or not 0 < args.learning_rate <= .001
            or not 0 <= args.video_episodes <= args.test_episodes):
        parser.error("invalid training/evaluation settings")
    args.out, args.config, args.layout = [p.resolve() for p in (args.out, args.config, args.layout)]
    if SIMULATION / "data" not in args.out.parents:
        parser.error("keep a fresh run under simulation/data/")
    target = SIMULATION.parent / "results" / args.out.name
    cfg = load_config(args.config, overrides={"episode": {"terminate_on_success": False}})
    layout = json.loads(args.layout.read_text())
    if layout["object"] != "cube" or layout["scale"] != 1 or layout["obstacles"]:
        parser.error("baseline supports a unit cube and empty table")
    if not cfg.control.yaw.enabled:
        parser.error("baseline needs the existing 5-action interface")
    if args.stage:
        train(args.out, cfg, layout, args)
        return
    code = list((SIMULATION / "environment").glob("*.py"))
    code += [SIMULATION / "world_model/vision" / (name+".py") for name in
             ("rl_baseline", "rl_control", "task_control", "control_pipeline", "data", "pipeline")]
    code += [SIMULATION / "data_collection/scripted_policy.py", SIMULATION / "data_collection/route.py"]
    options = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()
               if k not in ("resume", "stage")}
    signature = {"config": cfg, "layout": layout, "options": options,
        "code_sha256": {str(p.relative_to(SIMULATION)): digest(p) for p in code},
        "gate": "SAC placement >=90% over >=100 fresh episodes; >=20 validation episodes; no JEPA",
        "test_seed_start": args.seed+1000000,
        "training": "Demonstration-assisted SB3 SAC; real reward-based updates; not from scratch"}
    path = args.out / "pipeline.json"
    if path.exists():
        state = json.loads(path.read_text())
        if not args.resume or state["inputs"] != signature:
            parser.error("existing run: exact --resume required, otherwise choose a new --out")
    else:
        if ((args.out.exists() and any(args.out.iterdir()))
                or (target.exists() and any(target.iterdir()))):
            parser.error("output/export exists; choose a fresh run name")
        state = {"inputs": signature, "provenance": provenance(), "complete": False}
        write_json(path, state)
    try:
        if not state.get("trained"):
            command = [sys.executable, "-u", "-m", "world_model.vision.rl_baseline", "--stage", "train"]
            for key, value in options.items():
                command += ["--"+key.replace("_", "-"), str(value)]
            run_command(command, args.out / "logs/train.txt")
            state["trained"] = {name: digest(args.out / "models" / name)
                for name in ("sac_best.zip", "bc_initial.zip", "rl_training.json")}
            write_json(path, state)
        for name, sha in state["trained"].items():
            if digest(args.out / "models" / name) != sha:
                raise ValueError(f"frozen training output changed: {name}")
        from stable_baselines3 import SAC
        import torch
        torch.set_num_threads(args.threads)
        training = json.loads((args.out / "models/rl_training.json").read_text())
        evaluation_path = args.out / "evaluations.json"
        if state.get("evaluations_sha256"):
            if digest(evaluation_path) != state["evaluations_sha256"]:
                raise ValueError("saved evaluations changed")
            evaluations = json.loads(evaluation_path.read_text())
        else:
            evaluations = {}
            test_args = SimpleNamespace(seed=args.seed+990000, position_jitter=args.position_jitter,
                                        validation_episodes=args.test_episodes)
            for method, checkpoint in (("bc_true", "bc_initial.zip"), ("rl_true", "sac_best.zip")):
                print(f"FROZEN EVALUATION {method}: {args.test_episodes} fresh episodes", flush=True)
                policy = SAC.load(args.out / "models" / checkpoint, device="cpu")
                evaluations[method] = validate(policy, cfg, layout, test_args)
                print(f"{method}: {evaluations[method]['successes']}/{args.test_episodes}", flush=True)
            write_json(evaluation_path, evaluations)
            state["evaluations_sha256"] = digest(evaluation_path)
            write_json(path, state)
        passed = (args.validation_episodes >= 20 and args.test_episodes >= 100
                  and evaluations["rl_true"]["success_rate"] >= .9)
        summary = {"gate_passed": passed, "complete": False, "inputs": "exact current simulator state + B",
            "BC_test_successes": evaluations["bc_true"]["successes"],
            "SAC_test_successes": evaluations["rl_true"]["successes"], "test_episodes": args.test_episodes,
            "SAC_checkpoint_sha256": state["trained"]["sac_best.zip"],
            "selected_rl_steps": training["best"]["training_steps"],
            "selected_validation_success_rate": training["best"]["success_rate"],
            "note": "Imitation and RL are reported separately. Passing does not prove an RL gain or broad robustness."}
        write_json(args.out / "summary.json", summary)
        # ponytail: reuse the existing exact-state episode/video writer; no second physics loop.
        from .control_pipeline import episode
        episode_signature = {"config": cfg, "layout": layout, "known_rest_z": 0, "encoder": None}
        for method, checkpoint in (("scripted", None), ("bc_true", "bc_initial.zip"), ("rl_true", "sac_best.zip")):
            for i in range(args.video_episodes):
                seed = args.seed+1000000+i
                result = args.out / "episodes" / method / f"seed_{seed}" / "result.json"
                if result.exists():
                    saved = json.loads(result.read_text())
                    if method != "scripted" and saved["task_success"] != evaluations[method]["episodes"][i]["task_success"]:
                        raise ValueError("video outcome differs from frozen evaluation")
                    continue
                video_args = SimpleNamespace(**vars(args), method=method, episode_seed=seed,
                    policy_checkpoint=args.out / "models" / checkpoint if checkpoint else None)
                episode(args.out, episode_signature, video_args)
                if method != "scripted" and json.loads(result.read_text())["task_success"] != evaluations[method]["episodes"][i]["task_success"]:
                    raise ValueError("video outcome differs from frozen evaluation")
        summary["complete"] = state["complete"] = True
        write_json(args.out / "summary.json", summary)
        write_json(path, state)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        state["last_error"] = str(error)
        write_json(path, state)
        raise
    finally:
        export(args.out, target, state)
    print(f"COMPLETE: {target}; videos/checkpoints remain under {args.out}", flush=True)


if __name__ == "__main__":
    main()
