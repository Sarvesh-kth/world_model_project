# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import json
import pathlib
from types import SimpleNamespace

from environment.config import load_config
from world_model.common import SIMULATION, write_json
from .rl_control import train, validate

# Train the exact-state SAC baseline on one fixed layout and evaluate it on fresh seeds.
# The run folder is what the controllers and the collectors load afterwards:
#   models/sac_best.zip     the frozen policy
#   models/bc_initial.zip   the behaviour cloned actor before RL, for comparison
#   models/rl_training.json validation history and settings
#   pipeline.json           config, layout and options the policy was trained with
#   evaluations.json        bc_true and rl_true on --test-episodes fresh seeds
#   summary.json            the headline numbers
#   python -m rl.rl_baseline --out data/rl_baseline_v1


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
  p.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/grade_e.yml"))
  p.add_argument("--layout", type=pathlib.Path, default=pathlib.Path("configs/grade_e_layout.json"))
  p.add_argument("--seed", type=int, default=20263005)
  p.add_argument("--demos", type=int, default=100, help="successful scripted episodes to clone")
  p.add_argument("--bc-epochs", type=int, default=240)
  p.add_argument("--bc-weight", type=float, default=100, help="weight of the BC term in every actor update")
  p.add_argument("--critic-warmup", type=int, default=2000, help="critic updates on the demos before RL")
  p.add_argument("--rl-steps", type=int, default=20000)
  p.add_argument("--learning-rate", type=float, default=3e-5)
  p.add_argument("--ent-coef", default="auto_0.005")
  p.add_argument("--eval-every", type=int, default=2500)
  p.add_argument("--validation-episodes", type=int, default=20)
  p.add_argument("--test-episodes", type=int, default=100)
  p.add_argument("--position-jitter", type=float, default=.01, help="cube start jitter per xy axis in metres")
  p.add_argument("--threads", type=int, default=4)
  args = p.parse_args()
  args.out = args.out.resolve()
  if SIMULATION / "data" not in args.out.parents:
    p.error("keep the run under simulation/data/")

  # the session ends episodes itself, so the env must not stop early on its own success event
  cfg = load_config(args.config, overrides={"episode": {"terminate_on_success": False}})
  layout = json.loads(args.layout.read_text())
  options = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}
  args.out.mkdir(parents=True, exist_ok=True)
  write_json(args.out / "pipeline.json", {"inputs": {"config": cfg, "layout": layout, "options": options},
                                          "complete": False})

  train(args.out, cfg, layout, args)

  # evaluate the cloned actor and the final policy on seeds nothing was trained or validated on
  from stable_baselines3 import SAC
  import torch
  torch.set_num_threads(args.threads)
  evaluations = {}
  test_args = SimpleNamespace(seed=args.seed + 990000, position_jitter=args.position_jitter,
                              validation_episodes=args.test_episodes)
  for method, checkpoint in (("bc_true", "bc_initial.zip"), ("rl_true", "sac_best.zip")):
    print(f"evaluating {method} on {args.test_episodes} fresh episodes", flush=True)
    policy = SAC.load(args.out / "models" / checkpoint, device="cpu")
    evaluations[method] = validate(policy, cfg, layout, test_args)
    print(f"{method}: {evaluations[method]['successes']}/{args.test_episodes}", flush=True)
  write_json(args.out / "evaluations.json", evaluations)

  training = json.loads((args.out / "models/rl_training.json").read_text())
  summary = {"BC_test_successes": evaluations["bc_true"]["successes"],
             "SAC_test_successes": evaluations["rl_true"]["successes"], "test_episodes": args.test_episodes,
             "selected_rl_steps": training["best"]["training_steps"],
             "selected_validation_success_rate": training["best"]["success_rate"]}
  write_json(args.out / "summary.json", summary)
  write_json(args.out / "pipeline.json", {"inputs": {"config": cfg, "layout": layout, "options": options},
                                          "complete": True})
  print(json.dumps(summary, indent=2))
  print(f"DONE: {args.out}; videos of the policy: python -m controller.control_pipeline --methods rl_true "
        f"--baseline-run {args.out}")


if __name__ == "__main__":
  main()
