# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import json
import pathlib
from types import SimpleNamespace

from environment.config import load_config, Config
from environment.rewards import fill_weights
from world_model.common import SIMULATION, write_json
from .rl_control import train, validate

# Train the exact-state SAC baseline and evaluate it on fresh seeds. Two ways to use it:
#   one fixed layout (the old baseline): --layout, cube start jittered, optionally --obstacles low high
#     fresh obstacles per episode with the 8x8 height map in the observation
#   the world model's scene distribution (--random-ends): cube and goal anywhere on the reachable table,
#     --obstacles low high obstacles in between, the config and resting height of the models' own run
#     (--models-run), the 16x16 obstacle-only map in the observation (--obstacle-map), the recorded
#     episodes in the replay buffer (--recorded-run). The policy is run on Q's cube estimate afterwards
#     (controller rl_q), so --cube-noise jitters the cube it sees in training.
# The run folder is what the controllers and the collectors load afterwards:
#   models/sac_best.zip     the frozen policy
#   models/bc_initial.zip   the behaviour cloned actor before RL, for comparison
#   models/rl_training.json validation history and settings
#   pipeline.json           config, layout and options the policy was trained with
#   evaluations.json        bc_true and rl_true on --test-episodes fresh seeds
#   summary.json            the headline numbers
#   python -m rl.rl_baseline --out data/rl_baseline_v1
#   python -m rl.rl_baseline --out data/sac_priv_v1 --random-ends --obstacles 0 3 --obstacle-map window --recorded-run data/episodes_v3


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/rl_baseline_v1"))
  p.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/grade_e.yml"))
  p.add_argument("--layout", type=pathlib.Path, default=pathlib.Path("configs/grade_e_layout.json"))
  p.add_argument("--models-run", type=pathlib.Path, default=pathlib.Path("models/v3c"),
                 help="with --random-ends: the packaged model folder whose info.json config the scenes are sampled with")
  p.add_argument("--random-ends", action="store_true", help="a whole new scene per episode from the environment's own sampler")
  p.add_argument("--obstacle-map", choices=("full", "window", "both"), default=None,
                 help="the 16x16 obstacle-only depth map in the observation instead of the 8x8 height map: the whole map (256 numbers), the 9x9 window around the hand (83) or the window plus the map pooled to 8x8 (147)")
  p.add_argument("--recorded-run", type=pathlib.Path, default=None, help="a run folder (manifest.json) whose episodes fill the replay buffer")
  p.add_argument("--cube-noise", type=float, default=0.0, help="std of the noise on the cube position the policy sees in training, metres")
  p.add_argument("--pull", type=float, default=0.0, help="the planner's linear distance pull as a potential term of the training reward, per metre")
  p.add_argument("--net", type=int, nargs="+", default=[128, 128])
  p.add_argument("--seed", type=int, default=20263005)
  p.add_argument("--demos", type=int, default=100, help="successful scripted episodes to clone (0: only the recorded run)")
  p.add_argument("--bc-epochs", type=int, default=240)
  p.add_argument("--bc-weight", type=float, default=100, help="weight of the BC term in every actor update")
  p.add_argument("--critic-warmup", type=int, default=2000, help="critic updates on the demos before RL")
  p.add_argument("--rl-steps", type=int, default=20000)
  p.add_argument("--learning-rate", type=float, default=3e-5)
  p.add_argument("--ent-coef", default="auto_0.005")
  p.add_argument("--eval-every", type=int, default=2500)
  p.add_argument("--validation-episodes", type=int, default=20)
  p.add_argument("--validation-empty", type=int, default=None,
                 help="of the validation (and test) episodes, this many are empty tables, the rest get --validation-obstacles")
  p.add_argument("--validation-obstacles", type=int, nargs=2, default=[1, 3], metavar=("LOW", "HIGH"))
  p.add_argument("--test-episodes", type=int, default=100)
  p.add_argument("--position-jitter", type=float, default=.01, help="cube start jitter per xy axis in metres")
  p.add_argument("--obstacles", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"),
                 help="fresh obstacles per episode and a map in the observation, e.g. 0 3")
  p.add_argument("--safety", type=float, default=1.0,
                 help="multiplier on the collision, proximity and table-hit reward weights during training")
  p.add_argument("--threads", type=int, default=4)
  args = p.parse_args()
  args.out = args.out.resolve()
  if SIMULATION / "data" not in args.out.parents:
    p.error("keep the run under simulation/data/")
  if args.obstacle_map and args.obstacles is None:
    p.error("--obstacle-map needs --obstacles")

  # the session ends episodes itself, so the env must not stop early on its own success event
  if args.random_ends:
    # the models' own config (full_mix: grey cube, random size and mass), the way the controller benchmark samples scenes
    info = json.loads((args.models_run / "info.json").read_text())
    base = Config.nested(info["config"])
    fill_weights(base.rewards.weights)
  else:
    base = load_config(args.config)
  weights = {k: float(base.rewards.weights[k] * args.safety) for k in ("collision", "proximity", "table_hit")}
  overrides = {"episode": {"terminate_on_success": False}, "rewards": {"weights": weights}}
  if args.random_ends:
    cfg = json.loads(json.dumps(base))
    cfg["episode"].update(overrides["episode"])
    cfg["rewards"]["weights"].update(weights)
  else:
    cfg = load_config(args.config, overrides=overrides)
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
                              validation_episodes=args.test_episodes, obstacles=args.obstacles,
                              obstacle_map=args.obstacle_map, random_ends=args.random_ends,
                              validation_empty=args.validation_empty, validation_obstacles=args.validation_obstacles)
  for method, checkpoint in (("bc_true", "bc_initial.zip"), ("rl_true", "sac_best.zip")):
    print(f"evaluating {method} on {args.test_episodes} fresh episodes", flush=True)
    policy = SAC.load(args.out / "models" / checkpoint, device="cpu")
    evaluations[method] = validate(policy, cfg, layout, test_args)
    print(f"{method}: {evaluations[method]['successes']}/{args.test_episodes} (empty {evaluations[method]['empty_successes']}, "
          f"obstacles {evaluations[method]['obstacle_successes']}), "
          f"{evaluations[method]['obstacle_contact_steps']} contact steps", flush=True)
  write_json(args.out / "evaluations.json", evaluations)

  training = json.loads((args.out / "models/rl_training.json").read_text())
  summary = {"BC_test_successes": evaluations["bc_true"]["successes"],
             "SAC_test_successes": evaluations["rl_true"]["successes"], "test_episodes": args.test_episodes,
             "SAC_test_contact_steps": evaluations["rl_true"]["obstacle_contact_steps"],
             "selected_rl_steps": training["best"]["training_steps"],
             "selected_validation_success_rate": training["best"]["success_rate"]}
  write_json(args.out / "summary.json", summary)
  write_json(args.out / "pipeline.json", {"inputs": {"config": cfg, "layout": layout, "options": options},
                                          "complete": True})
  print(json.dumps(summary, indent=2))
  print(f"DONE: {args.out}; on the cameras: python -m controller.control_pipeline --methods rl_q "
        f"--baseline-run {args.out} --random-ends --obstacles 0 3 --headless")


if __name__ == "__main__":
  main()
