# offscreen camera rendering (EGL), has to be imported before mujoco
import offscreen

import argparse
import copy
import json
import pathlib

from world_model.common import SIMULATION, write_json
from . import control_pipeline as cp
from . import replay

# What happens when the MPC executes a worse plan than its best: the scene of a recorded mpc episode is run
# again executing the 5th, 25th and 64th best candidate at every step, and once the best for a while then the
# 64th. Real closed-loop episodes, only the chosen rank changes. A gif per variant lands in data/<out>/gifs/.
#   python -u -m controller.mpc_counterfactual --source-episode data/<run>/episodes/mpc/seed_<seed>

CASES = (("rank_05", 5, None, 5), ("rank_25", 25, None, 25), ("rank_64", 64, None, 64), ("best_60_then_64", 1, 60, 64))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--source-episode", type=pathlib.Path, required=True, help="a recorded mpc episode, data/<run>/episodes/mpc/seed_<seed>")
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/mpc_counterfactual"))
  p.add_argument("--size", type=int, default=512, help="gif frame size")
  p.add_argument("--speed", type=float, default=.5, help="gif playback speed, 0.5 is half speed")
  args = p.parse_args()
  source, args.out = args.source_episode.resolve(), args.out.resolve()
  if SIMULATION / "data" not in args.out.parents:
    p.error("keep --out under simulation/data/")
  result = json.loads((source / "result.json").read_text())
  if result["method"] != "mpc":
    p.error("the source has to be an mpc episode")

  # the flags and mpc knobs the episode saved, frozen into a config file of their own
  settings = json.loads((source.parents[2] / "settings.json").read_text())
  knobs = json.loads((source / "mpc_settings.json").read_text())
  args.out.mkdir(parents=True, exist_ok=True)
  config = args.out / "mpc_config.py"
  config.write_text("\n".join(f"{k} = {v!r}" for k, v in knobs.items() if k.isupper() and k != "CONFIG_FILE") + "\n")
  if knobs["POPULATION"] < max(case[1] for case in CASES):
    p.error(f"the source searched {knobs['POPULATION']} candidates per round, rank 64 needs at least 64")
  options = argparse.Namespace(**settings)
  options.headless, options.save_all_plans = True, True
  options.models_run, options.mpc_config = pathlib.Path(settings["models_run"]), config

  import torch
  torch.set_num_threads(options.threads)
  cfg, rest_z = cp.run_settings(options.models_run, settings["tag"])
  models = cp.WorldModels(options.models_run, settings["tag"], options)
  rows = []
  for label, rank, after, then in CASES:
    root = args.out / label
    folder = root / "episodes" / "mpc" / f"seed_{result['seed']}"
    if not (folder / "result.json").exists():
      options.mpc_rank, options.mpc_rank_after, options.mpc_rank_then = rank, after, then
      print(f"RUN {label}: rank {rank}" + (f", rank {then} after step {after}" if after else ""), flush=True)
      cp.episode(root, "mpc", result["seed"], copy.deepcopy(cfg), copy.deepcopy(result["layout"]), rest_z, options, models=models)
    r = json.loads((folder / "result.json").read_text())
    rows.append({"variant": label, "placed": r["task_success"], "steps": r["steps"],
                 "final_goal_distance_cm": r["final_goal_distance_cm"], "obstacle_contact_steps": r["obstacle_contact_steps"],
                 "fallback_steps": r["planner_fallback_steps"]})
    ep = replay.load_episode(folder, str(options.models_run), settings["tag"])
    ep["show_all_paths"], ep["label"] = True, label
    replay.video(ep, args.size, args.out / "gifs" / label, "overview", 1, args.speed)
    write_json(args.out / "summary.json", {"source": {"episode": str(source), "placed": result["task_success"], "steps": result["steps"],
                                                      "final_goal_distance_cm": result["final_goal_distance_cm"]}, "variants": rows})
  for row in rows:
    print(f"{row['variant']}: placed={row['placed']} steps={row['steps']} final distance {row['final_goal_distance_cm']:.1f} cm "
          f"contacts={row['obstacle_contact_steps']} fallbacks={row['fallback_steps']}", flush=True)
  print(f"DONE: gifs in {args.out / 'gifs'}", flush=True)


if __name__ == "__main__":
  main()
