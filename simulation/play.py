import argparse
import json
import os
import sys
import time
import numpy as np

# NVIDIA PRIME offload variables (often in ~/.bashrc on hybrid laptops) leave the viewer window blank,
# the viewer renders fine on the default GPU, so drop them for this process only
for var in ("__NV_PRIME_RENDER_OFFLOAD", "__GLX_VENDOR_LIBRARY_NAME", "__EGL_VENDOR_LIBRARY_FILENAMES"):
  os.environ.pop(var, None)

import mujoco.viewer
from environment import EpisodeLayout, PickPlaceEnv, load_config
from environment.rewards import COMPONENTS
from data_collection.scripted_policy import make_policy
from plot_rewards import LivePlot

# Watch a scenario in the mujoco viewer, with the rewards plotted live next to it
# and every bonus / penalty printed as it happens
#   python play.py success    scripted pick, weave between the obstacles, place
#   python play.py fail       picks, then drops the object mid transport
#   python play.py collide    picks, then carries straight into the obstacles
#   python play.py random     random actions
SCENARIOS = {"success": "success", "fail": "drop", "collide": "collide", "random": "random"}

# One off bonuses and penalties worth a line in the terminal, the shaped terms just show in the plot
EVENTS = ("place", "success", "fail", "collision", "table_hit", "drop")

# Step the env in real time (times speed) until the episode ends or the window is closed
def play(env, policy, speed, plot):
  cfg = env.cfg
  step_time = 1.0 / (cfg.control.hz * speed)
  sums = dict.fromkeys(COMPONENTS, 0.0)
  total = 0.0
  with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
      t0 = time.time()
      action = policy.act()
      _, reward, terminated, truncated, info = env.step(action, give_up=policy.done)
      viewer.sync()
      c = info["reward_components"]
      t = env.step_count / cfg.control.hz
      total += reward
      for k in COMPONENTS:
        sums[k] += cfg.rewards.weights[k] * c[k]
      plot.add(t, reward, c)
      for k in EVENTS:
        if c[k] != 0:
          print(f"  t={t:5.1f}s  {k:10s} {cfg.rewards.weights[k] * c[k]:+.2f}   stage={info['stage']}")
      if terminated or truncated:
        print(f"  episode over: success={info['success']} failed={info['failed']} steps={env.step_count} return={total:.2f}")
        print("  weighted sum per component:", {k: round(v, 2) for k, v in sums.items() if v != 0})
        return info
      time.sleep(max(0.0, step_time - (time.time() - t0)))
  return None

def main():
  p = argparse.ArgumentParser()
  p.add_argument("scenario", choices=SCENARIOS)
  p.add_argument("--config", default=None)
  p.add_argument("--seed", type=int, default=None)
  p.add_argument("--episodes", type=int, default=1)
  p.add_argument("--speed", type=float, default=1.0)
  p.add_argument("--holdout", action="store_true")
  p.add_argument("--layout", default=None, help="replay a stored layout, an episode's meta.json works")
  args = p.parse_args()

  cfg = load_config(args.config)
  env = PickPlaceEnv(cfg)
  layout = None
  if args.layout:
    with open(args.layout) as f:
      d = json.load(f)
    layout = EpisodeLayout.from_dict(d.get("layout", d))
  base_seed = cfg.seed if args.seed is None else args.seed
  plot = LivePlot()
  for ep in range(args.episodes):
    seed = base_seed + ep
    _, info = env.reset(seed=seed, holdout=args.holdout, layout=layout)
    print(f"episode {ep} (seed {seed}): {info['layout']}")
    plot.reset(f"{args.scenario} seed {seed}")
    policy = make_policy(SCENARIOS[args.scenario], env, np.random.default_rng(seed), cfg)
    if play(env, policy, args.speed, plot) is None:
      # window was closed
      break
  # Keep the plot up until it is closed
  print("close the plot window to exit")
  import matplotlib.pyplot as plt
  plt.ioff()
  plt.show()
  env.close()
  # The viewer thread can hang on exit, so leave the hard way
  sys.stdout.flush()
  os._exit(0)

if __name__ == "__main__":
  main()
