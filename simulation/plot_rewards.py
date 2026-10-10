import argparse
import csv
import pathlib

import matplotlib.pyplot as plt

from environment.rewards import COMPONENTS, BONUSES, PENALTIES

# One fixed colour per component so the same thing always looks the same
COLORS = {
  "reach": "#1f77b4", "grasp": "#2ca02c", "lift": "#17becf", "transport": "#9467bd",
  "place": "#e377c2", "success": "#bcbd22",
  "fail": "#d62728", "collision": "#ff7f0e", "proximity": "#ffbb78", "table_hit": "#8c564b",
  "drop": "#7f7f7f", "action": "#c49c94",
}

# Reward plots, three panels: reward per step with the running return, the bonus terms, the penalties
#   python plot_rewards.py data/smoke/episode_000003            plot a recorded episode
#   python plot_rewards.py data/smoke/episode_000003 --save     write rewards.png into the folder
# play.py uses LivePlot to show the same thing while the episode runs


def _make_figure(title):
  fig, (top, mid, bottom) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
  top.set_title(title)
  top.set_ylabel("reward per step")
  cum = top.twinx()
  cum.set_ylabel("return so far", color="gray")
  mid.set_ylabel("bonus terms (unweighted)")
  bottom.set_ylabel("penalties (unweighted)")
  bottom.set_xlabel("time (s)")
  for ax in (top, mid, bottom):
    ax.grid(alpha=0.3)
  return fig, top, cum, mid, bottom


def _running(totals):
  out, s = [], 0.0
  for v in totals:
    s += v
    out.append(s)
  return out


def plot_rewards(times, totals, components, title="", save=None):
  fig, top, cum, mid, bottom = _make_figure(title)
  top.plot(times, totals, color="black", label="reward")
  cum.plot(times, _running(totals), color="gray", linestyle="--", label="return")
  for ax, keys in ((mid, BONUSES), (bottom, PENALTIES)):
    for k in keys:
      # skip the ones that never fire, keeps the legend readable
      if any(v != 0 for v in components[k]):
        ax.plot(times, components[k], label=k, color=COLORS[k])
    ax.legend(loc="upper left", ncol=3, fontsize=8)
  fig.tight_layout()
  if save:
    fig.savefig(save, dpi=120)
    print(f"wrote {save}")
  else:
    plt.show()
  plt.close(fig)


# Same plot but updated step by step while an episode plays
class LivePlot:

  def __init__(self, title=""):
    plt.ion()
    self.fig, self.top, self.cum, self.mid, self.bottom = _make_figure(title)
    self.total_line, = self.top.plot([], [], color="black")
    self.cum_line, = self.cum.plot([], [], color="gray", linestyle="--")
    self.lines = {}
    for ax, keys in ((self.mid, BONUSES), (self.bottom, PENALTIES)):
      for k in keys:
        self.lines[k], = ax.plot([], [], label=k, color=COLORS[k])
      ax.legend(loc="upper left", ncol=3, fontsize=8)
    self.fig.tight_layout()
    self.reset(title)
    self.fig.show()

  def add(self, time, total, components):
    self.times.append(time)
    self.totals.append(total)
    self.returns.append(self.returns[-1] + total if self.returns else total)
    for k in COMPONENTS:
      self.components[k].append(components[k])
    self.total_line.set_data(self.times, self.totals)
    self.cum_line.set_data(self.times, self.returns)
    for k in COMPONENTS:
      self.lines[k].set_data(self.times, self.components[k])
    for ax in (self.top, self.cum, self.mid, self.bottom):
      ax.relim()
      ax.autoscale_view()
    self.fig.canvas.draw_idle()
    self.fig.canvas.flush_events()

  # Start a new episode on the same window
  def reset(self, title):
    self.times, self.totals, self.returns = [], [], []
    self.components = {k: [] for k in COMPONENTS}
    self.top.set_title(title)

  def close(self):
    plt.close(self.fig)


# Read the rewards back out of an episode's data.csv
def load_rewards(ep_dir):
  times, totals = [], []
  components = {k: [] for k in COMPONENTS}
  with open(pathlib.Path(ep_dir) / "data.csv") as f:
    for row in csv.DictReader(f):
      times.append(float(row["time"]))
      totals.append(float(row["reward_total"]))
      for k in COMPONENTS:
        components[k].append(float(row[f"reward_{k}"]))
  return times, totals, components


def main():
  p = argparse.ArgumentParser()
  p.add_argument("episode", help="episode folder, e.g. data/smoke/episode_000003")
  p.add_argument("--save", action="store_true", help="write rewards.png into the folder instead of showing")
  args = p.parse_args()
  ep_dir = pathlib.Path(args.episode)
  times, totals, components = load_rewards(ep_dir)
  save = ep_dir / "rewards.png" if args.save else None
  plot_rewards(times, totals, components, title=ep_dir.name, save=save)


if __name__ == "__main__":
  main()
