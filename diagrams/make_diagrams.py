# Draws the two architecture figures of the README as PNGs: python make_diagrams.py (matplotlib only).
#   architecture.png   what runs at every control step: cameras -> latent and maps -> Q R D O C -> the planner -> the arm
#   pipeline.png       how the models are made: collect -> run folder -> encode -> train -> package -> evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, FILL, SENSOR, MODEL, PLAN = "#1f2937", "#f3f4f6", "#dbeafe", "#dcfce7", "#fef3c7"


def box(ax, x, y, w, h, text, fill=FILL, size=9):
  ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fill, ec=INK, lw=1.2))
  ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.4)


def arrow(ax, a, b, text=None, style="-|>"):
  ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=INK, lw=1.1))
  if text:
    ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center", va="bottom", fontsize=8, color=INK)


def architecture():
  fig, ax = plt.subplots(figsize=(13, 7.4))
  ax.set_xlim(0, 13); ax.set_ylim(-0.15, 6.8); ax.axis("off")
  # sensors
  box(ax, 0.4, 5.6, 2.6, 1.0, "static camera\n256x256, 16 frames + depth", SENSOR)
  box(ax, 3.4, 5.6, 2.6, 1.0, "wrist camera\n16 frames", SENSOR)
  box(ax, 10.0, 5.6, 2.6, 1.0, "robot state p\n20 numbers", SENSOR)
  # perception
  box(ax, 0.4, 3.8, 2.6, 1.1, "V-JEPA 2 (frozen)\n16x16 grid, PCA 32 per cell", MODEL)
  box(ax, 3.4, 3.8, 2.6, 1.1, "V-JEPA 2 (frozen)\n8x8 grid, PCA 16 per cell", MODEL)
  box(ax, 6.6, 3.8, 3.3, 1.1, "obstacle map 16x16\ndepth with robot + cube pixels removed\n(segmentation), running max", SENSOR, 8)
  box(ax, 0.4, 2.3, 5.6, 0.8, "z = [ static 8192 | height map 8x8 | wrist 1024 ]", FILL)
  arrow(ax, (1.7, 5.6), (1.7, 4.9)); arrow(ax, (4.7, 5.6), (4.7, 4.9))
  arrow(ax, (1.7, 3.8), (1.7, 3.1)); arrow(ax, (4.7, 3.8), (4.7, 3.1))
  arrow(ax, (3.0, 5.75), (6.6, 4.75), "depth + segmentation")
  # models
  box(ax, 0.3, 0.9, 1.9, 0.9, "Q(z, p)\ncube xyz, held", MODEL)
  box(ax, 2.4, 0.9, 1.9, 0.9, "R(z, p)\npenalties", MODEL)
  box(ax, 4.5, 0.9, 2.3, 0.9, "D(z, p, a) -> z', p'\n8 steps ahead", MODEL)
  box(ax, 7.0, 0.9, 2.6, 0.9, "O(map window, hand, grip)\nproximity, P(contact)", MODEL, 8)
  box(ax, 9.8, 0.9, 2.9, 0.9, "C(map, target) -> cost-to-go\noptional, --cost-to-go", MODEL, 8)
  for x in (1.25, 3.35, 5.65):
    arrow(ax, (x, 2.3), (x, 1.8))
  arrow(ax, (7.7, 3.8), (7.7, 1.8)); arrow(ax, (9.2, 3.8), (10.6, 1.8))
  arrow(ax, (11.0, 5.6), (8.6, 1.8))
  ax.text(10.1, 3.9, "p", fontsize=9, color=INK)
  # planner
  box(ax, 0.3, -0.05, 12.4, 0.75, "latent MPC (controller/planner.py): 64 waypoint + gripper candidates, a go-to rule makes their actions, D imagines 8 steps, Q / R / O read every step,\n"
      "score = stage potential + pulls (straight line, or C's path length) + penalties, CEM x3, the first action of the best candidate goes to the arm", PLAN, 8)
  for x in (1.25, 3.35, 5.65, 8.3, 11.25):
    arrow(ax, (x, 0.9), (x, 0.7))
  fig.savefig("architecture.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def pipeline():
  fig, ax = plt.subplots(figsize=(13, 3.2))
  ax.set_xlim(0, 13); ax.set_ylim(0, 3.2); ax.axis("off")
  steps = [("collect\ndata_collection.collect\nfull_mix.yml, hack_mix.yml", SENSOR),
           ("run folder\nprepare_episodes\nmanifest, splits", FILL),
           ("encode\nV-JEPA + PCA\n0.26 s / state", MODEL),
           ("train\nQ, R, D on the latents\nO on the maps, C on\nsynthetic scenes", MODEL),
           ("package\nmodels/<tag>\n(Git LFS)", FILL),
           ("evaluate\ncontrol_pipeline\n20 random scenes", PLAN)]
  w, gap = 1.85, 0.32
  for i, (text, fill) in enumerate(steps):
    x = 0.2 + i * (w + gap)
    box(ax, x, 1.1, w, 1.6, text, fill, 8)
    if i:
      arrow(ax, (x - gap, 1.9), (x, 1.9))
  ax.text(6.5, 0.45, "bash cluster_round.sh <tag> 400 runs the whole row; every step restarts where it stopped", ha="center", fontsize=8, color=INK)
  fig.savefig("pipeline.png", dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
  architecture()
  pipeline()
  print("wrote architecture.png and pipeline.png")
