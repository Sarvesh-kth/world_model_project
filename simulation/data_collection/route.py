import heapq

import numpy as np

# 2D route planning around the obstacle footprints on the table
# A* on a coarse grid where cells inside an obstacle circle are expensive but not blocked,
# so a start that is already inside a circle (right after lifting next to an obstacle) still gets a path out
# the grid path is then straightened by line of sight smoothing

CELL = 0.02
BLOCKED_COST = 1000.0
NEIGHBOURS = [(dx, dy, float(np.hypot(dx, dy))) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
              if (dx, dy) != (0, 0)]


# Waypoints from start to goal (not including either), circles is a list of (centre_xy, radius)
# lo / hi bound the walkable area, reach is (base_xy, max_radius) the arm can get to
# returns [] when the straight line is already fine
def plan_route(start, goal, circles, lo, hi, reach=None):
  start, goal = np.asarray(start, float), np.asarray(goal, float)
  if not circles or segment_ok(start, goal, circles):
    return []
  lo, hi = np.asarray(lo, float), np.asarray(hi, float)
  shape = np.maximum(np.ceil((hi - lo) / CELL).astype(int) + 1, 2)
  xs = lo[0] + CELL * np.arange(shape[0])
  ys = lo[1] + CELL * np.arange(shape[1])
  gx, gy = np.meshgrid(xs, ys, indexing="ij")

  # Cost map, expensive inside the circles and out of the arm's reach
  cost = np.ones(shape)
  for centre, radius in circles:
    cost[(gx - centre[0]) ** 2 + (gy - centre[1]) ** 2 < radius ** 2] = BLOCKED_COST
  if reach is not None:
    base, radius = reach
    cost[(gx - base[0]) ** 2 + (gy - base[1]) ** 2 > radius ** 2] = BLOCKED_COST

  def cell(p):
    return tuple(int(v) for v in np.clip(np.round((p - lo) / CELL), 0, shape - 1))

  path = _astar(cost, cell(start), cell(goal))
  points = [start] + [lo + CELL * np.array(c) for c in path[1:-1]] + [goal]
  return _smooth(points, circles)[1:-1]


# How far p is inside the deepest circle, <= 0 when outside all of them
def depth(p, circles):
  return max((r - float(np.linalg.norm(p - c)) for c, r in circles), default=0.0)


# Same but for the closest point on the segment a -> b
def segment_depth(a, b, circles):
  ab = b - a
  denom = max(float(ab @ ab), 1e-12)
  worst = 0.0
  for centre, radius in circles:
    t = np.clip(((centre - a) @ ab) / denom, 0.0, 1.0)
    worst = max(worst, radius - float(np.linalg.norm(a + t * ab - centre)))
  return worst


# True when the segment cuts no deeper into a circle than a or b already do
def segment_ok(a, b, circles, tol=0.005):
  return segment_depth(a, b, circles) <= max(depth(a, circles), depth(b, circles), 0.0) + tol


# Plain A* over the grid, 8 connected
def _astar(cost, s, g):
  shape = cost.shape
  heur = lambda c: float(np.hypot(c[0] - g[0], c[1] - g[1]))
  best = {s: 0.0}
  parent = {}
  frontier = [(heur(s), 0.0, s)]
  while frontier:
    _, so_far, c = heapq.heappop(frontier)
    if c == g:
      # Walk back to the start
      path = [c]
      while c in parent:
        c = parent[c]
        path.append(c)
      return path[::-1]
    if so_far > best.get(c, np.inf):
      continue
    for dx, dy, step in NEIGHBOURS:
      n = (c[0] + dx, c[1] + dy)
      if not (0 <= n[0] < shape[0] and 0 <= n[1] < shape[1]):
        continue
      new = so_far + step * cost[n]
      if new < best.get(n, np.inf):
        best[n] = new
        parent[n] = c
        heapq.heappush(frontier, (new + heur(n), new, n))
  return [s, g]


# Skip ahead to the furthest point we can reach in a straight line, repeat
def _smooth(points, circles):
  out = [points[0]]
  i = 0
  while i < len(points) - 1:
    j = len(points) - 1
    while j > i + 1 and not segment_ok(points[i], points[j], circles):
      j -= 1
    out.append(points[j])
    i = j
  return out
