"""Draw what a planning agent is thinking into a MuJoCo scene.

Works on any mjvScene: the live viewer's `viewer.user_scn` and an offscreen `mujoco.Renderer`'s
`renderer.scene` (for videos), so the picture is the same in both.

    orange, thick   the plan CEM chose: the imagined gripper path of its H actions
    orange, faint   the next-best (elite) plans of the same search
    blue dots       where the gripper actually went (the executed path)
    green ball      the goal gripper position (JEPA agent: where the goal clip's arm is)
"""

import cv2
import mujoco
import numpy as np

PLAN_RGBA = (1.0, 0.45, 0.0, 0.95)
ELITE_RGBA = (1.0, 0.7, 0.25, 0.35)
TRAIL_RGBA = (0.15, 0.45, 1.0, 0.9)
GOAL_RGBA = (0.1, 0.85, 0.2, 0.6)
_IDENTITY = np.eye(3).flatten()


def _next_geom(scene):
    if scene.ngeom >= scene.maxgeom:
        return None
    scene.ngeom += 1
    return scene.geoms[scene.ngeom - 1]


def add_sphere(scene, pos, radius, rgba):
    g = _next_geom(scene)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, float),
                            _IDENTITY, np.asarray(rgba, np.float32))


def add_segment(scene, a, b, width, rgba):
    g = _next_geom(scene)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), _IDENTITY,
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))


def add_path(scene, points, width, rgba, dots=True):
    for a, b in zip(points[:-1], points[1:]):
        if np.linalg.norm(b - a) > 1e-5:
            add_segment(scene, a, b, width, rgba)
    if dots:
        for p in points[1:]:
            add_sphere(scene, p, 1.6 * width, rgba)


class PlanOverlay:
    """Keeps the latest imagined plan, the executed gripper trail and the goal, and draws them."""

    def __init__(self, n_elites=8, trail_len=300):
        self.n_elites, self.trail_len = n_elites, trail_len
        self.plan, self.elites, self.goal, self.trail = None, [], None, []
        self._seen_plan = None

    def update(self, agent, gripper_pos):
        """Call after every step: records the trail, and re-reads the plan when the agent replanned."""
        self.trail = (self.trail + [np.asarray(gripper_pos, float)])[-self.trail_len:]
        planner = getattr(agent, "planner", None)
        if planner is not None and planner.last_plan is not None and planner.last_plan is not self._seen_plan:
            self._seen_plan = planner.last_plan
            paths = agent.imagined_paths(self.n_elites)
            if paths is not None:
                self.plan, self.elites = paths
        if hasattr(agent, "goal_position"):
            self.goal = agent.goal_position()

    def draw(self, scene):
        for path in self.elites:
            add_path(scene, path, 0.004, ELITE_RGBA, dots=False)
        if self.plan is not None:
            add_path(scene, self.plan, 0.007, PLAN_RGBA)
        for p in self.trail[::2]:
            add_sphere(scene, p, 0.006, TRAIL_RGBA)
        if self.goal is not None:
            add_sphere(scene, self.goal, 0.02, GOAL_RGBA)


def put_text(frame, lines, scale=0.45):
    """Write lines of text in the top-left corner of an RGB frame (for videos)."""
    frame = np.ascontiguousarray(frame)
    for i, line in enumerate(lines):
        y = 18 + 18 * i
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
