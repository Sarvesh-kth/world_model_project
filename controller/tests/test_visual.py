"""Tests for the visualizer's overlay and the agents' imagined-path replay."""

import mujoco
import numpy as np
import pytest

from controller.visual import PlanOverlay


def test_overlay_draws_into_an_offscreen_scene():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 1'/></worldbody></mujoco>")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 64, 64)
    renderer.update_scene(data)
    before = renderer.scene.ngeom
    overlay = PlanOverlay()
    overlay.plan = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], float)
    overlay.elites, overlay.trail, overlay.goal = [overlay.plan + 0.01], [np.zeros(3)] * 4, np.ones(3)
    overlay.draw(renderer.scene)
    assert renderer.scene.ngeom > before
    assert renderer.render().shape == (64, 64, 3)
    renderer.close()


@pytest.mark.m1
def test_oracle_agent_replays_its_plan_as_gripper_paths():
    try:
        from controller.adapters.m1_adapter import grade_e_adapter
    except FileNotFoundError:
        pytest.skip("M1's simulation/ folder not found")
    from controller.agents import OracleCEMAgent
    from controller.config import CEMConfig
    from controller.costs import reach_cost

    adapter = grade_e_adapter()
    obs, _ = adapter.reset(seed=0)
    agent = OracleCEMAgent(reach_cost, CEMConfig(horizon=4, n_samples=8, n_elites=3, n_iters=1, execute_steps=2))
    agent.reset(adapter, obs)
    assert agent.imagined_paths() is None  # nothing planned yet
    ee = adapter.task_features()[0:3]
    agent.act(adapter, obs)
    plan, elites = agent.imagined_paths(n_elites=2)
    assert plan.shape == (5, 3) and elites.shape == (2, 5, 3)
    assert np.allclose(plan[0], ee)  # every imagined path starts at the real gripper
    adapter.close()
