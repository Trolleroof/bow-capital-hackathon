from __future__ import annotations

import math

import pytest


pybullet = pytest.importorskip("pybullet")
pytest.importorskip("pybullet_data")

from swarm.bus import ScriptedPyBulletRuntime


def test_pybullet_runtime_smoke() -> None:
    runtime = ScriptedPyBulletRuntime(n_agents=3)
    try:
        runtime.reset()
        for _ in range(50):
            runtime.step(1.0 / 10.0)

        payload = runtime.message()
        assert len(payload["agents"]) == 3
        for agent in payload["agents"]:
            assert math.isfinite(agent["x"])
            assert math.isfinite(agent["y"])
            assert math.isfinite(agent["z"])
            assert math.isfinite(agent["yaw"])
    finally:
        runtime.close()
