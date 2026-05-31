"""PyBullet renderer subprocess for the browser sim.

This process intentionally depends only on system Python + pybullet. The training
service owns policy execution and sends pose frames over stdin; this process
renders those poses in a real PyBullet world and writes RGBA camera frames to
stdout as JSON lines.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys


def _box(p, half_extents, position, color, mass=0.0):
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def _cylinder(p, radius, height, position, color):
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color,
    )
    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def _build_world(p, env_id: str) -> None:
    _box(p, [12.0, 12.0, 0.015], [0, 0, -0.015], [0.025, 0.038, 0.035, 1])
    _box(p, [10.5, 0.04, 0.018], [0, -10.5, 0.02], [0.18, 0.54, 0.4, 1])
    _box(p, [10.5, 0.04, 0.018], [0, 10.5, 0.02], [0.18, 0.54, 0.4, 1])
    _box(p, [0.04, 10.5, 0.018], [-10.5, 0, 0.02], [0.18, 0.54, 0.4, 1])
    _box(p, [0.04, 10.5, 0.018], [10.5, 0, 0.02], [0.18, 0.54, 0.4, 1])

    for x in range(-8, 9, 4):
        _box(p, [0.02, 10.0, 0.01], [x, 0, 0.015], [0.08, 0.18, 0.16, 1])
        _box(p, [10.0, 0.02, 0.01], [0, x, 0.016], [0.08, 0.18, 0.16, 1])

    layouts = {
        "drone-vs-drone": [(-3.0, -1.8), (3.0, 1.8), (0.0, 3.2)],
        "moving-target-track": [(-4.0, 0.0), (0.0, 2.4), (4.0, -1.6)],
        "search-and-interdict": [(-3.8, -3.2), (-1.4, 2.8), (2.8, 1.1), (4.3, -3.4)],
        "defend-asset": [(-4.4, 0.0), (4.4, 0.0), (0.0, -4.4), (0.0, 4.4)],
        "swarm-vs-swarm-race": [(-5.0, -2.0), (-1.8, 2.0), (1.8, -2.0), (5.0, 2.0)],
    }
    for idx, (x, y) in enumerate(layouts.get(env_id, layouts["search-and-interdict"])):
        height = 0.45 + 0.18 * (idx % 2)
        _box(p, [0.7, 0.45, height], [x, y, height], [0.11, 0.24, 0.22, 1])

    if env_id == "defend-asset":
        _cylinder(p, 0.95, 0.2, [0.0, 0.0, 0.1], [0.28, 0.78, 0.56, 1])
    elif env_id == "moving-target-track":
        _box(p, [0.6, 0.32, 0.16], [2.5, -2.7, 0.18], [0.88, 0.34, 0.16, 1])
    elif env_id == "swarm-vs-swarm-race":
        for x in (-6.0, -2.0, 2.0, 6.0):
            _box(p, [1.25, 0.25, 0.018], [x, 0, 0.03], [0.13, 0.38, 0.26, 1])


def _spawn_drone(p, agent_id: int, position) -> int:
    color = [0.22, 0.78, 1.0, 1.0] if agent_id % 2 == 0 else [1.0, 0.54, 0.22, 1.0]
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.28, 0.18, 0.06], rgbaColor=color)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.28, 0.18, 0.06])
    return p.createMultiBody(
        baseMass=0.85,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def _update_bodies(p, bodies: dict[int, int], agents: list[dict]) -> None:
    for agent in agents:
        agent_id = int(agent["id"])
        pos = [float(agent["x"]), float(agent["y"]), float(agent["z"])]
        yaw = float(agent.get("yaw", 0.0))
        if agent_id not in bodies:
            bodies[agent_id] = _spawn_drone(p, agent_id, pos)
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(bodies[agent_id], pos, quat)


def _render_frame(p, agents: list[dict], width: int, height: int) -> dict:
    if agents:
        cx = sum(float(agent["x"]) for agent in agents) / len(agents)
        cy = sum(float(agent["y"]) for agent in agents) / len(agents)
    else:
        cx = cy = 0.0
    target = [cx, cy, 1.1]
    eye = [cx + 7.2, cy - 8.8, 5.2]
    view = p.computeViewMatrix(eye, target, [0.0, 0.0, 1.0])
    proj = p.computeProjectionMatrixFOV(58, width / height, 0.1, 80)
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER,
    )
    try:
        raw = bytes(rgba)
    except TypeError:
        raw = memoryview(rgba).tobytes()
    return {
        "topic": "pybullet_frame",
        "width": width,
        "height": height,
        "encoding": "rgba",
        "data": base64.b64encode(raw).decode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render policy poses in PyBullet")
    parser.add_argument("--env-id", default="search-and-interdict")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    import pybullet as p
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    _build_world(p, args.env_id)
    bodies: dict[int, int] = {}

    try:
        for raw in sys.stdin:
            if not raw.strip():
                continue
            message = json.loads(raw)
            agents = message.get("agents", [])
            _update_bodies(p, bodies, agents)
            p.stepSimulation()
            frame = _render_frame(p, agents, args.width, args.height)
            frame["t"] = message.get("t", 0)
            frame["env_id"] = message.get("env_id", args.env_id)
            print(json.dumps(frame), flush=True)
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()
