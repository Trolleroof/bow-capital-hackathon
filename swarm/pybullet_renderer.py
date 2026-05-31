"""PyBullet renderer subprocess for the browser sim.

This process intentionally depends only on system Python + pybullet. The training
service owns policy execution and sends pose frames over stdin; this process
renders those poses in a real PyBullet world and writes RGBA camera frames to
stdout as JSON lines.

The swarm drones ("us", blue) are driven by the policy poses arriving on stdin.
Each scenario also dresses the world with semantically-correct 3D props and
SCRIPTED adversaries/targets (red enemy drones, a green moving target, inbound
attackers, ...) so the sim visually matches the scenario description even though
the underlying ``SwarmEnv`` only models the blue coverage team. Scripted entities
are animated purely from the frame time ``t`` carried on each pose message.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from typing import Literal

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from .obstacles import obstacles_for as _obstacles_for
except ImportError:  # direct-script execution
    from obstacles import obstacles_for as _obstacles_for  # type: ignore

# ── team / prop colours ─────────────────────────────────────────────────────
BLUE = [0.22, 0.78, 1.0, 1.0]    # the swarm — "us"
RED = [1.0, 0.30, 0.34, 1.0]     # hostile drones / movers
GREEN = [0.30, 0.85, 0.50, 1.0]  # tracked target / defended asset
GREEN_SOFT = [0.28, 0.78, 0.56, 0.35]  # translucent zone discs
GROUND = [0.36, 0.30, 0.21, 1.0]
SOIL = [0.33, 0.24, 0.17, 0.78]
CONCRETE = [0.31, 0.31, 0.33, 1.0]
WRECKAGE = [0.20, 0.21, 0.22, 1.0]
BERM = [0.42, 0.34, 0.22, 1.0]
TROOP = [0.32, 0.42, 0.24, 1.0]
CameraMode = Literal["observer", "chase", "fpv"]


def _box(p, half_extents, position, color, mass=0.0):
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def _cylinder(p, radius, height, position, color, mass=0.0):
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color,
    )
    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def _capsule(p, radius, height, position, color, mass=0.0):
    visual = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=radius,
        length=height,
        rgbaColor=color,
        specularColor=[0.1, 0.1, 0.1],
    )
    collision = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=height)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def _drone_body(p, position, color, scale=1.0):
    """Spawn a quad-ish drone (flat box) in the given team colour."""
    half = [0.28 * scale, 0.18 * scale, 0.06 * scale]
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def _orbit(cx, cy, r, ang, squash=1.0):
    return cx + math.cos(ang) * r, cy + (math.sin(ang) * r * squash)


# ── per-scenario world + scripted adversaries ───────────────────────────────


def _arena(p) -> None:
    """Shared battlefield floor, perimeter walls, grid, and cover props."""
    size = 22.0
    _box(p, [size, size, 0.015], [0, 0, -0.015], GROUND)
    _box(p, [size * 0.68, size * 0.42, 0.05], [0, 0, 0.05], SOIL)
    _box(p, [size, 0.15, 0.7], [0, -size, 0.7], [0.17, 0.18, 0.20, 1.0])
    _box(p, [size, 0.15, 0.7], [0, size, 0.7], [0.17, 0.18, 0.20, 1.0])
    _box(p, [0.15, size, 0.7], [-size, 0, 0.7], [0.17, 0.18, 0.20, 1.0])
    _box(p, [0.15, size, 0.7], [size, 0, 0.7], [0.17, 0.18, 0.20, 1.0])
    for x in range(-16, 17, 4):
        _box(p, [0.02, 19.0, 0.01], [x, 0, 0.02], [0.08, 0.18, 0.16, 1])
        _box(p, [19.0, 0.02, 0.01], [0, x, 0.022], [0.08, 0.18, 0.16, 1])

    for x, y in [
        (-11.5, -7.5),
        (-4.0, 8.5),
        (6.5, -4.5),
        (12.0, 6.0),
        (16.5, -9.0),
        (-15.5, 3.0),
    ]:
        _cylinder(p, 1.5, 0.04, [x, y, 0.03], [0.15, 0.12, 0.11, 0.95])

    for x, y, yaw in [
        (-8.5, -1.5, 0.35),
        (-2.0, 5.5, -0.2),
        (7.0, 1.2, 0.7),
        (13.5, -5.0, -0.55),
        (4.0, -10.0, 0.15),
    ]:
        body = _box(p, [2.2, 0.45, 0.45], [x, y, 0.42], BERM)
        p.resetBasePositionAndOrientation(
            body,
            [x, y, 0.42],
            p.getQuaternionFromEuler([0.0, 0.0, yaw]),
        )

    for x, y, half_extents, color, yaw in [
        (-13.0, 10.0, [1.3, 1.3, 1.5], CONCRETE, 0.15),
        (10.5, 10.5, [1.3, 1.3, 1.5], CONCRETE, -0.32),
        (2.5, -13.0, [1.3, 1.3, 1.5], CONCRETE, 0.4),
        (-16.0, -11.0, [0.9, 0.45, 0.35], WRECKAGE, -0.6),
        (15.0, 1.0, [0.9, 0.45, 0.35], WRECKAGE, 0.22),
    ]:
        z = 0.8 if half_extents[2] > 1.0 else 0.38
        body = _box(p, half_extents, [x, y, z], color)
        p.resetBasePositionAndOrientation(
            body,
            [x, y, z],
            p.getQuaternionFromEuler([0.0, 0.0, yaw]),
        )


def _add_troop_patrols(p) -> list[dict]:
    """Add the moving ground units from the updated PyBullet demo."""
    scripted: list[dict] = []
    anchor_bases = [(-12.0, -6.0), (-4.0, 8.0), (8.5, -2.0), (14.0, 7.5)]
    for idx in range(20):
        anchor = idx % len(anchor_bases)
        radial = 1.4 + (idx % 5) * 0.82
        theta = idx * 1.618
        offset = (math.cos(theta) * radial, math.sin(theta) * radial)
        phase = idx * 0.73
        x = anchor_bases[anchor][0] + offset[0]
        y = anchor_bases[anchor][1] + offset[1]
        body = _capsule(p, 0.18, 1.0, [x, y, 1.0], TROOP)

        def fn(t, anchor=anchor, offset=offset, phase=phase):
            anchor_positions = [list(base) for base in anchor_bases]
            anchor_positions[0][0] += t * 0.45
            anchor_positions[0][1] += 1.2 * math.sin(t * 0.22)
            anchor_positions[1][0] += 0.8 * math.sin(t * 0.18)
            anchor_positions[1][1] -= 0.7 * t * 0.18
            anchor_positions[2][0] += 0.95 * t * 0.2
            anchor_positions[2][1] += 1.0 * math.sin(t * 0.31)
            anchor_positions[3][0] -= 0.65 * math.sin(t * 0.21)
            anchor_positions[3][1] -= 0.55 * t * 0.16
            drift_x = 0.55 * math.sin(t * 0.75 + phase)
            drift_y = 0.55 * math.cos(t * 0.63 + phase * 0.7)
            x = anchor_positions[anchor][0] + offset[0] + drift_x
            y = anchor_positions[anchor][1] + offset[1] + drift_y
            yaw = math.atan2(anchor_positions[anchor][1] - y, anchor_positions[anchor][0] - x)
            return [x, y, 1.0], yaw

        scripted.append({"body": body, "fn": fn})
    return scripted


_SCENERY_BOX = [0.16, 0.3, 0.28, 1]
_SCENERY_CRATE = [0.18, 0.32, 0.3, 1]
_SCENERY_JAMMER = [0.5, 0.32, 0.16, 1]
_SCENERY_MAST = [0.2, 0.42, 0.4, 1]
_SCENERY_VEHICLE = [0.2, 0.36, 0.34, 1]


def _scenery_color_for(scenario_id: str, obstacle) -> list[float]:
    """Pick a sensible color for a registry obstacle (keeps visuals consistent)."""
    if obstacle.kind == "cylinder":
        if scenario_id == "search-and-interdict":
            return _SCENERY_JAMMER
        if scenario_id == "drone-vs-drone":
            return _SCENERY_MAST
        if scenario_id == "defend-asset":
            return GREEN
        return _SCENERY_MAST
    if scenario_id == "search-and-interdict":
        return _SCENERY_CRATE
    if scenario_id == "moving-target-track" and obstacle.z_extent < 0.8:
        return _SCENERY_VEHICLE
    return _SCENERY_BOX


def _spawn_registered_obstacles(p, scenario_id: str) -> None:
    """Draw every entry from the shared obstacle registry as a PyBullet body."""
    for obstacle in _obstacles_for(scenario_id):
        color = _scenery_color_for(scenario_id, obstacle)
        if obstacle.kind == "cylinder":
            _cylinder(
                p,
                obstacle.sx,
                obstacle.z_extent * 2.0,
                [obstacle.cx, obstacle.cy, obstacle.z_center],
                color,
            )
        else:
            _box(
                p,
                [obstacle.sx, obstacle.sy, obstacle.z_extent],
                [obstacle.cx, obstacle.cy, obstacle.z_center],
                color,
            )


def _build_world(p, env_id: str) -> list[dict]:
    """Build static props for ``env_id`` and return the SCRIPTED entities.

    Each scripted entity is ``{"body": <id>, "fn": fn}`` where ``fn(t)`` returns
    ``([x, y, z], yaw)`` for that body at frame time ``t`` (seconds).

    The collidable scenery is sourced from ``swarm/obstacles.py`` so the env
    that trains the policy and the world that displays it see the same shapes.
    """
    _arena(p)
    scripted: list[dict] = _add_troop_patrols(p)
    _spawn_registered_obstacles(p, env_id)

    if env_id == "drone-vs-drone":
        # contested center lane (visual-only disc; walls + mast come from registry)
        _cylinder(p, 4.2, 0.02, [0, 0, 0.02], GREEN_SOFT)
        # 3 RED enemy drones patrolling the right half, contesting the lane
        for i in range(3):
            body = _drone_body(p, [4.5, 0, 1.0], RED)
            r = 2.4 + i * 0.8

            def fn(t, i=i, r=r):
                ang = t * 0.5 + i * 2.1
                x, y = _orbit(4.8, 0.0, r, ang, 0.85)
                return [x, y, 1.0 + 0.15 * math.sin(t * 0.8 + i)], ang + math.pi / 2

            scripted.append({"body": body, "fn": fn, "hostile": True, "alive": True})

    elif env_id == "moving-target-track":
        # Warehouses + truck come from the registry; only the moving target
        # itself (non-collidable, scripted) is added here.
        target = _box(p, [0.8, 0.5, 0.3], [0, 0, 0.3], GREEN)

        def target_fn(t):
            x = 6.0 * math.sin(t * 0.22)
            y = 4.5 * math.sin(t * 0.41 + 0.6)
            yaw = math.atan2(
                4.5 * 0.41 * math.cos(t * 0.41 + 0.6),
                6.0 * 0.22 * math.cos(t * 0.22),
            )
            return [x, y, 0.3], yaw

        scripted.append({"body": target, "fn": target_fn})

    elif env_id == "search-and-interdict":
        # Crates + jammer come from the registry; one RED ground mover added here.
        mover = _drone_body(p, [0, 0, 0.35], RED, scale=1.1)

        def mover_fn(t):
            x = 5.0 * math.sin(t * 0.18) + 1.5
            y = 5.0 * math.cos(t * 0.27)
            yaw = t * 0.27 + math.pi
            return [x, y, 0.35], yaw

        scripted.append({"body": mover, "fn": mover_fn})

    elif env_id == "defend-asset":
        # Asset + hardpoints come from the registry; only the standoff disc
        # (visual-only) is added here. RED attackers spiral inward.
        _cylinder(p, 6.5, 0.02, [0, 0, 0.02], GREEN_SOFT)
        for i in range(3):
            body = _drone_body(p, [9.0, 0, 1.0], RED)

            def fn(t, i=i):
                period = 13.0
                phase = ((t + i * 4.3) % period) / period
                radius = 9.2 - 7.2 * phase
                ang = i * 2.1 + t * 0.35
                x, y = _orbit(0.0, 0.0, radius, ang)
                return [x, y, 1.0], ang + math.pi / 2

            scripted.append({"body": body, "fn": fn})

    elif env_id == "hunt-and-seek":
        # The evading "user": a single bright mover driven LIVE from the env's
        # target_pos (3D) carried on each pose message — NOT time-scripted. We
        # create the body here and update it in the main loop. A faint ground
        # ring tracks beneath it so the operator can see where it is.
        target = _drone_body(p, [0.0, 0.0, 2.0], RED, scale=1.15)
        scripted.append({"body": target, "hunt_target": True, "alive": True})

    elif env_id == "navigate-to-target":
        # Static goal beacon at the far end of the obstacle corridor.
        # SwarmEnv places it at [0.85 * world_half, 0.0] = [8.5, 0.0].
        goal_x, goal_y = 8.5, 0.0
        # Pulsing green ring on the ground marking the goal zone
        _cylinder(p, 1.2, 0.02, [goal_x, goal_y, 0.02], GREEN_SOFT)
        # Tall beacon post so it's visible from observer cam
        _cylinder(p, 0.12, 3.0, [goal_x, goal_y, 1.5], GREEN)
        # Small cap sphere (approximated as a flattened cylinder) on top
        _cylinder(p, 0.45, 0.12, [goal_x, goal_y, 3.1], GREEN)

    return scripted


def _update_scripted(p, scripted: list[dict], t: float) -> None:
    for entity in scripted:
        # Hunt target is driven live from the pose message, not by a time fn.
        if entity.get("hunt_target"):
            continue
        if not entity.get("alive", True):
            p.resetBasePositionAndOrientation(entity["body"], [0.0, 0.0, -20.0], [0, 0, 0, 1])
            continue
        pos, yaw = entity["fn"](t)
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(entity["body"], pos, quat)


def _update_hunt_target(p, scripted: list[dict], message: dict) -> None:
    """Drive the hunt-and-seek evader body from the live target_pos in the message."""
    tp = message.get("target_pos")
    if not tp:
        return
    x = float(tp[0])
    y = float(tp[1])
    z = float(tp[2]) if len(tp) >= 3 else 2.0
    for entity in scripted:
        if entity.get("hunt_target"):
            p.resetBasePositionAndOrientation(entity["body"], [x, y, z], [0, 0, 0, 1])


def _apply_drone_engagement(scripted: list[dict], agents: list[dict], t: float, radius: float = 2.5) -> None:
    """Mark hostile scripted drones eliminated when a friendly drone closes in."""
    live_agents = [agent for agent in agents if agent.get("alive", True)]
    if not live_agents:
        return
    for entity in scripted:
        if not entity.get("hostile", False) or not entity.get("alive", True):
            continue
        pos, _ = entity["fn"](t)
        hx, hy = float(pos[0]), float(pos[1])
        for agent in live_agents:
            dx = float(agent.get("x", 0.0)) - hx
            dy = float(agent.get("y", 0.0)) - hy
            if (dx * dx + dy * dy) ** 0.5 <= radius:
                entity["alive"] = False
                break


def _spawn_drone(p, position, scale: float = 1.0) -> int:
    """Spawn a blue swarm drone ("us"). Colour is team-consistent, not per-id."""
    return _drone_body(p, position, BLUE, scale=scale)


def _update_bodies(p, bodies: dict[int, int], agents: list[dict], scale: float = 1.0) -> None:
    for agent in agents:
        agent_id = int(agent["id"])
        pos = [float(agent["x"]), float(agent["y"]), float(agent["z"])]
        yaw = float(agent.get("yaw", 0.0))
        if agent_id not in bodies:
            bodies[agent_id] = _spawn_drone(p, pos, scale=scale)
        # Killed agents are removed from the scene (sunk far below the floor).
        if not agent.get("alive", True):
            p.resetBasePositionAndOrientation(bodies[agent_id], [pos[0], pos[1], -50.0], [0, 0, 0, 1])
            continue
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(bodies[agent_id], pos, quat)


def _mean_agent_position(agents: list[dict]) -> list[float]:
    live_agents = [agent for agent in agents if agent.get("alive", True)]
    if not live_agents:
        return [0.0, 0.0, 1.0]
    n = len(live_agents)
    return [
        sum(float(agent.get("x", 0.0)) for agent in live_agents) / n,
        sum(float(agent.get("y", 0.0)) for agent in live_agents) / n,
        sum(float(agent.get("z", 1.0)) for agent in live_agents) / n,
    ]


def _selected_agent(agents: list[dict], selected_drone: int) -> dict | None:
    if not agents:
        return None
    for agent in agents:
        if int(agent.get("id", -1)) == selected_drone:
            return agent
    return agents[max(0, min(selected_drone, len(agents) - 1))]


def _camera_vectors(
    agents: list[dict],
    t: float,
    mode: CameraMode,
    selected_drone: int,
) -> tuple[list[float], list[float], float]:
    if mode in {"chase", "fpv"}:
        agent = _selected_agent(agents, selected_drone)
        if agent is not None:
            yaw = float(agent.get("yaw", 0.0))
            x = float(agent.get("x", 0.0))
            y = float(agent.get("y", 0.0))
            z = float(agent.get("z", 2.0))
            forward = [math.cos(yaw), math.sin(yaw), 0.0]
            if mode == "fpv":
                tilt = math.radians(58.0)
                eye = [
                    x + 0.18 * forward[0],
                    y + 0.18 * forward[1],
                    z - 0.08,
                ]
                target = [
                    eye[0] + math.cos(yaw) * math.cos(tilt) * 18.0,
                    eye[1] + math.sin(yaw) * math.cos(tilt) * 18.0,
                    eye[2] - math.sin(tilt) * 18.0,
                ]
                return eye, target, 78.0
            eye = [x - forward[0] * 8.0, y - forward[1] * 8.0, z + 3.4]
            target = [x + forward[0] * 5.5, y + forward[1] * 5.5, z - 1.2]
            return eye, target, 62.0

    focus = _mean_agent_position(agents)
    yaw = math.radians(38.0 + 18.0 * math.sin(t * 0.08))
    distance = 24.0
    pitch = math.radians(36.0)
    eye = [
        focus[0] + math.cos(yaw) * math.cos(pitch) * distance,
        focus[1] + math.sin(yaw) * math.cos(pitch) * distance,
        focus[2] + math.sin(pitch) * distance + 2.0,
    ]
    target = [focus[0] * 0.45, focus[1] * 0.45, max(1.4, focus[2] + 0.6)]
    return eye, target, 58.0


def _render_frame(
    p,
    width: int,
    height: int,
    agents: list[dict],
    t: float,
    camera_mode: CameraMode,
    selected_drone: int,
    lit: bool = False,
) -> dict:
    eye, target, fov = _camera_vectors(agents, t, camera_mode, selected_drone)
    view = p.computeViewMatrix(eye, target, [0.0, 0.0, 1.0])
    proj = p.computeProjectionMatrixFOV(fov, width / height, 0.05, 120)
    # ``lit`` adds directional light + shadows so the photoreal hunt scene reads
    # well; the primitive worlds keep the cheaper flat render.
    light_kwargs = (
        dict(
            lightDirection=[0.42, 0.18, 1.0],
            lightColor=[1.0, 0.96, 0.9],
            lightAmbientCoeff=0.55,
            lightDiffuseCoeff=0.68,
            lightSpecularCoeff=0.08,
            shadow=1,
        )
        if lit
        else {}
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER,
        **light_kwargs,
    )
    try:
        raw = bytes(rgba)
    except TypeError:
        raw = memoryview(rgba).tobytes()

    if _HAS_PIL:
        # JPEG-encode so each frame fits inside the WebSocket 1 MB frame limit
        # (raw RGBA at 960x540 is ~2 MB base64 and trips uvicorn's 1009 cutoff).
        img = Image.frombytes("RGBA", (width, height), raw).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=72)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        encoding: Literal["jpeg", "rgba"] = "jpeg"
    else:
        encoded = base64.b64encode(raw).decode("ascii")
        encoding = "rgba"

    return {
        "topic": "pybullet_frame",
        "width": width,
        "height": height,
        "encoding": encoding,
        "camera_mode": camera_mode,
        "selected_drone": selected_drone,
        "data": encoded,
    }


# ── rich hunt-and-seek scene (pybullet_swarm_video assets) ───────────────────


def _spawn_extra_drone(sim, p, position) -> int:
    """Fallback drone body when a pose arrives for an id beyond the spawned ring."""
    asset = getattr(sim, "_drone_mesh_asset_ref", None)
    cid = sim.client_id
    collision = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.08], physicsClientId=cid
    )
    if asset is not None:
        visual = sim._create_mesh_visual(asset)
    else:
        visual = None
    if visual is None:
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.08],
            rgbaColor=[0.90, 0.90, 0.92, 1.0],
            physicsClientId=cid,
        )
    body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=list(position),
        physicsClientId=cid,
    )
    if asset is not None:
        sim._apply_body_texture(body, asset.texture_path)
    return body


def _sync_rich_drones(sim, p, agents: list[dict], ground_offset: float) -> None:
    """Drive the FPV drone meshes from incoming PPO poses (blue swarm = "us")."""
    cid = sim.client_id
    ref = getattr(sim, "_drone_mesh_asset_ref", None)
    roll = ref.roll_offset if ref else 0.0
    pitch = ref.pitch_offset if ref else 0.0
    yaw_off = ref.yaw_offset if ref else 0.0
    vz = ref.vertical_offset if ref else 0.0
    for agent in agents:
        idx = int(agent["id"])
        x = float(agent["x"])
        y = float(agent["y"])
        z = float(agent["z"]) + ground_offset
        if idx >= len(sim.drone_ids):
            sim.drone_ids.append(_spawn_extra_drone(sim, p, [x, y, z]))
            sim._drone_marker_ids.append(None)
        body = sim.drone_ids[idx]
        if not agent.get("alive", True):
            p.resetBasePositionAndOrientation(body, [x, y, -50.0], [0, 0, 0, 1], physicsClientId=cid)
            marker = sim._drone_marker_ids[idx] if idx < len(sim._drone_marker_ids) else None
            if marker is not None:
                p.resetBasePositionAndOrientation(marker, [x, y, -50.0], [0, 0, 0, 1], physicsClientId=cid)
            continue
        yaw = float(agent.get("yaw", 0.0))
        quat = p.getQuaternionFromEuler([roll, pitch, yaw + yaw_off])
        p.resetBasePositionAndOrientation(body, [x, y, z + vz], quat, physicsClientId=cid)
        marker = sim._drone_marker_ids[idx] if idx < len(sim._drone_marker_ids) else None
        if marker is not None:
            p.resetBasePositionAndOrientation(marker, [x, y, z + 0.9], [0, 0, 0, 1], physicsClientId=cid)


def _sync_hunt_human(sim, p, message: dict, ground_offset: float, state: dict) -> None:
    """Drive troop[0] (a soldier mesh) from the live target_pos — the hunted human."""
    tp = message.get("target_pos")
    if not tp or not sim.troop_ids:
        return
    x = float(tp[0])
    y = float(tp[1])
    idx = 0
    cid = sim.client_id
    use_mesh = bool(sim._troop_mesh_mask[idx]) if len(sim._troop_mesh_mask) else False
    base_z = 1.0 + ground_offset + (sim._troop_visual_z_offset if use_mesh else 0.0)
    prev = state.get("human_xy")
    if prev is not None:
        dx, dy = x - prev[0], y - prev[1]
        if math.hypot(dx, dy) > 0.01:
            state["human_yaw"] = math.atan2(dy, dx)
    state["human_xy"] = (x, y)
    yaw = state.get("human_yaw", 0.0) + sim._troop_visual_yaw_offset
    roll = sim._troop_mesh_asset_ref.roll_offset if (use_mesh and sim._troop_mesh_asset_ref) else 0.0
    pitch = sim._troop_mesh_asset_ref.pitch_offset if (use_mesh and sim._troop_mesh_asset_ref) else 0.0
    quat = p.getQuaternionFromEuler([roll, pitch, yaw])
    p.resetBasePositionAndOrientation(sim.troop_ids[idx], [x, y, base_z], quat, physicsClientId=cid)
    if idx < len(sim._troop_marker_ids) and sim._troop_marker_ids[idx] is not None:
        p.resetBasePositionAndOrientation(
            sim._troop_marker_ids[idx], [x, y, base_z + 1.4], [0, 0, 0, 1], physicsClientId=cid
        )


def _run_rich_hunt(args) -> bool:
    """Render hunt-and-seek with pybullet_swarm_video scenery + assets.

    Returns ``True`` once it has taken over the stdin→stdout loop (i.e. the
    rich scene built successfully). Returns ``False`` if setup fails *before*
    any stdin is consumed, so the caller can fall back to the primitive world.
    """
    try:
        from pathlib import Path

        here = Path(__file__).resolve()
        pkg_root = here.parents[1] / "pybullet_swarm_video"
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))

        # The renderer's Python ships a bare pybullet .so without the
        # ``pybullet_data`` package, which simulation.py imports for the ground
        # plane. Provide a tiny shim backed by vendored plane data so the rich
        # scene works without altering the Python install.
        if "pybullet_data" not in sys.modules:
            try:
                import pybullet_data  # noqa: F401  (real package if present)
            except ImportError:
                import types

                data_dir = str(here.parent / "_bullet_data")
                shim = types.ModuleType("pybullet_data")
                shim.getDataPath = lambda: data_dir  # type: ignore[attr-defined]
                sys.modules["pybullet_data"] = shim

        from pybullet_swarm_video.config import SimulationConfig
        from pybullet_swarm_video.simulation import DroneSurveillanceSimulation
        import pybullet as p

        config = SimulationConfig(
            num_drones=5,          # matches hunt-and-seek N_AGENTS
            num_troops=6,          # 1 hunted human (idx 0) + ambient soldiers
            resources_dir=pkg_root / "resources",
        )
        sim = DroneSurveillanceSimulation(config, gui=False)
        # The sim logs asset issues via print() → stdout, which is our JSON
        # frame channel. Reroute to stderr so a missing texture can't corrupt
        # the stream the dashboard parses.
        sim._log_asset = lambda msg, _seen=sim._asset_messages: (  # type: ignore[assignment]
            None if msg in _seen else (_seen.add(msg), print(f"[sim-asset] {msg}", file=sys.stderr, flush=True))[0]
        )
        sim.reset()
    except Exception as exc:  # noqa: BLE001 - any failure → primitive fallback
        print(f"[pybullet-renderer] rich hunt setup failed: {exc}", file=sys.stderr, flush=True)
        return False

    ground_offset = float(getattr(sim, "_ground_offset_z", 0.0))
    cid = sim.client_id

    # Highlight the hunted human (troop 0) with a red marker; ambient soldiers
    # stay green. Everything else in the scene is static scenery.
    if sim.troop_ids and sim._troop_marker_ids and sim._troop_marker_ids[0] is not None:
        try:
            p.changeVisualShape(
                sim._troop_marker_ids[0], -1, rgbaColor=[0.95, 0.22, 0.24, 0.95], physicsClientId=cid
            )
        except Exception:
            pass

    human_state: dict = {}
    try:
        for raw in sys.stdin:
            if not raw.strip():
                continue
            message = json.loads(raw)
            agents = message.get("agents", [])
            t = float(message.get("t", 0.0))
            _sync_rich_drones(sim, p, agents, ground_offset)
            _sync_hunt_human(sim, p, message, ground_offset, human_state)
            p.stepSimulation(physicsClientId=cid)
            frame = _render_frame(
                p,
                args.width,
                args.height,
                [
                    {**a, "z": float(a["z"]) + ground_offset}
                    for a in agents
                    if a.get("alive", True)
                ],
                t,
                args.camera_mode,
                args.selected_drone,
                lit=True,
            )
            frame["t"] = message.get("t", 0)
            frame["env_id"] = message.get("env_id", args.env_id)
            print(json.dumps(frame), flush=True)
    finally:
        sim.close()
    return True


def _run_primitive(args) -> None:
    import pybullet as p
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    scripted = _build_world(p, args.env_id)
    bodies: dict[int, int] = {}

    # Single-agent scenarios get a larger drone so it's visible from observer cam.
    drone_scale = 2.0 if args.env_id == "navigate-to-target" else 1.0

    try:
        for raw in sys.stdin:
            if not raw.strip():
                continue
            message = json.loads(raw)
            agents = message.get("agents", [])
            t = float(message.get("t", 0.0))
            if args.env_id == "drone-vs-drone":
                _apply_drone_engagement(scripted, agents, t)
            _update_bodies(p, bodies, agents, scale=drone_scale)
            _update_scripted(p, scripted, t)
            if args.env_id == "hunt-and-seek":
                _update_hunt_target(p, scripted, message)
            p.stepSimulation()
            frame = _render_frame(
                p,
                args.width,
                args.height,
                agents,
                t,
                args.camera_mode,
                args.selected_drone,
            )
            frame["t"] = message.get("t", 0)
            frame["env_id"] = message.get("env_id", args.env_id)
            print(json.dumps(frame), flush=True)
    finally:
        p.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render policy poses in PyBullet")
    parser.add_argument("--env-id", default="search-and-interdict")
    # Default kept modest so the raw-RGBA fallback (when Pillow is absent in the
    # renderer's Python) stays under the 1 MB WebSocket frame limit: 512x288 RGBA
    # base64 ≈ 786 KB. With Pillow present, frames are JPEG and far smaller, so a
    # higher resolution can be requested via --width/--height.
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument(
        "--camera-mode",
        choices=["observer", "chase", "fpv"],
        default="observer",
    )
    parser.add_argument("--selected-drone", type=int, default=0)
    args = parser.parse_args()

    # hunt-and-seek gets the photoreal pybullet_swarm_video scenery; everything
    # else keeps the lightweight primitive world. The pose/frame protocol is
    # identical for both paths.
    if args.env_id == "hunt-and-seek" and _run_rich_hunt(args):
        return
    _run_primitive(args)


if __name__ == "__main__":
    main()
