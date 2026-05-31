"""WebSocket bus for the CombatOS swarm runtime.

The primary demo path is now a scripted PyBullet swarm that publishes the
existing `swarm` bus payload at ~10 Hz. When PyBullet is unavailable, the module
falls back to the older point-mass runtime so the UI still has a nonblank scene.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from dataclasses import dataclass
from importlib import resources

import numpy as np
import websockets

try:
    from .drones_broll import _make_fallback_drone
except ImportError:  # pragma: no cover - direct-script execution
    from drones_broll import _make_fallback_drone  # type: ignore

try:
    from .env import ALTITUDE, DT, MAX_SPEED, ROLES, SwarmEnv
except ImportError:  # pragma: no cover - direct-script execution
    from env import ALTITUDE, DT, MAX_SPEED, ROLES, SwarmEnv  # type: ignore

HOST = "0.0.0.0"
PORT = 8765
HZ = 10.0

_CLIENTS: set = set()


def _try_cf2x_urdf() -> str | None:
    candidates = (
        ("gym_pybullet_drones.assets", "cf2x.urdf"),
        ("gym_pybullet_drones.assets", "cf2p.urdf"),
    )
    for package, name in candidates:
        try:
            ref = resources.files(package).joinpath(name)
            if ref.is_file():
                return str(ref)
        except (ModuleNotFoundError, FileNotFoundError):
            continue
    return None


def _mission_routes(env_id: str, n_agents: int) -> list[list[tuple[float, float]]]:
    templates: dict[str, list[list[tuple[float, float]]]] = {
        "drone-vs-drone": [
            [(-8, -4), (-4, -2), (-1.2, 0), (-4, 2), (-8, 4)],
            [(-8, 0), (-4, -1.6), (-0.8, 0.6), (-4, 1.7)],
            [(-7, 4), (-3, 3), (-1.4, 0.7), (-5, -2.8)],
            [(8, -4), (4, -2), (1.4, 0), (4, 2), (8, 4)],
            [(8, 0), (4, -1.4), (1, -0.5), (4, 1.8)],
            [(7, 4), (3, 3), (1.2, 0.6), (5, -2.8)],
        ],
        "moving-target-track": [
            [(-8, -2), (-3, -4), (2, -3), (7, -1), (3, 2), (-4, 2)],
            [(-7, 3), (-2, 5), (4, 4), (8, 2), (2, -1), (-5, -1)],
            [(-5, 0), (0, -2), (5, -1), (6, 3), (0, 4)],
            [(-6, -4), (-1, -5), (5, -4), (8, 0), (1, 1)],
        ],
        "defend-asset": [
            [(-4, 0), (-2, 3.5), (2, 3.5), (4, 0), (2, -3.5), (-2, -3.5)],
            [(0, 4.5), (3.8, 1.5), (2.3, -3.8), (-2.8, -3.6), (-4, 1.2)],
            [(4.5, 0), (1.5, -3.8), (-3.8, -2.2), (-3.5, 2.6), (1.4, 4)],
            [(0, -4.5), (-3.8, -1.5), (-2.3, 3.8), (2.8, 3.6), (4, -1.2)],
            [(-5.5, -5.5), (-4, 0), (-5.5, 5.5), (0, 4), (5.5, 5.5), (4, 0), (5.5, -5.5), (0, -4)],
        ],
        "swarm-vs-swarm-race": [
            [(-8, -7), (-5, -3), (-8, 1), (-5, 6), (-1, 4), (-2, -5)],
            [(-5, -8), (-2, -4), (-5, 1), (-1, 7), (2, 3), (1, -6)],
            [(-2, -7), (1, -3), (-1, 2), (2, 7), (5, 2), (4, -5)],
            [(8, 7), (5, 3), (8, -1), (5, -6), (1, -4), (2, 5)],
            [(5, 8), (2, 4), (5, -1), (1, -7), (-2, -3), (-1, 6)],
            [(2, 7), (-1, 3), (1, -2), (-2, -7), (-5, -2), (-4, 5)],
        ],
        "search-and-interdict": [
            [(-8, -7), (-3, -7), (2, -6), (8, -5), (8, -1), (2, -1), (-5, -2)],
            [(-8, 2), (-3, 3), (2, 2), (8, 1), (8, 6), (2, 7), (-6, 6)],
            [(-7, -3), (-3, -2), (0, 0), (4, 1), (8, 4)],
            [(7, 7), (4, 4), (1, 1), (-2, 0), (-6, 1)],
            [(-6, 7), (-1, 5), (4, 6), (7, 2), (3, -2), (-2, -4)],
        ],
    }
    base = templates.get(env_id, templates["search-and-interdict"])
    return [base[i % len(base)] for i in range(n_agents)]


def _route_point(route: list[tuple[float, float]], phase: float) -> tuple[float, float]:
    if len(route) == 1:
        return route[0]
    raw = math.floor(phase)
    i = raw % len(route)
    ax, ay = route[i]
    bx, by = route[(i + 1) % len(route)]
    t = phase - raw
    eased = t * t * (3 - 2 * t)
    return ax + (bx - ax) * eased, ay + (by - ay) * eased


@dataclass
class LiveAgentState:
    pos: np.ndarray
    vel: np.ndarray
    yaw: float
    alive: bool = True


class BaseRuntime:
    n_agents: int
    time_s: float

    def reset(self) -> None: ...
    def step(self, dt: float) -> None: ...
    def message(self) -> dict: ...
    def close(self) -> None: ...


class ScriptedPointMassRuntime(BaseRuntime):
    def __init__(self, env_id: str, n_agents: int, seed: int = 0) -> None:
        self.env_id = env_id
        self.env = SwarmEnv(n_agents=n_agents, seed=seed)
        self.n_agents = self.env.n
        self.routes = _mission_routes(env_id, self.n_agents)
        self.time_s = 0.0
        self.reset()

    def reset(self) -> None:
        self.env.reset()
        self.time_s = 0.0
        self.env.alive[:] = True
        for i, route in enumerate(self.routes):
            self.env.pos[i, 0] = route[0][0]
            self.env.pos[i, 1] = route[0][1]
        self.env.covered.fill(False)
        self.env._mark_covered()

    def _actions(self) -> np.ndarray:
        actions = np.zeros((self.env.n, 2), dtype=np.float32)
        phase_base = self.time_s / 5.2
        step_scale = MAX_SPEED * DT
        for i in range(self.env.n):
            if not self.env.alive[i]:
                continue
            tx, ty = _route_point(self.routes[i], phase_base + i * 0.18)
            px, py = self.env.pos[i]
            ax = (tx - px) / step_scale
            ay = (ty - py) / step_scale
            for j in range(self.env.n):
                if i == j or not self.env.alive[j]:
                    continue
                dx = px - self.env.pos[j, 0]
                dy = py - self.env.pos[j, 1]
                d2 = dx * dx + dy * dy
                if 0.0001 < d2 < 2.8:
                    push = (2.8 - d2) / 2.8
                    inv = math.sqrt(d2)
                    ax += (dx / inv) * push * 0.85
                    ay += (dy / inv) * push * 0.85
            mag = max(1.0, math.hypot(ax, ay))
            actions[i, 0] = max(-1.0, min(1.0, ax / mag))
            actions[i, 1] = max(-1.0, min(1.0, ay / mag))
        return actions

    def step(self, dt: float) -> None:
        self.time_s += dt
        self.env.step(self._actions())

    def message(self) -> dict:
        agents = []
        for i in range(self.env.n):
            vx, vy = float(self.env.vel[i, 0]), float(self.env.vel[i, 1])
            yaw = math.atan2(vy, vx) if (vx or vy) else 0.0
            role = ROLES[self.env.roles[i]] if self.env.roles[i] < len(ROLES) else "scout"
            agents.append(
                {
                    "id": i,
                    "x": round(float(self.env.pos[i, 0]), 3),
                    "y": round(float(self.env.pos[i, 1]), 3),
                    "z": round(float(ALTITUDE), 3),
                    "yaw": round(float(yaw), 3),
                    "role": role,
                    "alive": bool(self.env.alive[i]),
                }
            )
        return {"t": round(self.time_s, 3), "comms": "denied", "agents": agents}

    def close(self) -> None:
        return


class ScriptedPyBulletRuntime(BaseRuntime):
    def __init__(self, env_id: str = "search-and-interdict", n_agents: int = 5) -> None:
        import pybullet as p
        import pybullet_data

        self.p = p
        self.pybullet_data = pybullet_data
        self.env_id = env_id
        self.n_agents = n_agents
        self.routes = _mission_routes(env_id, n_agents)
        self.time_s = 0.0
        self.physics_hz = 240.0
        self.altitude = 1.8
        self.client = self.p.connect(self.p.DIRECT)
        self.p.setAdditionalSearchPath(self.pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        self.p.setTimeStep(1.0 / self.physics_hz)
        self.plane = self.p.loadURDF("plane.urdf")
        self.urdf = _try_cf2x_urdf()
        self.bodies: list[int] = []
        self.states: list[LiveAgentState] = []
        self.reset()

    def _spawn_body(self, position: tuple[float, float, float]) -> int:
        if self.urdf:
            return self.p.loadURDF(self.urdf, list(position), globalScaling=1.4)
        return _make_fallback_drone(self.p, position)

    def reset(self) -> None:
        for body in getattr(self, "bodies", []):
            self.p.removeBody(body)
        self.bodies = []
        self.states = []
        self.time_s = 0.0
        for i in range(self.n_agents):
            start = self.routes[i][0]
            position = (start[0], start[1], self.altitude + 0.08 * math.sin(i))
            body = self._spawn_body(position)
            self.bodies.append(body)
            self.states.append(
                LiveAgentState(
                    pos=np.array(position, dtype=np.float32),
                    vel=np.zeros(3, dtype=np.float32),
                    yaw=0.0,
                )
            )

    def step(self, dt: float) -> None:
        substeps = max(1, round(dt * self.physics_hz))
        target_time = self.time_s + dt
        for _ in range(substeps):
            t = self.time_s
            for i, body in enumerate(self.bodies):
                state = self.states[i]
                tx, ty = _route_point(self.routes[i], t / 5.2 + i * 0.18)
                pos, quat = self.p.getBasePositionAndOrientation(body)
                vel, _ = self.p.getBaseVelocity(body)
                cur = np.asarray(pos, dtype=np.float32)
                lin = np.asarray(vel, dtype=np.float32)
                target = np.array([tx, ty, self.altitude + 0.12 * math.sin(t * 0.7 + i)], dtype=np.float32)
                err = target - cur
                force = np.array(
                    [5.6 * err[0] - 1.7 * lin[0], 5.6 * err[1] - 1.7 * lin[1], 9.81 + 7.2 * err[2] - 2.4 * lin[2]],
                    dtype=np.float32,
                )
                # Neighbor repulsion keeps the scene legible.
                for j, other_body in enumerate(self.bodies):
                    if i == j:
                        continue
                    other_pos, _ = self.p.getBasePositionAndOrientation(other_body)
                    dx = cur[0] - other_pos[0]
                    dy = cur[1] - other_pos[1]
                    d2 = dx * dx + dy * dy
                    if 0.0001 < d2 < 3.4:
                        gain = (3.4 - d2) / 3.4
                        inv = math.sqrt(d2)
                        force[0] += (dx / inv) * gain * 1.2
                        force[1] += (dy / inv) * gain * 1.2
                self.p.applyExternalForce(body, -1, force.tolist(), pos, self.p.WORLD_FRAME)
                yaw = math.atan2(force[1], force[0]) if (force[0] or force[1]) else state.yaw
                orientation = self.p.getQuaternionFromEuler([0.0, 0.0, yaw])
                self.p.resetBasePositionAndOrientation(body, pos, orientation)
                state.yaw = yaw
                state.pos = cur
                state.vel = lin
                _ = quat
            self.p.stepSimulation()
            self.time_s += 1.0 / self.physics_hz
        self.time_s = target_time
        for i, body in enumerate(self.bodies):
            pos, quat = self.p.getBasePositionAndOrientation(body)
            vel, _ = self.p.getBaseVelocity(body)
            self.states[i].pos = np.asarray(pos, dtype=np.float32)
            self.states[i].vel = np.asarray(vel, dtype=np.float32)
            self.states[i].yaw = self.p.getEulerFromQuaternion(quat)[2]

    def message(self) -> dict:
        agents = []
        for i, state in enumerate(self.states):
            agents.append(
                {
                    "id": i,
                    "x": round(float(state.pos[0]), 3),
                    "y": round(float(state.pos[1]), 3),
                    "z": round(float(state.pos[2]), 3),
                    "yaw": round(float(state.yaw), 3),
                    "role": "scout",
                    "alive": state.alive,
                }
            )
        return {"t": round(self.time_s, 3), "comms": "denied", "agents": agents}

    def close(self) -> None:
        if getattr(self, "client", None) is not None:
            self.p.disconnect(self.client)
            self.client = None


async def _register(ws):
    _CLIENTS.add(ws)
    try:
        await ws.wait_closed()
    finally:
        _CLIENTS.discard(ws)


async def broadcast(topic: str, payload: dict) -> None:
    if not _CLIENTS:
        return
    msg = json.dumps({"topic": topic, **payload})
    dead = []
    for ws in list(_CLIENTS):
        try:
            await ws.send(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _CLIENTS.discard(ws)


def make_runtime(kind: str, env_id: str, n_agents: int) -> tuple[BaseRuntime, str]:
    if kind == "pybullet":
        try:
            return ScriptedPyBulletRuntime(env_id=env_id, n_agents=n_agents), "pybullet-scripted"
        except ModuleNotFoundError:
            print("[bus] pybullet stack unavailable; falling back to point-mass runtime")
    return ScriptedPointMassRuntime(env_id=env_id, n_agents=n_agents), "pointmass-scripted"


async def serve(
    *,
    backend: str,
    env_id: str,
    n_agents: int,
    host: str = HOST,
    port: int = PORT,
    hz: float = HZ,
) -> None:
    runtime, label = make_runtime(backend, env_id, n_agents)
    dt = 1.0 / hz
    try:
        async with websockets.serve(_register, host, port):
            print(f"[bus] swarm bus on ws://{host}:{port} (~{hz:.0f} Hz, backend={label}, env={env_id})")
            while True:
                runtime.step(dt)
                await broadcast("swarm", runtime.message())
                await asyncio.sleep(dt)
    finally:
        runtime.close()


def main() -> None:
    p = argparse.ArgumentParser(description="CombatOS swarm WebSocket bus")
    p.add_argument("--backend", choices=["pybullet", "pointmass"], default="pybullet")
    p.add_argument("--env-id", default="search-and-interdict")
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--hz", type=float, default=HZ)
    args = p.parse_args()

    try:
        asyncio.run(
            serve(
                backend=args.backend,
                env_id=args.env_id,
                n_agents=args.agents,
                port=args.port,
                hz=args.hz,
            )
        )
    except KeyboardInterrupt:
        print("\n[bus] stopped")


if __name__ == "__main__":
    main()
