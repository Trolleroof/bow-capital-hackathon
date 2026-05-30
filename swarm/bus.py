"""WebSocket bus for the CombatOS swarm vertical (Phase 0).

A tiny broadcast server: every connected client receives JSON frames tagged with a
topic. This Phase-0 entrypoint drives `SwarmEnv` with a RANDOM policy and streams
the `swarm` topic at ~10 Hz, matching SWARM.md §4 / TEAM_PLAN §5:

    { "topic": "swarm",
      "t": 1234.56, "comms": "denied",
      "agents": [ {"id":0,"x":1.2,"y":-0.4,"z":2.1,"yaw":0.3,
                   "role":"scout","alive":true}, ... ] }

The dashboard (frontend/src/panels/SwarmPanel.tsx) subscribes and renders the agents.

Run:  cd swarm && uv run python -m swarm.bus
 or:  cd swarm && uv run python bus.py
"""

from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import websockets

# Support both `python -m swarm.bus` (from repo root) and `python bus.py` (in swarm/).
try:
    from .env import SwarmEnv, ROLES, ALTITUDE
except ImportError:  # pragma: no cover - direct-script execution
    from env import SwarmEnv, ROLES, ALTITUDE  # type: ignore

HOST = "0.0.0.0"
PORT = 8765
HZ = 10.0  # broadcast rate for the swarm topic

# connected clients
_CLIENTS: set = set()


async def _register(ws):
    _CLIENTS.add(ws)
    try:
        await ws.wait_closed()
    finally:
        _CLIENTS.discard(ws)


async def broadcast(topic: str, payload: dict) -> None:
    """Send a JSON frame on `topic` to every connected client."""
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


def swarm_message(env: SwarmEnv) -> dict:
    """Build the `swarm` bus payload from current env state (SWARM.md §4)."""
    agents = []
    for i in range(env.n):
        vx, vy = float(env.vel[i, 0]), float(env.vel[i, 1])
        yaw = math.atan2(vy, vx) if (vx or vy) else 0.0
        role = ROLES[env.roles[i]] if env.roles[i] < len(ROLES) else "scout"
        agents.append(
            {
                "id": i,
                "x": round(float(env.pos[i, 0]), 3),
                "y": round(float(env.pos[i, 1]), 3),
                "z": round(float(ALTITUDE), 3),
                "yaw": round(float(yaw), 3),
                "role": role,
                "alive": bool(env.alive[i]),
            }
        )
    return {"t": round(float(env.t), 3), "comms": "denied", "agents": agents}


async def run_random(host: str = HOST, port: int = PORT, hz: float = HZ) -> None:
    """Serve the bus and stream a RANDOM-policy rollout on the `swarm` topic."""
    env = SwarmEnv(seed=0)
    env.reset()
    dt = 1.0 / hz
    killed_demo = False

    async with websockets.serve(_register, host, port):
        print(f"[bus] swarm bus on ws://{host}:{port}  (~{hz:.0f} Hz, comms=denied)")
        while True:
            # RANDOM policy — no learning in Phase 0
            actions = env.rng.uniform(-1, 1, size=(env.n, 2)).astype(np.float32)
            _, _, dones, info = env.step(actions)

            # demo: kill an agent partway through each episode
            if not killed_demo and env.steps == 200:
                env.kill(env.n - 1)
                killed_demo = True
                print("[bus] killed agent", env.n - 1, "(comms-denied re-cover demo)")

            await broadcast("swarm", swarm_message(env))

            if dones.all():
                env.reset()
                killed_demo = False
            await asyncio.sleep(dt)


if __name__ == "__main__":
    try:
        asyncio.run(run_random())
    except KeyboardInterrupt:
        print("\n[bus] stopped")
