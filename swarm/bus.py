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

import argparse
import asyncio
import json
import math
import os

import numpy as np
import websockets

# Support both `python -m swarm.bus` (from repo root) and `python bus.py` (in swarm/).
try:
    from .env import SwarmEnv, ROLES, ALTITUDE
except ImportError:  # pragma: no cover - direct-script execution
    from env import SwarmEnv, ROLES, ALTITUDE  # type: ignore

_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "policy.pt")

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


def _random_policy(env: SwarmEnv):
    """A policy_fn that ignores observations and acts randomly (Phase 0)."""
    def act(_obs):
        return env.rng.uniform(-1, 1, size=(env.n, 2)).astype(np.float32)
    return act


def _trained_policy(ckpt_path: str):
    """Load the trained MAPPO actor and return a policy_fn(obs) -> actions.

    The actor is the exact graph exported in Phase 2: obs (n,36) -> action (n,2),
    deterministic mean, tanh-squashed to [-1,1]. Local obs only — no comms.
    """
    import torch

    try:
        from .models import Actor
    except ImportError:  # pragma: no cover - direct-script execution
        from models import Actor  # type: ignore

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor = Actor(ck["obs_dim"], ck["act_dim"], ck["actor_hidden"], ck["log_std_init"])
    actor.load_state_dict(ck["actor_state_dict"])
    actor.eval()
    print(f"[bus] loaded trained actor from {ckpt_path} "
          f"(coverage≈{ck.get('coverage', '?')})")

    def act(obs):
        with torch.no_grad():
            return actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    return act


async def serve(policy_fn_factory, label: str,
                host: str = HOST, port: int = PORT, hz: float = HZ) -> None:
    """Serve the bus and stream a rollout driven by `policy_fn_factory(env)`."""
    env = SwarmEnv(seed=0)
    obs = env.reset()
    policy = policy_fn_factory(env)
    dt = 1.0 / hz
    killed_demo = False

    async with websockets.serve(_register, host, port):
        print(f"[bus] swarm bus on ws://{host}:{port}  "
              f"(~{hz:.0f} Hz, comms=denied, policy={label})")
        while True:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # demo: kill an agent partway through each episode
            if not killed_demo and env.steps == 200:
                env.kill(env.n - 1)
                killed_demo = True
                print("[bus] killed agent", env.n - 1, "(comms-denied re-cover demo)")

            await broadcast("swarm", swarm_message(env))

            if dones.all():
                obs = env.reset()
                killed_demo = False
            await asyncio.sleep(dt)


# Back-compat alias (Phase 0 entrypoint).
async def run_random(host: str = HOST, port: int = PORT, hz: float = HZ) -> None:
    await serve(_random_policy, "random", host, port, hz)


def main() -> None:
    p = argparse.ArgumentParser(description="CombatOS swarm WebSocket bus")
    p.add_argument(
        "--policy", choices=["trained", "random"], default="trained",
        help="trained = MAPPO actor from checkpoint (default); random = Phase 0",
    )
    p.add_argument("--ckpt", default=_CKPT, help="path to policy.pt")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--hz", type=float, default=HZ)
    args = p.parse_args()

    if args.policy == "trained" and os.path.exists(args.ckpt):
        factory = lambda env: _trained_policy(args.ckpt)  # noqa: E731
        label = "trained"
    else:
        if args.policy == "trained":
            print(f"[bus] no checkpoint at {args.ckpt} — falling back to random policy")
        factory = _random_policy
        label = "random"

    try:
        asyncio.run(serve(factory, label, port=args.port, hz=args.hz))
    except KeyboardInterrupt:
        print("\n[bus] stopped")


if __name__ == "__main__":
    main()
