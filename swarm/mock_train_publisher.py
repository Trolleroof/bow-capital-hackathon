"""Mock WebSocket publisher for `topic: "train"` frontend development.

Runs a tiny WS broadcast server that emits shaped train progress events at a
steady cadence so frontend work can proceed without a real training loop.

Run:
    uv run --project swarm python -m swarm.mock_train_publisher --env-id drone-vs-drone
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from typing import Any

import websockets

HOST = "0.0.0.0"
PORT = 8766
HZ = 2.0

CLIENTS: set[Any] = set()


async def register(ws):
    CLIENTS.add(ws)
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)


async def broadcast(payload: dict) -> None:
    if not CLIENTS:
        return
    msg = json.dumps({"topic": "train", **payload})
    dead = []
    for ws in list(CLIENTS):
        try:
            await ws.send(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)


def make_payload(env_id: str, profile: str, idx: int, total_steps: int) -> dict:
    step = min(total_steps, idx * 800)
    progress = min(1.0, step / max(total_steps, 1))
    reward = 8.0 + progress * 42.0 + math.sin(idx / 3) * 2.5
    coverage = min(0.99, 0.18 + progress * 0.74)
    return {
        "env_id": env_id,
        "profile": profile,
        "phase": "update" if progress < 1.0 else "final",
        "step": step,
        "reward_mean": round(reward, 4),
        "coverage": round(coverage, 4),
        "losses": {
            "pg_loss": round(max(0.01, 0.18 - progress * 0.11), 4),
            "v_loss": round(max(0.02, 0.42 - progress * 0.23), 4),
            "entropy": round(max(0.01, 0.12 - progress * 0.08), 4),
            "approx_kl": round(0.01 + progress * 0.015, 4),
        },
        "params_hash": f"mock-{env_id[:6]}",
    }


async def main_loop(env_id: str, profile: str, host: str, port: int, hz: float, timesteps: int) -> None:
    async with websockets.serve(register, host, port):
        print(f"[mock-train] ws://{host}:{port} topic=train env_id={env_id} profile={profile}")
        idx = 0
        while True:
            await broadcast(make_payload(env_id, profile, idx, timesteps))
            idx += 1
            await asyncio.sleep(1.0 / hz)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="search-and-interdict")
    parser.add_argument("--profile", choices=["garrison", "combat"], default="combat")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--hz", type=float, default=HZ)
    parser.add_argument("--timesteps", type=int, default=300_000)
    args = parser.parse_args()
    try:
        asyncio.run(main_loop(args.env_id, args.profile, HOST, args.port, args.hz, args.timesteps))
    except KeyboardInterrupt:
        print("\n[mock-train] stopped")


if __name__ == "__main__":
    main()
