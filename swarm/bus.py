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
import base64
import io
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
PYBULLET_FRAME_HZ = 8.0
PYBULLET_WIDTH = 1280
PYBULLET_HEIGHT = 720

# connected clients
_CLIENTS: set = set()


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
    msg: dict = {"t": round(float(env.t), 3), "comms": "denied", "agents": agents}
    # Expose target position so the PyBullet renderer can draw it correctly.
    if hasattr(env, "target_pos") and env.target_pos is not None:
        msg["target_pos"] = [round(float(env.target_pos[0]), 3), round(float(env.target_pos[1]), 3)]
    return msg


def _random_policy(env: SwarmEnv):
    """A policy_fn that ignores observations and acts randomly (Phase 0)."""
    def act(_obs):
        return env.rng.uniform(-1, 1, size=(env.n, 2)).astype(np.float32)
    return act


def _trained_policy(ckpt_path: str):
    """Load the trained MAPPO actor and return a policy_fn(obs) -> actions.

    The actor is the exact graph exported in Phase 2: obs (n,OBS_DIM) ->
    action (n,2), deterministic mean, tanh-squashed to [-1,1]. Local obs only
    — no comms. OBS_DIM is read from the checkpoint, so the same loader works
    against both legacy 36-dim policies and the new 48-dim obstacle-aware ones.
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


class PyBulletSwarmRuntime:
    """Drive the learned swarm env and render it through a real PyBullet camera."""

    def __init__(self, env_id: str, policy_fn_factory, label: str) -> None:
        try:
            import pybullet as p
            from PIL import Image
        except ModuleNotFoundError as exc:  # pragma: no cover - local dependency gate
            raise RuntimeError(
                "PyBullet sim dependencies are missing. Run `uv sync --project swarm`."
            ) from exc

        self.p = p
        self.Image = Image
        self.env_id = env_id
        self.label = label
        self.env = SwarmEnv(seed=0)
        self.obs = self.env.reset()
        self.policy = policy_fn_factory(self.env)
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        self._build_world()
        self.bodies = self._spawn_drones()

    def close(self) -> None:
        self.p.disconnect(self.client)

    def _box(
        self,
        half_extents: list[float],
        position: list[float],
        color: list[float],
        mass: float = 0.0,
    ) -> int:
        visual = self.p.createVisualShape(
            self.p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
        )
        collision = self.p.createCollisionShape(
            self.p.GEOM_BOX,
            halfExtents=half_extents,
        )
        return self.p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
        )

    def _cylinder(
        self,
        radius: float,
        height: float,
        position: list[float],
        color: list[float],
    ) -> int:
        visual = self.p.createVisualShape(
            self.p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
        )
        collision = self.p.createCollisionShape(
            self.p.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )
        return self.p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
        )

    def _build_world(self) -> None:
        self._box([12.0, 12.0, 0.015], [0, 0, -0.015], [0.025, 0.038, 0.035, 1])
        self._box([10.5, 0.04, 0.018], [0, -10.5, 0.02], [0.18, 0.54, 0.4, 1])
        self._box([10.5, 0.04, 0.018], [0, 10.5, 0.02], [0.18, 0.54, 0.4, 1])
        self._box([0.04, 10.5, 0.018], [-10.5, 0, 0.02], [0.18, 0.54, 0.4, 1])
        self._box([0.04, 10.5, 0.018], [10.5, 0, 0.02], [0.18, 0.54, 0.4, 1])

        for x in range(-8, 9, 4):
            self._box([0.02, 10.0, 0.01], [x, 0, 0.015], [0.08, 0.18, 0.16, 1])
            self._box([10.0, 0.02, 0.01], [0, x, 0.016], [0.08, 0.18, 0.16, 1])

        layouts = {
            "drone-vs-drone": [(-3.0, -1.8), (3.0, 1.8), (0.0, 3.2)],
            "moving-target-track": [(-4.0, 0.0), (0.0, 2.4), (4.0, -1.6)],
            "search-and-interdict": [(-3.8, -3.2), (-1.4, 2.8), (2.8, 1.1), (4.3, -3.4)],
            "defend-asset": [(-4.4, 0.0), (4.4, 0.0), (0.0, -4.4), (0.0, 4.4)],
            "swarm-vs-swarm-race": [(-5.0, -2.0), (-1.8, 2.0), (1.8, -2.0), (5.0, 2.0)],
        }
        for idx, (x, y) in enumerate(layouts.get(self.env_id, layouts["search-and-interdict"])):
            height = 0.45 + 0.18 * (idx % 2)
            self._box(
                [0.7, 0.45, height],
                [x, y, height],
                [0.11, 0.24, 0.22, 1],
            )

        if self.env_id == "defend-asset":
            self._cylinder(0.95, 0.2, [0.0, 0.0, 0.1], [0.28, 0.78, 0.56, 1])
        elif self.env_id == "moving-target-track":
            self._box([0.6, 0.32, 0.16], [2.5, -2.7, 0.18], [0.88, 0.34, 0.16, 1])
        elif self.env_id == "swarm-vs-swarm-race":
            for x in (-6.0, -2.0, 2.0, 6.0):
                self._box([1.25, 0.25, 0.018], [x, 0, 0.03], [0.13, 0.38, 0.26, 1])

    def _spawn_drones(self) -> list[int]:
        bodies = []
        for i in range(self.env.n):
            x, y = self.env.pos[i]
            color = [0.22, 0.78, 1.0, 1.0] if i % 2 == 0 else [1.0, 0.54, 0.22, 1.0]
            visual = self.p.createVisualShape(
                self.p.GEOM_BOX,
                halfExtents=[0.28, 0.18, 0.06],
                rgbaColor=color,
            )
            collision = self.p.createCollisionShape(
                self.p.GEOM_BOX,
                halfExtents=[0.28, 0.18, 0.06],
            )
            body = self.p.createMultiBody(
                baseMass=0.85,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[float(x), float(y), float(ALTITUDE)],
            )
            bodies.append(body)
        return bodies

    def step(self, dt: float) -> dict:
        actions = self.policy(self.obs)
        self.obs, _, dones, _ = self.env.step(actions)
        if dones.all():
            self.obs = self.env.reset()

        substeps = max(1, round(240 * dt))
        for _ in range(substeps):
            for i, body in enumerate(self.bodies):
                target = np.array(
                    [self.env.pos[i, 0], self.env.pos[i, 1], ALTITUDE],
                    dtype=np.float32,
                )
                pos, _ = self.p.getBasePositionAndOrientation(body)
                vel, _ = self.p.getBaseVelocity(body)
                err = target - np.asarray(pos, dtype=np.float32)
                cmd = 5.8 * err - 1.55 * np.asarray(vel, dtype=np.float32)
                force = [float(cmd[0]), float(cmd[1]), float(9.81 + cmd[2])]
                self.p.applyExternalForce(body, -1, force, pos, self.p.WORLD_FRAME)
                yaw = math.atan2(float(actions[i, 1]), float(actions[i, 0]))
                quat = self.p.getQuaternionFromEuler([0.0, 0.0, yaw])
                self.p.resetBasePositionAndOrientation(body, pos, quat)
            self.p.stepSimulation()
        return self.swarm_message()

    def swarm_message(self) -> dict:
        agents = []
        for i, body in enumerate(self.bodies):
            pos, quat = self.p.getBasePositionAndOrientation(body)
            yaw = self.p.getEulerFromQuaternion(quat)[2]
            role = ROLES[self.env.roles[i]] if self.env.roles[i] < len(ROLES) else "scout"
            agents.append(
                {
                    "id": i,
                    "x": round(float(pos[0]), 3),
                    "y": round(float(pos[1]), 3),
                    "z": round(float(pos[2]), 3),
                    "yaw": round(float(yaw), 3),
                    "role": role,
                    "alive": bool(self.env.alive[i]),
                }
            )
        covered = int(self.env.covered.sum())
        total = int(self.env.covered.size)
        return {
            "t": round(float(self.env.t), 3),
            "comms": "denied",
            "policy": self.label,
            "env_id": self.env_id,
            "coverage": round(covered / total, 3) if total else 0.0,
            "agents": agents,
        }

    def frame_message(self) -> dict:
        live_positions = np.asarray(
            [[agent["x"], agent["y"], agent["z"]] for agent in self.swarm_message()["agents"]],
            dtype=np.float32,
        )
        center = live_positions.mean(axis=0) if len(live_positions) else np.array([0.0, 0.0, 1.0])
        target = [float(center[0]), float(center[1]), 1.0]
        eye = [target[0] + 7.2, target[1] - 8.8, 5.2]
        view = self.p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=[0.0, 0.0, 1.0],
        )
        proj = self.p.computeProjectionMatrixFOV(
            58,
            PYBULLET_WIDTH / PYBULLET_HEIGHT,
            0.1,
            80,
        )
        _, _, rgba, _, _ = self.p.getCameraImage(
            PYBULLET_WIDTH,
            PYBULLET_HEIGHT,
            view,
            proj,
            renderer=self.p.ER_TINY_RENDERER,
        )
        rgb = np.reshape(np.asarray(rgba, dtype=np.uint8), (PYBULLET_HEIGHT, PYBULLET_WIDTH, 4))[:, :, :3]
        image = self.Image.fromarray(rgb, "RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=76, optimize=True)
        return {
            "t": round(float(self.env.t), 3),
            "env_id": self.env_id,
            "width": PYBULLET_WIDTH,
            "height": PYBULLET_HEIGHT,
            "encoding": "jpeg",
            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        }


async def serve_pybullet(
    env_id: str,
    policy_fn_factory,
    label: str,
    host: str = HOST,
    port: int = PORT,
    hz: float = HZ,
    frame_hz: float = PYBULLET_FRAME_HZ,
) -> None:
    runtime = PyBulletSwarmRuntime(env_id, policy_fn_factory, label)
    dt = 1.0 / hz
    frame_every = max(1, round(hz / max(1.0, frame_hz)))
    tick = 0

    try:
        async with websockets.serve(_register, host, port, max_size=4 * 1024 * 1024):
            print(
                f"[bus] pybullet sim on ws://{host}:{port} "
                f"(env={env_id}, ~{hz:.0f} Hz, frames~{frame_hz:.0f} Hz, policy={label})"
            )
            while True:
                swarm = runtime.step(dt)
                await broadcast("swarm", swarm)
                if tick % frame_every == 0:
                    await broadcast("pybullet_frame", runtime.frame_message())
                tick += 1
                await asyncio.sleep(dt)
    finally:
        runtime.close()


# Back-compat alias (Phase 0 entrypoint).
async def run_random(host: str = HOST, port: int = PORT, hz: float = HZ) -> None:
    await serve(_random_policy, "random", host, port, hz)


def main() -> None:
    p = argparse.ArgumentParser(description="CombatOS swarm WebSocket bus")
    p.add_argument(
        "--backend",
        choices=["pointmass", "pybullet"],
        default="pointmass",
        help="pointmass streams JSON poses; pybullet streams poses plus camera frames",
    )
    p.add_argument("--env-id", default="search-and-interdict")
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
        if args.backend == "pybullet":
            asyncio.run(
                serve_pybullet(
                    args.env_id,
                    factory,
                    label,
                    port=args.port,
                    hz=args.hz,
                )
            )
        else:
            asyncio.run(serve(factory, label, port=args.port, hz=args.hz))
    except KeyboardInterrupt:
        print("\n[bus] stopped")


if __name__ == "__main__":
    main()
