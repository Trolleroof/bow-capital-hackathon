"""Record Phase-6 quadrotor B-roll without touching the trained policy or bus.

This is intentionally a sandbox. It runs a PyBullet-only drone shot, writes a
camera clip, and exports the same pose stream shape the frontend compositor can
overlay later. If the optional gym-pybullet-drones package is present, its
Crazyflie URDF is used; otherwise the script falls back to a simple PyBullet body
so the integration contract is still testable.

Setup:
    cd swarm
    uv sync --extra drones

Run:
    uv run python drones_broll.py --out artifacts/drones_broll.mp4
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np


@dataclass
class DroneState:
    x: float
    y: float
    z: float
    yaw: float


def _try_cf2x_urdf() -> str | None:
    """Return a gym-pybullet-drones Crazyflie URDF path when installed."""
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


def _make_fallback_drone(pybullet, position: tuple[float, float, float]) -> int:
    """Create a lightweight quadrotor-ish body if no packaged URDF is present."""
    collision = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.18, 0.18, 0.04])
    visual = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=[0.18, 0.18, 0.04],
        rgbaColor=[0.15, 0.95, 0.62, 1.0],
    )
    return pybullet.createMultiBody(
        baseMass=0.9,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def _waypoint(t: float, idx: int, n: int) -> np.ndarray:
    phase = idx * 2.0 * math.pi / max(1, n)
    radius = 2.8 + 0.28 * math.sin(t * 0.45 + idx)
    return np.array(
        [
            radius * math.cos(t * 0.32 + phase),
            radius * math.sin(t * 0.32 + phase),
            1.25 + 0.32 * math.sin(t * 0.85 + phase),
        ],
        dtype=np.float32,
    )


def _state_from_body(pybullet, body: int) -> DroneState:
    pos, quat = pybullet.getBasePositionAndOrientation(body)
    yaw = pybullet.getEulerFromQuaternion(quat)[2]
    return DroneState(float(pos[0]), float(pos[1]), float(pos[2]), float(yaw))


def _draw_disk(frame: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    h, w, _ = frame.shape
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    frame[y0:y1, x0:x1][mask] = color


def _record_kinematic(out: Path, trajectory_out: Path, seconds: float, fps: int, n: int) -> None:
    import imageio.v2 as imageio

    out.parent.mkdir(parents=True, exist_ok=True)
    trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    frames: list[dict] = []
    width, height = 1280, 720

    for frame_idx in range(round(seconds * fps)):
        t = frame_idx / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, :] = np.array([7, 11, 13], dtype=np.uint8)
        for row in range(0, height, 36):
            frame[row : row + 1, :, :] = np.array([19, 35, 30], dtype=np.uint8)
        for col in range(0, width, 64):
            frame[:, col : col + 1, :] = np.array([19, 35, 30], dtype=np.uint8)

        swarm = []
        for i in range(n):
            target = _waypoint(t, i, n)
            yaw = math.atan2(float(target[1]), float(target[0]))
            swarm.append(
                {
                    "id": i,
                    "x": round(float(target[0]), 3),
                    "y": round(float(target[1]), 3),
                    "z": round(float(target[2]), 3),
                    "yaw": round(yaw, 3),
                    "role": "scout",
                    "alive": True,
                }
            )
            sx = int(width * 0.5 + target[0] * 72 + target[1] * 20)
            sy = int(height * 0.56 - target[1] * 42 - target[2] * 72)
            _draw_disk(frame, sx + 9, sy + 12, 18, (0, 0, 0))
            _draw_disk(frame, sx, sy, 13, (78, 240, 160))
            _draw_disk(frame, sx, sy, 4, (230, 246, 238))

        frames.append({"t": round(t, 3), "agents": swarm})
        writer.append_data(frame)

    writer.close()
    trajectory_out.write_text(json.dumps({"source": "kinematic-fallback-broll", "fps": fps, "frames": frames}, indent=2))
    print(f"wrote {out}")
    print(f"wrote {trajectory_out}")
    print("pybullet unavailable; wrote deterministic kinematic fallback clip")


def record(out: Path, trajectory_out: Path, seconds: float, fps: int, n: int) -> None:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise SystemExit("Install the drones extra first: uv sync --extra drones") from exc

    try:
        import pybullet as p
        import pybullet_data
    except ModuleNotFoundError:
        _record_kinematic(out, trajectory_out, seconds, fps, n)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    trajectory_out.parent.mkdir(parents=True, exist_ok=True)

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.loadURDF("plane.urdf")

    urdf = _try_cf2x_urdf()
    bodies: list[int] = []
    for i in range(n):
        start = _waypoint(0, i, n)
        if urdf:
            bodies.append(p.loadURDF(urdf, start.tolist(), globalScaling=1.4))
        else:
            bodies.append(_make_fallback_drone(p, tuple(float(v) for v in start)))

    frames: list[dict] = []
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    steps_per_frame = max(1, round(240 / fps))

    for frame_idx in range(round(seconds * fps)):
        t = frame_idx / fps
        for _ in range(steps_per_frame):
            for i, body in enumerate(bodies):
                target = _waypoint(t, i, n)
                pos, _ = p.getBasePositionAndOrientation(body)
                vel, _ = p.getBaseVelocity(body)
                err = target - np.asarray(pos, dtype=np.float32)
                cmd = 5.2 * err - 1.6 * np.asarray(vel, dtype=np.float32)
                force = np.array([cmd[0], cmd[1], 9.81 + cmd[2]], dtype=np.float32)
                p.applyExternalForce(body, -1, force.tolist(), pos, p.WORLD_FRAME)
                yaw = math.atan2(float(cmd[1]), float(cmd[0]))
                p.resetBaseVelocity(body, angularVelocity=[0.0, 0.0, 0.8 * yaw])
            p.stepSimulation()

        swarm = []
        for i, body in enumerate(bodies):
            state = _state_from_body(p, body)
            swarm.append(
                {
                    "id": i,
                    "x": round(state.x, 3),
                    "y": round(state.y, 3),
                    "z": round(state.z, 3),
                    "yaw": round(state.yaw, 3),
                    "role": "scout",
                    "alive": True,
                }
            )
        frames.append({"t": round(t, 3), "agents": swarm})

        view = p.computeViewMatrix(
            cameraEyePosition=[5.6, -7.4, 4.2],
            cameraTargetPosition=[0.0, 0.0, 1.1],
            cameraUpVector=[0.0, 0.0, 1.0],
        )
        proj = p.computeProjectionMatrixFOV(55, 16 / 9, 0.1, 80)
        _, _, rgba, _, _ = p.getCameraImage(1280, 720, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.reshape(rgba, (720, 1280, 4))[:, :, :3]
        writer.append_data(rgb)

    writer.close()
    p.disconnect(client)
    trajectory_out.write_text(json.dumps({"source": "pybullet-drones-broll", "fps": fps, "frames": frames}, indent=2))
    print(f"wrote {out}")
    print(f"wrote {trajectory_out}")
    print("used gym-pybullet-drones Crazyflie URDF" if urdf else "used fallback PyBullet body")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record CombatOS Phase-6 drone B-roll")
    parser.add_argument("--out", type=Path, default=Path("artifacts/drones_broll.mp4"))
    parser.add_argument("--trajectory-out", type=Path, default=Path("artifacts/drones_broll_trajectory.json"))
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--agents", type=int, default=5)
    args = parser.parse_args()
    record(args.out, args.trajectory_out, args.seconds, args.fps, args.agents)


if __name__ == "__main__":
    main()
