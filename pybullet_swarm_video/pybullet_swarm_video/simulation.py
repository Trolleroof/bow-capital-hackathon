from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import SimulationConfig
from .policies import ScriptedSurveillancePolicy

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover - import guard only
    raise RuntimeError(
        "pybullet is required for the PyBullet swarm video prototype. "
        "Install dependencies with `uv sync --project pybullet_swarm_video`."
    ) from exc


@dataclass
class SimulationSnapshot:
    sim_time: float
    drone_positions: np.ndarray
    troop_positions: np.ndarray


@dataclass
class CameraPose:
    eye: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    width: int
    height: int
    fov_deg: float


class DroneSurveillanceSimulation:
    def __init__(self, config: SimulationConfig, gui: bool = False) -> None:
        self.config = config
        self.gui = gui
        self.client_id: int | None = None

        self.drone_ids: list[int] = []
        self.troop_ids: list[int] = []
        self.drone_positions = np.zeros((config.num_drones, 3), dtype=np.float32)
        self.drone_velocities = np.zeros((config.num_drones, 3), dtype=np.float32)
        self.drone_yaws = np.zeros(config.num_drones, dtype=np.float32)
        self.troop_positions = np.zeros((config.num_troops, 3), dtype=np.float32)
        self.troop_yaws = np.zeros(config.num_troops, dtype=np.float32)
        self.sim_time = 0.0

        self.policy = ScriptedSurveillancePolicy(
            num_drones=config.num_drones,
            ring_radius_m=config.drone_ring_radius_m,
            cruise_altitude_m=config.drone_altitude_m,
            max_speed_mps=config.drone_speed_mps,
            separation_gain=config.drone_separation_gain,
        )

    def __enter__(self) -> DroneSurveillanceSimulation:
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self) -> None:
        self.close()
        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.config.time_step, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        if self.gui:
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI,
                0,
                physicsClientId=self.client_id,
            )

        self._spawn_ground_markers()
        self._spawn_troops()
        self._spawn_drones()
        self.sim_time = 0.0

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = None

    def _spawn_ground_markers(self) -> None:
        size = self.config.world_half_extent_m
        marker_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size * 0.65, size * 0.35, 0.05],
            rgbaColor=[0.28, 0.31, 0.19, 0.55],
            physicsClientId=self.client_id,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=marker_visual,
            basePosition=[0.0, 0.0, 0.05],
            physicsClientId=self.client_id,
        )

        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size, 0.15, 0.7],
            rgbaColor=[0.17, 0.18, 0.2, 1.0],
            physicsClientId=self.client_id,
        )
        for sign in (-1.0, 1.0):
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=wall_visual,
                basePosition=[0.0, sign * size, 0.7],
                physicsClientId=self.client_id,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=wall_visual,
                basePosition=[sign * size, 0.0, 0.7],
                baseOrientation=p.getQuaternionFromEuler(
                    [0.0, 0.0, math.pi / 2.0]
                ),
                physicsClientId=self.client_id,
            )

    def _spawn_troops(self) -> None:
        collision = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=0.18,
            height=1.0,
            physicsClientId=self.client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=0.18,
            length=1.0,
            rgbaColor=[0.32, 0.42, 0.24, 1.0],
            specularColor=[0.1, 0.1, 0.1],
            physicsClientId=self.client_id,
        )
        cols = max(4, math.ceil(math.sqrt(self.config.num_troops)))
        rows = math.ceil(self.config.num_troops / cols)
        base_x = -self.config.world_half_extent_m * 0.55
        base_y = -0.5 * rows * self.config.troop_spacing_m

        self.troop_ids = []
        for idx in range(self.config.num_troops):
            row = idx // cols
            col = idx % cols
            x = base_x + col * self.config.troop_spacing_m
            y = base_y + row * self.config.troop_spacing_m
            z = 1.0
            body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, z],
                physicsClientId=self.client_id,
            )
            self.troop_ids.append(body)
            self.troop_positions[idx] = (x, y, z)
            self.troop_yaws[idx] = 0.0

    def _spawn_drones(self) -> None:
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.08],
            physicsClientId=self.client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.08],
            rgbaColor=[0.18, 0.18, 0.24, 1.0],
            specularColor=[0.5, 0.5, 0.5],
            physicsClientId=self.client_id,
        )
        center = self._troop_centroid()

        self.drone_ids = []
        for idx in range(self.config.num_drones):
            theta = 2.0 * math.pi * idx / self.config.num_drones
            x = center[0] + self.config.drone_ring_radius_m * math.cos(theta)
            y = center[1] + self.config.drone_ring_radius_m * math.sin(theta)
            z = self.config.drone_altitude_m
            yaw = math.atan2(center[1] - y, center[0] - x)
            body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, z],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )
            self.drone_ids.append(body)
            self.drone_positions[idx] = (x, y, z)
            self.drone_yaws[idx] = yaw

    def _troop_centroid(self) -> np.ndarray:
        return self.troop_positions.mean(axis=0)

    def step(self) -> SimulationSnapshot:
        self._advance_troops()
        self._advance_drones()
        if self.gui:
            self._update_observer_camera()
        p.stepSimulation(physicsClientId=self.client_id)
        self.sim_time += self.config.time_step
        return SimulationSnapshot(
            sim_time=self.sim_time,
            drone_positions=self.drone_positions.copy(),
            troop_positions=self.troop_positions.copy(),
        )

    def _update_observer_camera(self) -> None:
        focus = self._troop_centroid()
        mean_drone = self.drone_positions.mean(axis=0)
        look_target = [
            round(float((focus[0] + mean_drone[0]) * 0.5), 3),
            round(float((focus[1] + mean_drone[1]) * 0.5), 3),
            round(float(focus[2] + 2.0), 3),
        ]
        yaw = 38.0 + 18.0 * math.sin(self.sim_time * 0.08)
        pitch = -36.0
        distance = max(18.0, self.config.world_half_extent_m * 0.95)
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=look_target,
            physicsClientId=self.client_id,
        )

    def _advance_troops(self) -> None:
        lane_bias = 4.0 * math.sin(self.sim_time * 0.18)
        heading = math.atan2(0.18 * 4.0 * math.cos(self.sim_time * 0.18), 1.0)

        cols = max(4, math.ceil(math.sqrt(self.config.num_troops)))
        rows = math.ceil(self.config.num_troops / cols)
        route_start_x = -self.config.world_half_extent_m * 0.55
        route_end_x = self.config.world_half_extent_m * 0.35
        route_length = route_end_x - route_start_x
        route_progress = (self.sim_time * self.config.troop_stride_mps) % route_length
        base_x = route_start_x + route_progress
        base_y = -0.5 * rows * self.config.troop_spacing_m + lane_bias

        for idx, body in enumerate(self.troop_ids):
            row = idx // cols
            col = idx % cols
            jitter = 0.3 * math.sin(self.sim_time * 1.2 + idx * 0.7)
            x = base_x + col * self.config.troop_spacing_m + jitter * 0.3
            y = base_y + row * self.config.troop_spacing_m + jitter
            z = 1.0
            yaw = heading
            self.troop_positions[idx] = (x, y, z)
            self.troop_yaws[idx] = yaw
            p.resetBasePositionAndOrientation(
                body,
                [x, y, z],
                p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )

    def _advance_drones(self) -> None:
        velocities, desired_yaws = self.policy.commands(
            sim_time=self.sim_time,
            drone_positions=self.drone_positions,
            troop_positions=self.troop_positions,
        )
        next_positions = self.drone_positions + velocities * self.config.time_step
        limit = self.config.world_half_extent_m - 1.0
        next_positions[:, 0:2] = np.clip(next_positions[:, 0:2], -limit, limit)
        next_positions[:, 2] = np.clip(next_positions[:, 2], 6.0, 20.0)

        self.drone_velocities = velocities
        self.drone_positions = next_positions.astype(np.float32)
        self.drone_yaws = desired_yaws.astype(np.float32)

        for idx, body in enumerate(self.drone_ids):
            p.resetBasePositionAndOrientation(
                body,
                self.drone_positions[idx].tolist(),
                p.getQuaternionFromEuler([0.0, 0.0, float(self.drone_yaws[idx])]),
                physicsClientId=self.client_id,
            )

    def camera_pose(self, drone_idx: int) -> CameraPose:
        cam_cfg = self.config.camera
        pos = self.drone_positions[drone_idx]
        yaw = float(self.drone_yaws[drone_idx])
        tilt_rad = math.radians(cam_cfg.tilt_deg)
        eye = pos + np.array(
            [
                cam_cfg.forward_offset_m * math.cos(yaw),
                cam_cfg.forward_offset_m * math.sin(yaw),
                -0.08,
            ],
            dtype=np.float32,
        )
        forward = np.array(
            [
                math.cos(yaw) * math.cos(tilt_rad),
                math.sin(yaw) * math.cos(tilt_rad),
                -math.sin(tilt_rad),
            ],
            dtype=np.float32,
        )
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= right_norm
        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-6)
        return CameraPose(
            eye=eye,
            forward=forward,
            up=up.astype(np.float32),
            width=cam_cfg.width,
            height=cam_cfg.height,
            fov_deg=cam_cfg.fov_deg,
        )

    def troop_targets(self) -> list[dict]:
        return [
            {
                "id": idx,
                "cls": "troop",
                "x": round(float(pos[0]), 4),
                "y": round(float(pos[1]), 4),
                "z": round(float(pos[2]), 4),
                "width_m": 0.55,
                "height_m": 1.75,
            }
            for idx, pos in enumerate(self.troop_positions)
        ]

    def render_drone_camera(self, drone_idx: int) -> np.ndarray:
        camera = self.camera_pose(drone_idx)
        target = camera.eye + camera.forward * 30.0
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera.eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=camera.up.tolist(),
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=camera.fov_deg,
            aspect=camera.width / camera.height,
            nearVal=self.config.camera.near,
            farVal=self.config.camera.far,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=camera.width,
            height=camera.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(camera.height, camera.width, 4)[
            :, :, :3
        ]

    def render_all_drone_cameras(self) -> list[np.ndarray]:
        return [self.render_drone_camera(idx) for idx in range(self.config.num_drones)]

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            sim_time=self.sim_time,
            drone_positions=self.drone_positions.copy(),
            troop_positions=self.troop_positions.copy(),
        )

    def drone_states(self) -> Sequence[tuple[np.ndarray, float]]:
        return [
            (self.drone_positions[idx].copy(), float(self.drone_yaws[idx]))
            for idx in range(self.config.num_drones)
        ]
