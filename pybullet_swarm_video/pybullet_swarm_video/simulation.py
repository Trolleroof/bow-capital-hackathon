from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
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


class SimulationDisconnectedError(RuntimeError):
    """Raised when the PyBullet client has disconnected during a running sim."""


class DroneSurveillanceSimulation:
    def __init__(self, config: SimulationConfig, gui: bool = False) -> None:
        self.config = config
        self.gui = gui
        self.client_id: int | None = None
        self.rng = np.random.default_rng(7)

        self.drone_ids: list[int] = []
        self.troop_ids: list[int] = []
        self._ruin_ids: list[int] = []
        self.drone_positions = np.zeros((config.num_drones, 3), dtype=np.float32)
        self.drone_velocities = np.zeros((config.num_drones, 3), dtype=np.float32)
        self.drone_yaws = np.zeros(config.num_drones, dtype=np.float32)
        self.troop_positions = np.zeros((config.num_troops, 3), dtype=np.float32)
        self.troop_yaws = np.zeros(config.num_troops, dtype=np.float32)
        self._troop_offsets = np.zeros((config.num_troops, 2), dtype=np.float32)
        self._troop_anchor_ids = np.zeros(config.num_troops, dtype=np.int32)
        self._troop_anchor_bases = np.zeros((4, 2), dtype=np.float32)
        self._troop_anchor_dirs = np.zeros(4, dtype=np.float32)
        self._troop_personal_phase = np.zeros(config.num_troops, dtype=np.float32)
        self.sim_time = 0.0
        self.camera_mode: Literal["observer", "chase", "fpv"] = "observer"
        self.selected_drone_id = 0
        self.manual_drone_id: int | None = None
        self._plane_id: int | None = None
        self._fpv_hidden_drone_id: int | None = None

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
        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
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
        if self.gui:
            self._print_controls()

    def close(self) -> None:
        if self.client_id is None:
            return
        try:
            if self.is_connected():
                p.disconnect(physicsClientId=self.client_id)
        except Exception:
            pass
        finally:
            self.client_id = None

    def is_connected(self) -> bool:
        if self.client_id is None:
            return False
        try:
            if hasattr(p, "getConnectionInfo"):
                info = p.getConnectionInfo(physicsClientId=self.client_id)
                if isinstance(info, dict) and "isConnected" in info:
                    return bool(info["isConnected"])
        except Exception:
            return False

        try:
            if hasattr(p, "isConnected"):
                try:
                    return bool(p.isConnected(self.client_id))
                except TypeError:
                    return bool(p.isConnected())
        except Exception:
            return False

        try:
            p.getPhysicsEngineParameters(physicsClientId=self.client_id)
            return True
        except Exception:
            return False

    def _require_connection(self) -> int:
        if not self.is_connected():
            raise SimulationDisconnectedError("PyBullet client disconnected")
        assert self.client_id is not None
        return self.client_id

    def _print_controls(self) -> None:
        print(
            "[sim] controls: C cycle camera | B observer | H chase | F fpv | 1-9 select drone | "
            "M toggle manual for selected drone | R return selected drone to scripted mode | "
            "I/K forward-back | J/L strafe | U/O altitude | Z/X yaw"
        )

    def _spawn_ground_markers(self) -> None:
        size = self.config.world_half_extent_m
        if self._plane_id is not None:
            p.changeVisualShape(
                self._plane_id,
                -1,
                rgbaColor=[0.36, 0.30, 0.21, 1.0],
                physicsClientId=self.client_id,
            )
        marker_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size * 0.68, size * 0.42, 0.05],
            rgbaColor=[0.33, 0.24, 0.17, 0.75],
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

        blast_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=1.5,
            length=0.04,
            rgbaColor=[0.15, 0.12, 0.11, 0.95],
            physicsClientId=self.client_id,
        )
        berm_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[2.2, 0.45, 0.45],
            rgbaColor=[0.42, 0.34, 0.22, 1.0],
            physicsClientId=self.client_id,
        )
        concrete_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[1.3, 1.3, 1.5],
            rgbaColor=[0.31, 0.31, 0.33, 1.0],
            physicsClientId=self.client_id,
        )
        wreck_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.9, 0.45, 0.35],
            rgbaColor=[0.20, 0.21, 0.22, 1.0],
            physicsClientId=self.client_id,
        )

        blast_positions = [
            (-11.5, -7.5),
            (-4.0, 8.5),
            (6.5, -4.5),
            (12.0, 6.0),
            (16.5, -9.0),
            (-15.5, 3.0),
        ]
        for x, y in blast_positions:
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=blast_visual,
                basePosition=[x, y, 0.03],
                physicsClientId=self.client_id,
            )

        for x, y, yaw in [
            (-8.5, -1.5, 0.35),
            (-2.0, 5.5, -0.2),
            (7.0, 1.2, 0.7),
            (13.5, -5.0, -0.55),
            (4.0, -10.0, 0.15),
        ]:
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=berm_visual,
                basePosition=[x, y, 0.42],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )

        ruin_specs = [
            (-13.0, 10.0, concrete_visual, 0.15),
            (10.5, 10.5, concrete_visual, -0.32),
            (2.5, -13.0, concrete_visual, 0.4),
            (-16.0, -11.0, wreck_visual, -0.6),
            (15.0, 1.0, wreck_visual, 0.22),
        ]
        self._ruin_ids = []
        for x, y, visual, yaw in ruin_specs:
            body = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, 0.8 if visual == concrete_visual else 0.38],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )
            self._ruin_ids.append(body)

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
        self._troop_anchor_bases = np.array(
            [
                [-12.0, -6.0],
                [-4.0, 8.0],
                [8.5, -2.0],
                [14.0, 7.5],
            ],
            dtype=np.float32,
        )
        self._troop_anchor_dirs = np.array([0.15, -0.5, 0.3, -0.25], dtype=np.float32)

        self.troop_ids = []
        for idx in range(self.config.num_troops):
            anchor = idx % len(self._troop_anchor_bases)
            radial = self.rng.uniform(1.2, 5.8)
            theta = self.rng.uniform(0.0, 2.0 * math.pi)
            offset = np.array(
                [math.cos(theta) * radial, math.sin(theta) * radial],
                dtype=np.float32,
            )
            self._troop_offsets[idx] = offset
            self._troop_anchor_ids[idx] = anchor
            self._troop_personal_phase[idx] = self.rng.uniform(0.0, 2.0 * math.pi)
            x = float(self._troop_anchor_bases[anchor, 0] + offset[0])
            y = float(self._troop_anchor_bases[anchor, 1] + offset[1])
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
        try:
            client_id = self._require_connection()
            if self.gui:
                self._handle_keyboard()
            self._advance_troops()
            self._advance_drones()
            if self.gui:
                self._update_observer_camera()
            p.stepSimulation(physicsClientId=client_id)
        except SimulationDisconnectedError:
            raise
        except Exception as exc:
            if (not self.is_connected()) or ("Not connected" in str(exc)):
                raise SimulationDisconnectedError(
                    "PyBullet client disconnected during step"
                ) from exc
            raise
        self.sim_time += self.config.time_step
        return SimulationSnapshot(
            sim_time=self.sim_time,
            drone_positions=self.drone_positions.copy(),
            troop_positions=self.troop_positions.copy(),
        )

    def _update_observer_camera(self) -> None:
        if self.camera_mode == "fpv":
            self._update_fpv_camera()
            return
        if self.camera_mode == "chase":
            self._update_chase_camera()
            return
        self._restore_fpv_drone_visibility()

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

    def _update_chase_camera(self) -> None:
        self._restore_fpv_drone_visibility()
        pos = self.drone_positions[self.selected_drone_id]
        yaw = float(self.drone_yaws[self.selected_drone_id])
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        camera_pos = pos - forward * 8.0 + np.array([0.0, 0.0, 3.4], dtype=np.float32)
        target = pos + forward * 5.5 + np.array([0.0, 0.0, -1.2], dtype=np.float32)
        self._set_debug_camera(camera_pos, target)

    def _update_fpv_camera(self) -> None:
        self._hide_selected_drone_for_fpv()
        camera = self.camera_pose(self.selected_drone_id)
        camera_pos = camera.eye - camera.forward * 0.15 + camera.up * 0.02
        target = camera.eye + camera.forward * 8.0
        self._set_debug_camera(camera_pos, target)

    def _set_debug_camera(self, camera_pos: np.ndarray, target: np.ndarray) -> None:
        delta = np.asarray(camera_pos, dtype=np.float32) - np.asarray(target, dtype=np.float32)
        distance = max(0.2, float(np.linalg.norm(delta)))
        horiz = max(1e-6, math.hypot(float(delta[0]), float(delta[1])))
        yaw = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
        pitch = -math.degrees(math.atan2(float(delta[2]), horiz))
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=[float(target[0]), float(target[1]), float(target[2])],
            physicsClientId=self.client_id,
        )

    def _hide_selected_drone_for_fpv(self) -> None:
        if self._fpv_hidden_drone_id == self.selected_drone_id:
            return
        self._restore_fpv_drone_visibility()
        body = self.drone_ids[self.selected_drone_id]
        p.changeVisualShape(
            body,
            -1,
            rgbaColor=[0.18, 0.18, 0.24, 0.0],
            physicsClientId=self.client_id,
        )
        self._fpv_hidden_drone_id = self.selected_drone_id

    def _restore_fpv_drone_visibility(self) -> None:
        if self._fpv_hidden_drone_id is None:
            return
        body = self.drone_ids[self._fpv_hidden_drone_id]
        p.changeVisualShape(
            body,
            -1,
            rgbaColor=[0.18, 0.18, 0.24, 1.0],
            physicsClientId=self.client_id,
        )
        self._fpv_hidden_drone_id = None

    def _handle_keyboard(self) -> None:
        client_id = self._require_connection()
        events = p.getKeyboardEvents(physicsClientId=client_id)
        for digit in range(1, min(self.config.num_drones, 9) + 1):
            if events.get(ord(str(digit)), 0) & p.KEY_WAS_TRIGGERED:
                self.selected_drone_id = digit - 1
        if events.get(ord("c"), 0) & p.KEY_WAS_TRIGGERED:
            modes = ("observer", "chase", "fpv")
            idx = (modes.index(self.camera_mode) + 1) % len(modes)
            self.camera_mode = modes[idx]
        if events.get(ord("b"), 0) & p.KEY_WAS_TRIGGERED:
            self.camera_mode = "observer"
        if events.get(ord("h"), 0) & p.KEY_WAS_TRIGGERED:
            self.camera_mode = "chase"
        if events.get(ord("f"), 0) & p.KEY_WAS_TRIGGERED:
            self.camera_mode = "fpv"
        if events.get(ord("m"), 0) & p.KEY_WAS_TRIGGERED:
            self.manual_drone_id = (
                None if self.manual_drone_id == self.selected_drone_id else self.selected_drone_id
            )
        if events.get(ord("r"), 0) & p.KEY_WAS_TRIGGERED:
            if self.manual_drone_id == self.selected_drone_id:
                self.manual_drone_id = None

    def _advance_troops(self) -> None:
        anchor_positions = self._troop_anchor_bases.copy()
        anchor_positions[0] += np.array(
            [self.sim_time * 0.45, 1.2 * math.sin(self.sim_time * 0.22)],
            dtype=np.float32,
        )
        anchor_positions[1] += np.array(
            [0.8 * math.sin(self.sim_time * 0.18), -0.7 * self.sim_time * 0.18],
            dtype=np.float32,
        )
        anchor_positions[2] += np.array(
            [0.95 * self.sim_time * 0.2, 1.0 * math.sin(self.sim_time * 0.31)],
            dtype=np.float32,
        )
        anchor_positions[3] += np.array(
            [-0.65 * math.sin(self.sim_time * 0.21), -0.55 * self.sim_time * 0.16],
            dtype=np.float32,
        )

        for idx, body in enumerate(self.troop_ids):
            anchor = self._troop_anchor_ids[idx]
            phase = float(self._troop_personal_phase[idx])
            spread = self._troop_offsets[idx]
            drift = np.array(
                [
                    0.55 * math.sin(self.sim_time * 0.75 + phase),
                    0.55 * math.cos(self.sim_time * 0.63 + phase * 0.7),
                ],
                dtype=np.float32,
            )
            x = float(anchor_positions[anchor, 0] + spread[0] + drift[0])
            y = float(anchor_positions[anchor, 1] + spread[1] + drift[1])
            z = 1.0
            heading_vec = anchor_positions[anchor] - np.array([x, y], dtype=np.float32)
            yaw = math.atan2(float(heading_vec[1]), float(heading_vec[0]))
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
        if self.manual_drone_id is not None and self.gui:
            velocity, yaw = self._manual_command(self.manual_drone_id)
            velocities[self.manual_drone_id] = velocity
            desired_yaws[self.manual_drone_id] = yaw
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

    def _manual_command(self, drone_idx: int) -> tuple[np.ndarray, float]:
        client_id = self._require_connection()
        events = p.getKeyboardEvents(physicsClientId=client_id)
        yaw = float(self.drone_yaws[drone_idx])
        speed = self.config.drone_speed_mps
        vel = np.zeros(3, dtype=np.float32)
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)

        if events.get(ord("i"), 0) & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            vel[0:2] += forward * speed
        if events.get(ord("k"), 0) & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            vel[0:2] -= forward * speed
        if events.get(ord("j"), 0) & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            vel[0:2] -= right * speed
        if events.get(ord("l"), 0) & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            vel[0:2] += right * speed
        if events.get(ord("u"), 0) & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            vel[2] += speed * 0.55
        if events.get(ord("o"), 0) & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            vel[2] -= speed * 0.55
        if events.get(ord("z"), 0) & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            yaw += 1.9 * self.config.time_step * math.pi
        if events.get(ord("x"), 0) & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            yaw -= 1.9 * self.config.time_step * math.pi

        norm = float(np.linalg.norm(vel))
        if norm > speed:
            vel *= speed / norm
        return vel, yaw

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
        client_id = self._require_connection()
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
        try:
            _, _, rgba, _, _ = p.getCameraImage(
                width=camera.width,
                height=camera.height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=client_id,
            )
        except Exception as exc:
            raise SimulationDisconnectedError(
                "PyBullet client disconnected during camera render"
            ) from exc
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
