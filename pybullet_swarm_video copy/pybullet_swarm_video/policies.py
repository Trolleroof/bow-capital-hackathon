from __future__ import annotations

import math

import numpy as np


class ScriptedSurveillancePolicy:
    """Simple controller that keeps drones orbiting and scanning over troops."""

    def __init__(
        self,
        num_drones: int,
        ring_radius_m: float,
        cruise_altitude_m: float,
        max_speed_mps: float,
        separation_gain: float,
    ) -> None:
        self.num_drones = num_drones
        self.ring_radius_m = ring_radius_m
        self.cruise_altitude_m = cruise_altitude_m
        self.max_speed_mps = max_speed_mps
        self.separation_gain = separation_gain
        self.phase_offsets = np.linspace(0.0, 2.0 * math.pi, num_drones, endpoint=False)

    def commands(
        self,
        sim_time: float,
        drone_positions: np.ndarray,
        troop_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        troop_centroid = troop_positions.mean(axis=0)
        desired_positions = np.zeros_like(drone_positions)
        desired_yaws = np.zeros(self.num_drones, dtype=np.float32)

        for idx in range(self.num_drones):
            phase = self.phase_offsets[idx]
            sweep_theta = sim_time * 0.22 + phase
            radial_scale = 1.0 + 0.14 * math.sin(sim_time * 0.41 + phase)
            orbit_xy = np.array(
                [
                    math.cos(sweep_theta),
                    math.sin(sweep_theta),
                ],
                dtype=np.float32,
            ) * (self.ring_radius_m * radial_scale)
            lane_offset = np.array(
                [
                    0.0,
                    1.2 * math.sin(sim_time * 0.7 + phase * 1.5),
                ],
                dtype=np.float32,
            )
            desired_positions[idx, 0:2] = troop_centroid[0:2] + orbit_xy + lane_offset
            desired_positions[idx, 2] = self.cruise_altitude_m + 0.7 * math.sin(
                sim_time * 0.85 + phase
            )

            look_vec = troop_centroid[0:2] - drone_positions[idx, 0:2]
            desired_yaws[idx] = math.atan2(float(look_vec[1]), float(look_vec[0]))

        velocities = desired_positions - drone_positions
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        speeds = np.clip(norms, 1e-6, self.max_speed_mps)
        velocities = velocities / speeds * np.minimum(norms, self.max_speed_mps)

        for idx in range(self.num_drones):
            repel = np.zeros(3, dtype=np.float32)
            for other in range(self.num_drones):
                if idx == other:
                    continue
                delta = drone_positions[idx] - drone_positions[other]
                dist_sq = float(np.dot(delta, delta))
                if dist_sq < 1e-4:
                    continue
                if dist_sq < 16.0:
                    repel += delta / dist_sq
            velocities[idx] += repel * self.separation_gain

        speed_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        clip = np.maximum(speed_norms / self.max_speed_mps, 1.0)
        velocities = velocities / clip
        return velocities.astype(np.float32), desired_yaws


def policy_note() -> str:
    return (
        "Scripted surveillance controller active. Replace this layer with a policy "
        "adapter if you want to reuse observations or actions from swarm/SWARM.md."
    )
