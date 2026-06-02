from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OrchestratorConfig:
    control_ws_url: str = "ws://localhost:8000"
    image_ws_url: str = "ws://localhost:8001"
    raw_topic: str = "drone_fpv_raw"
    hud_topic: str = "drone_fpv_hud"
    state_topic: str = "drone_fpv_state"
    detections_topic: str = "drone_detections"
    dashboard_raw_topic: str = "fpv_raw"
    dashboard_hud_topic: str = "fpv_hud"
    dashboard_drone_id: int = 0


@dataclass(frozen=True)
class DroneCameraConfig:
    width: int = 640
    height: int = 360
    fov_deg: float = 78.0
    near: float = 0.05
    far: float = 120.0
    tilt_deg: float = 58.0
    forward_offset_m: float = 0.18


@dataclass(frozen=True)
class RecordingConfig:
    output_path: Path = Path("output/drone_spy_demo.mp4")
    fps: int = 12


@dataclass(frozen=True)
class SimulationConfig:
    num_drones: int = 5
    num_troops: int = 6
    duration_sec: float = 12.0
    time_step: float = 1.0 / 30.0
    world_half_extent_m: float = 30.0
    drone_altitude_m: float = 12.0
    drone_speed_mps: float = 5.0
    drone_ring_radius_m: float = 9.0
    drone_separation_gain: float = 2.4
    troop_spacing_m: float = 1.6
    troop_stride_mps: float = 1.0
    resources_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "resources"
    )
    camera: DroneCameraConfig = field(default_factory=DroneCameraConfig)
