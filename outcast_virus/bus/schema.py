"""Pydantic models for the §5 message contract.

These are the authoritative definitions for every topic on the bus.
Any module that publishes to the orchestrator should match these shapes.
Validation is applied at the WS server boundary — malformed messages are
logged and dropped rather than crashing the bus.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class PoseMessage(BaseModel):
    """topic: 'pose'  — emitted by nav (⓵) at ~30 Hz from the Jetson."""
    t: float
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    gps: bool = False          # always False — GPS denied
    tracking: str = "OK"       # "OK" | "NO_LOCK" | "LOST"


class SlamStatusMessage(BaseModel):
    """topic: 'slam_status' — emitted by the ROS2 SLAM bridge."""
    t: float
    tracking: str = "NO_LOCK"
    connected: bool = True
    camera_frames: int = 0
    annotated_frames: int = 0
    dropped_frames: int = 0


class SlamPoint(BaseModel):
    x: float
    y: float
    z: float


class SlamPose(SlamPoint):
    t: float = 0.0


class SlamOdometryMessage(SlamPose):
    """topic: 'slam_odometry' — normalized nav_msgs/Odometry from ORB-SLAM3."""
    frame_id: str = "map"
    child_frame_id: str = "camera"
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    tracking: str = "NO_LOCK"


class SlamPathMessage(BaseModel):
    """topic: 'slam_path' — compact nav_msgs/Path for browser rendering."""
    t: float
    frame_id: str = "map"
    poses: list[SlamPose] = []


class SlamPointCloudMessage(BaseModel):
    """topic: 'slam_point_cloud' — downsampled /slam/point_cloud map points."""
    t: float
    frame_id: str = "map"
    points: list[SlamPoint] = []
    total_points: int = 0


class SlamFrameMessage(BaseModel):
    """topic: 'camera_frame' | 'slam_frame' — base64 JPEG test stream."""
    t: float
    frame_id: str = ""
    source: str = ""
    encoding: Literal["jpeg"] = "jpeg"
    width: int
    height: int
    seq: int
    data: str


class Detection(BaseModel):
    id: int
    cls: str
    conf: float
    bbox: list[float]          # [x, y, w, h] — pixel coords, normalised 0-1
    is_target: bool = False
    confirmed: bool = False    # True after operator clicks CONFIRM
    allegiance: str | None = None  # "friend" | "foe" | None (IFF disabled or unknown)


class DetectionsMessage(BaseModel):
    """topic: 'detections' — emitted by perception (⓶) per-frame from the Jetson."""
    t: float
    source: str = ""
    drone_id: int | None = None
    objects: list[Detection] = []


class ReconMessage(BaseModel):
    """topic: 'recon' — emitted by the orchestrator recon poller when asset lands."""
    status: Literal["training", "ready", "error"]
    splat_url: str = ""
    frames_used: int = 0


class AgentState(BaseModel):
    id: int
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    role: str = "scout"
    alive: bool = True


class SwarmMessage(BaseModel):
    """topic: 'swarm' — emitted by swarm (⓸) at ~10 Hz."""
    t: float
    comms: Literal["denied"] = "denied"
    agents: list[AgentState] = []


class DroneCameraTarget(BaseModel):
    id: int
    cls: str = "troop"
    x: float
    y: float
    z: float
    width_m: float = 0.55
    height_m: float = 1.75


class DroneFpvStateMessage(BaseModel):
    """topic: 'drone_fpv_state' — camera pose + target world state for FPV round-trip."""
    t: float
    seq: int
    frame_id: str
    drone_id: int
    source: str
    width: int
    height: int
    fov_deg: float
    eye: list[float]
    forward: list[float]
    up: list[float]
    targets: list[DroneCameraTarget] = []


class ModuleHealth(BaseModel):
    nav: str = "down"          # "up" | "degraded" | "down"
    perception: str = "down"
    recon: str = "down"
    swarm: str = "down"


class StatusMessage(BaseModel):
    """topic: 'status' — emitted by orchestrator at 1 Hz; drives the hero banner."""
    gps: Literal["DENIED", "ACTIVE"] = "DENIED"   # always DENIED — that's the product
    link: Literal["NONE", "UP"] = "NONE"
    localized: bool = False
    modules: ModuleHealth = ModuleHealth()
