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


class DetectionsMessage(BaseModel):
    """topic: 'detections' — emitted by perception (⓶) per-frame from the Jetson."""
    t: float
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
