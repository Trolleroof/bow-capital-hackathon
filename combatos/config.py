"""CombatOS Orchestrator configuration — all tunable constants in one place."""
from __future__ import annotations
import os

# ── Bus ──────────────────────────────────────────────────────────────────────
BUS_HOST = os.getenv("COMBATOS_HOST", "0.0.0.0")
BUS_PORT = int(os.getenv("COMBATOS_PORT", "8000"))
IMAGE_BUS_HOST = os.getenv("COMBATOS_IMAGE_HOST", BUS_HOST)
IMAGE_BUS_PORT = int(os.getenv("COMBATOS_IMAGE_PORT", "8001"))

# ── Module health ─────────────────────────────────────────────────────────────
# A module that hasn't published in this many seconds is marked "degraded".
HEARTBEAT_TIMEOUT = float(os.getenv("COMBATOS_HEARTBEAT_TIMEOUT", "6.0"))

# ── Swarm sub-bus ─────────────────────────────────────────────────────────────
# The existing swarm/bus.py WebSocket server. Orchestrator connects as a client
# and relays every "swarm" message onto the main bus (port 8000).
ENABLE_SWARM = os.getenv("COMBATOS_SWARM", "0") == "1"
SWARM_BUS_URL = os.getenv("SWARM_BUS_URL", "ws://localhost:8765")

# ── Recon ─────────────────────────────────────────────────────────────────────
# Path (relative to repo root) where the finished splat file will appear.
# The orchestrator polls for it and flips "recon" from training → ready.
RECON_ASSET_PATH = os.getenv("RECON_ASSET_PATH", "recon/assets/field.splat")
# URL path served to the browser (must match whatever static-file route you set up).
RECON_ASSET_URL = os.getenv("RECON_ASSET_URL", "/assets/field.splat")
RECON_FRAMES_SIDECAR = os.getenv("RECON_FRAMES_SIDECAR", "recon/assets/field.json")
RECON_POLL_INTERVAL = float(os.getenv("RECON_POLL_INTERVAL", "5.0"))

# ── Status broadcast ──────────────────────────────────────────────────────────
STATUS_HZ = float(os.getenv("COMBATOS_STATUS_HZ", "1.0"))

# ── Mock data ─────────────────────────────────────────────────────────────────
# When nav/perception are offline the orchestrator emits these stubs so the
# dashboard panels render something instead of going blank.
EMIT_MOCK_POSE = os.getenv("COMBATOS_MOCK_POSE", "1") == "1"
EMIT_MOCK_DETECTIONS = os.getenv("COMBATOS_MOCK_DETECTIONS", "1") == "1"
MOCK_POSE_HZ = float(os.getenv("COMBATOS_MOCK_POSE_HZ", "1.0"))

# ── Desktop ROS2 SLAM bridge ────────────────────────────────────────────────
# Runs inside the orchestrator process on the desktop. It subscribes to ROS2
# topics published by the Jetson over DDS and performs image serialization on
# the desktop instead of on the Jetson.
ENABLE_ROS_SLAM = os.getenv("COMBATOS_ROS_SLAM", "1") == "1"
ROS_SLAM_POSE_TOPIC = os.getenv("COMBATOS_ROS_SLAM_POSE_TOPIC", "/slam/pose")
ROS_SLAM_ODOM_TOPIC = os.getenv("COMBATOS_ROS_SLAM_ODOM_TOPIC", "/slam/odometry")
ROS_SLAM_PATH_TOPIC = os.getenv("COMBATOS_ROS_SLAM_PATH_TOPIC", "/slam/path")
ROS_SLAM_STATUS_TOPIC = os.getenv("COMBATOS_ROS_SLAM_STATUS_TOPIC", "/slam/status")
ROS_SLAM_CAMERA_TOPIC = os.getenv("COMBATOS_ROS_SLAM_CAMERA_TOPIC", "/oak/left/image_raw/compressed")
ROS_SLAM_ANNOTATED_TOPIC = os.getenv("COMBATOS_ROS_SLAM_ANNOTATED_TOPIC", "/slam/tracked_image/compressed")
ROS_SLAM_IMAGE_TRANSPORT = os.getenv("COMBATOS_ROS_SLAM_IMAGE_TRANSPORT", "compressed")
ROS_SLAM_ENABLE_CAMERA = os.getenv("COMBATOS_ROS_SLAM_ENABLE_CAMERA", "1") == "1"
ROS_SLAM_ENABLE_ANNOTATED = os.getenv("COMBATOS_ROS_SLAM_ENABLE_ANNOTATED", "1") == "1"
ROS_SLAM_VIDEO_FPS = float(os.getenv("COMBATOS_ROS_SLAM_VIDEO_FPS", "10.0"))
ROS_SLAM_JPEG_QUALITY = int(os.getenv("COMBATOS_ROS_SLAM_JPEG_QUALITY", "70"))
ROS_SLAM_PATH_MAX_POSES = int(os.getenv("COMBATOS_ROS_SLAM_PATH_MAX_POSES", "240"))
