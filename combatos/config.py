"""CombatOS Orchestrator configuration — all tunable constants in one place."""
from __future__ import annotations
import os

# ── Bus ──────────────────────────────────────────────────────────────────────
BUS_HOST = os.getenv("COMBATOS_HOST", "0.0.0.0")
BUS_PORT = int(os.getenv("COMBATOS_PORT", "8000"))

# ── Module health ─────────────────────────────────────────────────────────────
# A module that hasn't published in this many seconds is marked "degraded".
HEARTBEAT_TIMEOUT = float(os.getenv("COMBATOS_HEARTBEAT_TIMEOUT", "6.0"))

# ── Swarm sub-bus ─────────────────────────────────────────────────────────────
# The existing swarm/bus.py WebSocket server. Orchestrator connects as a client
# and relays every "swarm" message onto the main bus (port 8000).
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
