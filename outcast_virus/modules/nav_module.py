"""Navigation module proxy.

The nav vertical (⓵, Vikram) runs ORB-SLAM3 on the Jetson and publishes 'pose'
messages by connecting TO this orchestrator's WS bus as a client.

This module's job:
  1. When nav is live — nothing to do; ws_server.py already relays Jetson frames
     to all dashboard subscribers and calls system_state.beat("nav").
  2. When nav is offline — emit a mock pose at MOCK_POSE_HZ so the dashboard
     panel renders LOCALIZED: false instead of going blank.

See outcast_virus/HOOK_NAV.md for exactly what Vikram needs to implement.
"""
from __future__ import annotations
import asyncio
import logging
import math
import time

from .base import AbstractModule
from ..bus import router
from .. import config
from ..state import fallback

log = logging.getLogger(__name__)

# Stub trajectory: slow circle at z=1.5 m, radius 2 m.
# Gives the dashboard something to render while Jetson is offline.
_CIRCLE_R = 2.0
_CIRCLE_SPEED = 0.1   # rad/s


def _mock_pose(t: float) -> dict:
    angle = t * _CIRCLE_SPEED
    return {
        "t": round(t, 3),
        "x": round(_CIRCLE_R * math.cos(angle), 3),
        "y": round(_CIRCLE_R * math.sin(angle), 3),
        "z": 1.5,
        "qx": 0.0, "qy": 0.0,
        "qz": round(math.sin(angle / 2), 4),
        "qw": round(math.cos(angle / 2), 4),
        "gps": False,
        "tracking": "NO_LOCK",
    }


class NavModule(AbstractModule):
    name = "nav"

    async def run(self) -> None:
        log.info("[nav] proxy active — Jetson publishes pose to ws://ORCHESTRATOR_IP:%d", config.BUS_PORT)
        dt = 1.0 / config.MOCK_POSE_HZ
        while True:
            await asyncio.sleep(dt)
            if config.EMIT_MOCK_POSE and not fallback.nav_is_live():
                await router.publish("pose", _mock_pose(time.monotonic()))
