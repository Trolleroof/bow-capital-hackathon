"""Perception module proxy.

The perception vertical (⓶, Matthieu) runs YOLO + face detector on the Jetson
and publishes 'detections' messages by connecting TO this orchestrator's WS bus
as a client.

This module's job:
  1. When perception is live — nothing to do; ws_server.py already relays
     frames and beats system_state("perception").
  2. When perception is offline — emit an empty detections frame so the
     dashboard panel renders a live (empty) feed instead of going stale.

See combatos/HOOK_PERCEPTION.md for exactly what Matthieu needs to implement.
"""
from __future__ import annotations
import asyncio
import logging
import time

from .base import AbstractModule
from ..bus import router
from .. import config
from ..state import fallback

log = logging.getLogger(__name__)


class PerceptionModule(AbstractModule):
    name = "perception"

    async def run(self) -> None:
        log.info(
            "[perception] proxy active — Jetson publishes detections to "
            "ws://ORCHESTRATOR_IP:%d",
            config.BUS_PORT,
        )
        dt = 1.0 / config.MOCK_POSE_HZ  # same low rate as nav mock
        while True:
            await asyncio.sleep(dt)
            if config.EMIT_MOCK_DETECTIONS and not fallback.perception_is_live():
                await router.publish("detections", {
                    "t": round(time.monotonic(), 3),
                    "objects": [],
                })
