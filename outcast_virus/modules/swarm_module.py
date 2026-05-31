"""Swarm module — bridges the existing swarm/bus.py server to the main Outcast Virus bus.

The swarm vertical (⓸) already runs its own WebSocket server on SWARM_BUS_URL
(default ws://localhost:8765).  This module connects to it as a client and
relays every frame onto the main orchestrator bus (port 8000) so the React
dashboard only needs one connection.

Architecture:
    swarm/bus.py  →  ws://localhost:8765  →  SwarmModule (relay client)
                                               │
                                               ▼
                                    orchestrator bus :8000
                                               │
                                               ▼
                                      React dashboard

Retry logic: if the swarm bus is not yet running (Phase 0), the module waits
and retries every RETRY_DELAY seconds.  It won't crash the orchestrator.

See outcast_virus/HOOK_SWARM.md for the Phase 3 upgrade (direct publishing to :8000).
"""
from __future__ import annotations
import asyncio
import json
import logging

import websockets

from .base import AbstractModule
from ..bus import router
from .. import config
from ..state import system_state

log = logging.getLogger(__name__)

RETRY_DELAY = 3.0


class SwarmModule(AbstractModule):
    name = "swarm"

    async def run(self) -> None:
        while True:
            try:
                log.info("[swarm] connecting to %s", config.SWARM_BUS_URL)
                async with websockets.connect(
                    config.SWARM_BUS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    log.info("[swarm] relay active")
                    system_state.beat("swarm")

                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        topic = data.get("topic", "swarm")
                        payload = {k: v for k, v in data.items() if k != "topic"}
                        await router.publish(topic, payload)
                        system_state.beat("swarm")

            except (OSError, websockets.exceptions.WebSocketException) as exc:
                log.warning(
                    "[swarm] connection to %s lost (%s) — retrying in %.0f s",
                    config.SWARM_BUS_URL, exc, RETRY_DELAY,
                )
                system_state.set_status("swarm", "down")
                await asyncio.sleep(RETRY_DELAY)
