"""CombatOS image WebSocket server for frame transport."""
from __future__ import annotations

import asyncio
import json
import logging

import websockets
from websockets.server import WebSocketServerProtocol

from .. import config
from . import image_router

log = logging.getLogger(__name__)


async def _handler(ws: WebSocketServerProtocol) -> None:
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
    image_router.subscribe(q, topics=None)

    async def _sender() -> None:
        while True:
            msg = await q.get()
            try:
                await ws.send(msg)
            except websockets.exceptions.ConnectionClosed:
                break

    sender_task = asyncio.create_task(_sender())
    log.info("image client connected %s", ws.remote_address)

    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("non-JSON image frame from %s ignored", ws.remote_address)
                continue

            if data.get("type") == "subscribe":
                image_router.unsubscribe(q)
                image_router.subscribe(q, data.get("topics"))
                continue

            topic = data.get("topic")
            if not topic:
                continue

            payload = {k: v for k, v in data.items() if k != "topic"}
            await image_router.publish(topic, payload, exclude=q)
    except websockets.exceptions.ConnectionClosedError:
        pass
    finally:
        image_router.unsubscribe(q)
        sender_task.cancel()
        log.info("image client disconnected %s", ws.remote_address)


async def serve() -> None:
    async with websockets.serve(_handler, config.IMAGE_BUS_HOST, config.IMAGE_BUS_PORT):
        log.info(
            "CombatOS image bus ws://%s:%d (frame transport only)",
            config.IMAGE_BUS_HOST,
            config.IMAGE_BUS_PORT,
        )
        await asyncio.Future()
