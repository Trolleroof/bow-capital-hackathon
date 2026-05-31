"""Outcast Virus image WebSocket server for frame transport."""
from __future__ import annotations

import asyncio
import json
import logging

import websockets
from websockets.server import WebSocketServerProtocol

from .. import config
from . import image_router

log = logging.getLogger(__name__)


def _topics_from_subscription(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return []
    return [topic for topic in value if isinstance(topic, str)]


async def _handler(ws: WebSocketServerProtocol) -> None:
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=4)

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
                topics = _topics_from_subscription(data.get("topics"))
                image_router.subscribe(q, topics)
                image_router.replay_latest(q, topics)
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
    async with websockets.serve(
        _handler,
        config.IMAGE_BUS_HOST,
        config.IMAGE_BUS_PORT,
        max_size=config.WS_MAX_SIZE,
        ping_interval=config.WS_PING_INTERVAL,
        ping_timeout=config.WS_PING_TIMEOUT,
    ):
        log.info(
            "Outcast Virus image bus ws://%s:%d (frame transport only)",
            config.IMAGE_BUS_HOST,
            config.IMAGE_BUS_PORT,
        )
        await asyncio.Future()
