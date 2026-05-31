"""CombatOS WebSocket bus server.

Single endpoint for the React dashboard and all remote modules (nav, perception
on the Jetson).  Protocol:

  SUBSCRIBE  →  client sends  {"type": "subscribe", "topics": ["pose", "status"]}
               If omitted, client receives every topic.

  PUBLISH    →  client sends  {"topic": "pose", "t": 1.0, "x": 0, ...}
               Relayed to all other subscribers of that topic.
               Also updates the health heartbeat for the corresponding module.

  RECEIVE    ←  server sends  {"topic": "...", ...}  (same shape as publish)

Run:
    python -m combatos.orchestrator   (preferred — starts everything)
    python -m combatos.bus.ws_server  (bus only, for testing)
"""
from __future__ import annotations
import asyncio
import json
import logging

import websockets
from websockets.server import WebSocketServerProtocol

from .. import config
from . import image_router, router
from ..state import system_state

log = logging.getLogger(__name__)

_IMAGE_TOPICS = {"camera_frame", "camera_right_frame", "slam_frame", "fpv_raw", "fpv_hud"}

# Maps the incoming topic to its owning module so we can update health state
# whenever a remote client (Jetson, etc.) publishes on that topic.
_TOPIC_MODULE: dict[str, str] = {
    "pose": "nav",
    "slam_status": "nav",
    "slam_odometry": "nav",
    "slam_path": "nav",
    "slam_point_cloud": "nav",
    "slam_diagnostics": "nav",
    "detections": "perception",
    "drone_detections": "perception",
    "drone_fpv_state": "swarm",
    "swarm": "swarm",
    "recon": "recon",
}


async def _handler(ws: WebSocketServerProtocol) -> None:
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
    # Default: subscribe to all topics so dashboard gets everything without
    # sending an explicit subscribe message first.
    router.subscribe(q, topics=None)

    async def _sender() -> None:
        while True:
            msg = await q.get()
            try:
                await ws.send(msg)
            except websockets.exceptions.ConnectionClosed:
                break

    sender_task = asyncio.create_task(_sender())
    log.info("client connected  %s", ws.remote_address)

    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("non-JSON frame from %s — ignored", ws.remote_address)
                continue

            # ── Subscription control ──────────────────────────────────────────
            if data.get("type") == "subscribe":
                router.unsubscribe(q)
                topics = data.get("topics")  # None = all
                router.subscribe(q, topics)
                log.debug("client %s → subscribed to %s", ws.remote_address, topics)
                continue

            # ── Publish ───────────────────────────────────────────────────────
            topic = data.get("topic")
            if not topic:
                continue

            payload = {k: v for k, v in data.items() if k != "topic"}

            # Beat the health clock for whichever module owns this topic.
            module = _TOPIC_MODULE.get(topic)
            if module:
                system_state.beat(module)

            # Relay to all OTHER subscribers (exclude=q prevents echo).
            await router.publish(topic, payload, exclude=q)
            if topic in _IMAGE_TOPICS:
                await image_router.publish(topic, payload)

    except websockets.exceptions.ConnectionClosedError:
        pass
    finally:
        router.unsubscribe(q)
        sender_task.cancel()
        log.info("client disconnected %s", ws.remote_address)


async def serve() -> None:
    """Start the WebSocket server and run until cancelled."""
    async with websockets.serve(
        _handler,
        config.BUS_HOST,
        config.BUS_PORT,
        max_size=config.WS_MAX_SIZE,
        ping_interval=config.WS_PING_INTERVAL,
        ping_timeout=config.WS_PING_TIMEOUT,
    ):
        log.info(
            "CombatOS bus  ws://%s:%d  (dashboard + remote modules connect here)",
            config.BUS_HOST, config.BUS_PORT,
        )
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
