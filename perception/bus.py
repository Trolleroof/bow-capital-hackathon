"""WebSocket publisher — sends detections JSON to the CombatOS bus."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import websockets

import config
from tracker import TrackedObject


def serialize(objects: list[TrackedObject]) -> str:
    payload: dict[str, Any] = {
        "t": time.time(),
        "objects": [
            {
                "id":        o.id,
                "cls":       o.cls,
                "conf":      round(o.conf, 3),
                "bbox":      o.bbox,
                "has_face":  o.has_face,
                "is_target": o.is_primary or o.confirmed,
                "confirmed": o.confirmed,
            }
            for o in objects
        ],
    }
    return json.dumps({"topic": config.WS_TOPIC, "data": payload})


class BusPublisher:
    def __init__(self) -> None:
        self._uri = f"ws://{config.WS_HOST}:{config.WS_PORT}"
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._loop = asyncio.new_event_loop()

    def connect(self) -> None:
        self._loop.run_until_complete(self._connect())

    async def _connect(self) -> None:
        self._ws = await websockets.connect(self._uri)

    def publish(self, objects: list[TrackedObject]) -> None:
        if self._ws is None:
            return
        msg = serialize(objects)
        self._loop.run_until_complete(self._ws.send(msg))

    def close(self) -> None:
        if self._ws:
            self._loop.run_until_complete(self._ws.close())
