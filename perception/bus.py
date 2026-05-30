"""WebSocket bus client -- non-blocking async publish + incoming command receive.

Architecture
------------
The asyncio event loop runs permanently in a daemon background thread.
- publish()  fires a send coroutine and returns immediately (no blocking).
- Incoming messages arrive on the same WS connection; command messages
  (topic == "command") are put into self.commands (SimpleQueue) so the
  main CV loop can drain them each frame without blocking or locking.

Command message format (dashboard -> perception):
    {"topic": "command", "action": "confirm",       "track_id": 7}
    {"topic": "command", "action": "follow",         "track_id": 7}
    {"topic": "command", "action": "unconfirm"}
    {"topic": "command", "action": "release_follow"}
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from typing import Any
print("[import/bus] stdlib ok", flush=True)

import websockets
print("[import/bus] websockets ok", flush=True)

import config
print("[import/bus] config ok", flush=True)

from tracker import TrackedObject
print("[import/bus] tracker ok", flush=True)


def _serialize(objects: list[TrackedObject]) -> str:
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
    """Non-blocking WebSocket bus client with bidirectional command support."""

    def __init__(self) -> None:
        self._uri = f"ws://{config.WS_HOST}:{config.WS_PORT}"
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._ws: websockets.WebSocketClientProtocol | None = None
        # Incoming command queue -- drainable from the main thread each frame
        self.commands: queue.SimpleQueue[dict] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="bus-io")

    # ------------------------------------------------------------------
    # Lifecycle

    def connect(self) -> None:
        """Start the background I/O thread and open the WS connection.
        Raises on connection failure (caller should catch and fall back to local-only).
        """
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        future.result(timeout=5)

    def close(self) -> None:
        if self._ws:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result(timeout=2)
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        self._ws = await websockets.connect(self._uri)
        self._loop.create_task(self._receive_loop())

    # ------------------------------------------------------------------
    # Publish (non-blocking -- fire and forget)

    def publish(self, objects: list[TrackedObject]) -> None:
        if self._ws is None or self._ws.closed:
            return
        msg = _serialize(objects)
        asyncio.run_coroutine_threadsafe(self._ws.send(msg), self._loop)
        # Do NOT await the future -- returns immediately to the main thread

    # ------------------------------------------------------------------
    # Receive loop (runs inside the background asyncio loop)

    async def _receive_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    if msg.get("topic") == "command":
                        self.commands.put_nowait(msg)
                except (json.JSONDecodeError, KeyError):
                    pass
        except websockets.ConnectionClosed:
            pass
