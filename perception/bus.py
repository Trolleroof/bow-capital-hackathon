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
import base64
import json
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np
print("[import/bus] stdlib ok", flush=True)

import websockets
print("[import/bus] websockets ok", flush=True)

import config
print("[import/bus] config ok", flush=True)

from tracker import TrackedObject
print("[import/bus] tracker ok", flush=True)


def _normalize_bbox(bbox: list[int], fw: int, fh: int) -> list[float]:
    x, y, w, h = bbox
    return [round(x/fw, 4), round(y/fh, 4), round(w/fw, 4), round(h/fh, 4)]


def _serialize(objects: list[TrackedObject], fw: int, fh: int, candidate_id: int | None = None) -> str:
    # Flat layout: orchestrator strips "topic" and re-publishes {"topic": ..., **rest},
    # so objects must be top-level (not nested under "data") for the frontend to read msg.objects.
    return json.dumps({
        "topic": config.WS_TOPIC,
        "t": time.time(),
        "objects": [
            {
                "id":           o.id,
                "cls":          o.cls,
                "conf":         round(o.conf, 3),
                "bbox":         _normalize_bbox(o.bbox, fw, fh),
                "has_face":     o.has_face,
                "is_primary":   o.is_primary and not o.confirmed,
                "is_candidate": (candidate_id is not None and o.id == candidate_id
                                 and not o.is_primary and not o.confirmed),
                "confirmed":    o.confirmed,
                "allegiance":   o.allegiance,
            }
            for o in objects
        ],
    })


class BusPublisher:
    """Non-blocking WebSocket bus client with bidirectional command support."""

    def __init__(self) -> None:
        self._uri = f"ws://{config.WS_HOST}:{config.WS_PORT}"
        self._image_uri = f"ws://{config.WS_HOST}:{config.IMAGE_WS_PORT}"
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._image_ws: websockets.WebSocketClientProtocol | None = None
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
        if self._image_ws:
            asyncio.run_coroutine_threadsafe(self._image_ws.close(), self._loop).result(timeout=2)
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        self._ws = await websockets.connect(self._uri)
        try:
            self._image_ws = await websockets.connect(self._image_uri)
        except OSError as exc:
            print(f"[bus] image bus unavailable ({exc}); detections still publish on {self._uri}", flush=True)
            self._image_ws = None
        self._loop.create_task(self._receive_loop())

    # ------------------------------------------------------------------
    # Publish (non-blocking -- fire and forget)

    def _send(self, msg: str) -> None:
        if self._ws is None:
            return
        asyncio.run_coroutine_threadsafe(self._ws.send(msg), self._loop)

    def _send_image(self, msg: str) -> None:
        if self._image_ws is None:
            return
        asyncio.run_coroutine_threadsafe(self._image_ws.send(msg), self._loop)

    def publish(self, objects: list[TrackedObject], frame_w: int, frame_h: int,
               candidate_id: int | None = None) -> None:
        self._send(_serialize(objects, frame_w, frame_h, candidate_id))

    def publish_frame(self, frame: np.ndarray, topic: str, quality: int = 60) -> None:
        """Encode frame as JPEG and broadcast on the given topic."""
        if self._image_ws is None:
            return
        if config.GRAYSCALE_STREAM:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        b64 = base64.b64encode(buf).decode("ascii")
        self._send_image(json.dumps({"topic": topic, "data": b64, "grayscale": config.GRAYSCALE_STREAM}))

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
