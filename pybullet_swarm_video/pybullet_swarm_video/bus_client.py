from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets

from .config import OrchestratorConfig
from .messages import encode_jpeg_from_rgb


class OrchestratorBusClient:
    def __init__(self, config: OrchestratorConfig, jpeg_quality: int = 80) -> None:
        self.config = config
        self.jpeg_quality = jpeg_quality
        self.control_ws: websockets.WebSocketClientProtocol | None = None
        self.image_ws: websockets.WebSocketClientProtocol | None = None
        self._control_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._image_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    async def __aenter__(self) -> OrchestratorBusClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(
        self,
        control_topics: list[str] | None = None,
        image_topics: list[str] | None = None,
    ) -> None:
        await self.close()
        self.control_ws = await websockets.connect(self.config.control_ws_url)
        self.image_ws = await websockets.connect(self.config.image_ws_url)

        if control_topics is not None:
            await self.control_ws.send(
                json.dumps({"type": "subscribe", "topics": control_topics})
            )
        if image_topics is not None:
            await self.image_ws.send(
                json.dumps({"type": "subscribe", "topics": image_topics})
            )

        self._tasks = [
            asyncio.create_task(self._reader(self.control_ws, self._control_queue)),
            asyncio.create_task(self._reader(self.image_ws, self._image_queue)),
        ]

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        if self.control_ws is not None:
            await self.control_ws.close()
            self.control_ws = None
        if self.image_ws is not None:
            await self.image_ws.close()
            self.image_ws = None

    async def _reader(
        self,
        ws: websockets.WebSocketClientProtocol,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        try:
            async for raw in ws:
                try:
                    queue.put_nowait(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        except websockets.ConnectionClosed:
            return

    async def publish_control(self, topic: str, payload: dict[str, Any]) -> None:
        if self.control_ws is None:
            raise RuntimeError("control websocket is not connected")
        await self.control_ws.send(json.dumps({"topic": topic, **payload}))

    async def publish_rgb_frame(self, topic: str, payload: dict[str, Any], frame) -> None:
        if self.image_ws is None:
            raise RuntimeError("image websocket is not connected")
        encoded = encode_jpeg_from_rgb(frame, quality=self.jpeg_quality)
        await self.publish_encoded_frame(topic, payload, encoded)

    async def publish_encoded_frame(
        self,
        topic: str,
        payload: dict[str, Any],
        encoded_data: str,
    ) -> None:
        if self.image_ws is None:
            raise RuntimeError("image websocket is not connected")
        await self.image_ws.send(json.dumps({"topic": topic, **payload, "data": encoded_data}))

    async def next_control(self, timeout: float | None = None) -> dict[str, Any]:
        if timeout is None:
            return await self._control_queue.get()
        return await asyncio.wait_for(self._control_queue.get(), timeout=timeout)

    async def next_image(self, timeout: float | None = None) -> dict[str, Any]:
        if timeout is None:
            return await self._image_queue.get()
        return await asyncio.wait_for(self._image_queue.get(), timeout=timeout)
