"""Desktop ROS2 perception bridge.

The Jetson runs combatos_perception/yolox_node.py against the OAK stream and
publishes ROS2 detections plus an annotated image. This orchestrator module runs
on the main computer, subscribes over DDS, and relays compact WebSocket topics
to the frontend.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from .base import AbstractModule
from .. import config
from ..bus import router
from ..state import system_state

log = logging.getLogger(__name__)


def _stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass
class _Stats:
    detections: int = 0
    frames: int = 0
    dropped: int = 0
    last_frame_sent: float = 0.0


class RosPerceptionModule(AbstractModule):
    name = "ros_perception"

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stats = _Stats()
        self._video_period = 1.0 / max(0.1, config.ROS_PERCEPTION_VIDEO_FPS)
        self._jpeg_quality = max(20, min(95, config.ROS_PERCEPTION_JPEG_QUALITY))

    async def run(self) -> None:
        if not config.ENABLE_ROS_PERCEPTION:
            log.info("[ros_perception] disabled; set COMBATOS_ROS_PERCEPTION=1 to subscribe to Jetson YOLO topics")
            while True:
                await asyncio.sleep(3600)

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=256)
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin_ros, name="combatos_ros_perception", daemon=True)
        self._thread.start()
        log.info("[ros_perception] desktop ROS2 bridge enabled")

        try:
            while True:
                topic, payload = await self._queue.get()
                await router.publish(topic, payload)
                system_state.beat("perception")
        finally:
            self._stop.set()

    def _enqueue(self, topic: str, payload: dict[str, Any]) -> None:
        loop = self._loop
        queue = self._queue
        if loop is None or queue is None:
            return

        def put() -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                    self._stats.dropped += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait((topic, payload))
            except asyncio.QueueFull:
                self._stats.dropped += 1

        loop.call_soon_threadsafe(put)

    def _spin_ros(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
            from std_msgs.msg import String
        except Exception as exc:
            log.error("[ros_perception] ROS2 dependencies unavailable: %s", exc)
            log.error("[ros_perception] source ROS2 and install cv_bridge/python3-opencv before enabling")
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        node = Node("combatos_orchestrator_perception_bridge")
        reliable_qos = QoSProfile(depth=10)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        def on_detections(msg: String) -> None:
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError as exc:
                log.warning("[ros_perception] malformed detection JSON: %s", exc)
                return
            if not isinstance(payload, dict):
                return
            self._stats.detections += 1
            payload.setdefault("source", config.ROS_PERCEPTION_DETECTIONS_TOPIC)
            payload.setdefault("t", time.time())
            self._enqueue("detections", self._normalize_detection_payload(payload))

        def on_image(msg: Image) -> None:
            now = time.monotonic()
            if now - self._stats.last_frame_sent < self._video_period:
                self._stats.dropped += 1
                return
            self._stats.last_frame_sent = now
            try:
                import cv2
                from cv_bridge import CvBridge, CvBridgeError

                bridge = CvBridge()
                image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            except (CvBridgeError, ValueError) as exc:
                log.warning("[ros_perception] image encode failed: %s", exc)
                return
            except Exception as exc:
                log.warning("[ros_perception] image bridge needs cv2/cv_bridge: %s", exc)
                return
            if not ok:
                log.warning("[ros_perception] image encode failed")
                return
            self._stats.frames += 1
            self._enqueue("perception_frame", {
                "t": _stamp_seconds(msg.header.stamp),
                "frame_id": msg.header.frame_id,
                "source": config.ROS_PERCEPTION_ANNOTATED_TOPIC,
                "encoding": "jpeg",
                "width": int(msg.width),
                "height": int(msg.height),
                "seq": self._stats.frames,
                "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
            })

        node.create_subscription(String, config.ROS_PERCEPTION_DETECTIONS_TOPIC, on_detections, reliable_qos)
        if config.ROS_PERCEPTION_ENABLE_ANNOTATED:
            node.create_subscription(Image, config.ROS_PERCEPTION_ANNOTATED_TOPIC, on_image, image_qos)
        node.create_timer(1.0, self._enqueue_status)
        log.info(
            "[ros_perception] subscribed detections=%s annotated=%s",
            config.ROS_PERCEPTION_DETECTIONS_TOPIC,
            config.ROS_PERCEPTION_ANNOTATED_TOPIC,
        )

        try:
            while not self._stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()

    def _normalize_detection_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        objects = payload.get("objects")
        if not isinstance(objects, list):
            payload["objects"] = []
            return payload
        normalized: list[dict[str, Any]] = []
        for idx, obj in enumerate(objects, start=1):
            if not isinstance(obj, dict):
                continue
            bbox = obj.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                bbox = [float(v) for v in bbox]
            else:
                bbox = None
            normalized.append({
                "id": int(obj.get("id", idx)),
                "cls": str(obj.get("cls", obj.get("label", "unknown"))),
                "conf": float(obj.get("conf", 0.0)),
                "bbox": bbox,
                "is_primary": bool(obj.get("is_primary", False)),
                "is_candidate": bool(obj.get("is_candidate", idx == 1)),
                "confirmed": bool(obj.get("confirmed", False)),
            })
        payload["objects"] = normalized
        return payload

    def _enqueue_status(self) -> None:
        queue_depth = self._queue.qsize() if self._queue is not None else 0
        self._enqueue("perception_diagnostics", {
            "t": time.time(),
            "connected": True,
            "detections": self._stats.detections,
            "frames": self._stats.frames,
            "dropped_frames": self._stats.dropped,
            "queue_depth": queue_depth,
        })
