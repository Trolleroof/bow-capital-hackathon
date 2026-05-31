"""Desktop ROS2 perception bridge.

Subscribes to YOLOX ROS2 topics and publishes normalized CombatOS bus topics.
The annotated YOLOX image is exposed as ``camera_frame`` so the existing OAK
camera panel renders it without a second frontend code path.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import Any

from .base import AbstractModule
from .. import config
from ..bus import image_router, router
from ..state import system_state

log = logging.getLogger(__name__)


def _stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class RosPerceptionModule(AbstractModule):
    name = "ros_perception"

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, dict[str, Any], bool]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._seq = 0
        self._jpeg_quality = max(20, min(95, config.ROS_SLAM_JPEG_QUALITY))

    async def run(self) -> None:
        if not config.ENABLE_ROS_PERCEPTION:
            log.info("[ros_perception] disabled; set COMBATOS_ROS_PERCEPTION=1 to enable")
            while True:
                await asyncio.sleep(3600)

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=64)
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin_ros, name="combatos_ros_perception", daemon=True)
        self._thread.start()
        log.info("[ros_perception] desktop ROS2 bridge enabled")

        try:
            while True:
                topic, payload, is_image = await self._queue.get()
                if is_image:
                    await image_router.publish(topic, payload)
                else:
                    await router.publish(topic, payload)
                system_state.beat("perception")
        finally:
            self._stop.set()

    def _enqueue(self, topic: str, payload: dict[str, Any], *, is_image: bool = False) -> None:
        loop = self._loop
        queue = self._queue
        if loop is None or queue is None:
            return

        def put() -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait((topic, payload, is_image))
            except asyncio.QueueFull:
                pass

        loop.call_soon_threadsafe(put)

    def _spin_ros(self) -> None:
        try:
            import cv2
            import rclpy
            from cv_bridge import CvBridge, CvBridgeError
            from rclpy.node import Node
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
            from std_msgs.msg import String
        except Exception as exc:
            log.error("[ros_perception] ROS2 perception dependencies unavailable: %s", exc)
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        node = Node("combatos_orchestrator_perception_bridge")
        bridge = CvBridge()
        reliable_qos = QoSProfile(depth=10)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        def on_detections(msg: String) -> None:
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                log.warning("[ros_perception] invalid detections JSON on %s", config.ROS_PERCEPTION_DETECTIONS_TOPIC)
                return
            if not isinstance(payload, dict):
                return
            payload.pop("topic", None)
            self._enqueue("detections", payload)

        def on_annotated(msg: Image) -> None:
            try:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            except (CvBridgeError, ValueError) as exc:
                log.warning("[ros_perception] annotated image encode failed: %s", exc)
                return
            except Exception as exc:
                log.warning("[ros_perception] annotated image needs cv2/cv_bridge: %s", exc)
                return
            if not ok:
                log.warning("[ros_perception] annotated image encode failed")
                return

            self._seq += 1
            self._enqueue(
                config.ROS_PERCEPTION_FRAME_TOPIC,
                {
                    "t": _stamp_seconds(msg.header.stamp),
                    "frame_id": msg.header.frame_id,
                    "source": config.ROS_PERCEPTION_ANNOTATED_TOPIC,
                    "encoding": "jpeg",
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "seq": self._seq,
                    "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                },
                is_image=True,
            )

        if config.ROS_PERCEPTION_ENABLE_DETECTIONS:
            node.create_subscription(String, config.ROS_PERCEPTION_DETECTIONS_TOPIC, on_detections, reliable_qos)
        if config.ROS_PERCEPTION_ENABLE_ANNOTATED:
            node.create_subscription(Image, config.ROS_PERCEPTION_ANNOTATED_TOPIC, on_annotated, image_qos)

        log.info(
            "[ros_perception] subscribed detections=%s annotated=%s -> frame_topic=%s",
            config.ROS_PERCEPTION_DETECTIONS_TOPIC,
            config.ROS_PERCEPTION_ANNOTATED_TOPIC,
            config.ROS_PERCEPTION_FRAME_TOPIC,
        )

        try:
            while not self._stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
