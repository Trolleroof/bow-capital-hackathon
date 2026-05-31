"""Desktop ROS2 SLAM bridge.

This module runs inside the CombatOS orchestrator on the desktop. It subscribes
to ROS2 topics over DDS from the Jetson and publishes normalized CombatOS bus
topics for the browser. The expensive image encoding and JSON/base64 work stays
off the Jetson.
"""
from __future__ import annotations

import asyncio
import base64
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
class _StreamStats:
    last_sent: float = 0.0
    sent: int = 0
    dropped: int = 0


class RosSlamModule(AbstractModule):
    name = "ros_slam"

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._tracking = "NO_LOCK"
        self._path_max = max(1, config.ROS_SLAM_PATH_MAX_POSES)
        self._video_period = 1.0 / max(0.1, config.ROS_SLAM_VIDEO_FPS)
        self._jpeg_quality = max(20, min(95, config.ROS_SLAM_JPEG_QUALITY))
        self._stats = {
            "camera_frame": _StreamStats(),
            "slam_frame": _StreamStats(),
        }

    async def run(self) -> None:
        if not config.ENABLE_ROS_SLAM:
            log.info("[ros_slam] disabled; set COMBATOS_ROS_SLAM=1 to subscribe to ROS2 topics")
            while True:
                await asyncio.sleep(3600)

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=256)
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin_ros, name="combatos_ros_slam", daemon=True)
        self._thread.start()
        log.info("[ros_slam] desktop ROS2 bridge enabled")

        try:
            while True:
                topic, payload = await self._queue.get()
                await router.publish(topic, payload)
                system_state.beat("nav")
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
                    old_topic, _ = queue.get_nowait()
                    if old_topic in self._stats:
                        self._stats[old_topic].dropped += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait((topic, payload))
            except asyncio.QueueFull:
                if topic in self._stats:
                    self._stats[topic].dropped += 1

        loop.call_soon_threadsafe(put)

    def _spin_ros(self) -> None:
        try:
            import cv2
            import rclpy
            from cv_bridge import CvBridge, CvBridgeError
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Odometry, Path
            from rclpy.node import Node
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
            from std_msgs.msg import String
        except Exception as exc:
            log.error("[ros_slam] ROS2 dependencies unavailable: %s", exc)
            log.error("[ros_slam] source ROS2 and install cv_bridge/python3-opencv before enabling COMBATOS_ROS_SLAM")
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        node = Node("combatos_orchestrator_slam_bridge")
        bridge = CvBridge()
        reliable_qos = QoSProfile(depth=10)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        def on_pose(msg: PoseStamped) -> None:
            pose = msg.pose
            self._enqueue("pose", {
                "t": _stamp_seconds(msg.header.stamp),
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
                "qx": float(pose.orientation.x),
                "qy": float(pose.orientation.y),
                "qz": float(pose.orientation.z),
                "qw": float(pose.orientation.w),
                "gps": False,
                "tracking": self._tracking,
            })

        def on_odom(msg: Odometry) -> None:
            pose = msg.pose.pose
            twist = msg.twist.twist
            self._enqueue("slam_odometry", {
                "t": _stamp_seconds(msg.header.stamp),
                "frame_id": msg.header.frame_id,
                "child_frame_id": msg.child_frame_id,
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
                "qx": float(pose.orientation.x),
                "qy": float(pose.orientation.y),
                "qz": float(pose.orientation.z),
                "qw": float(pose.orientation.w),
                "vx": float(twist.linear.x),
                "vy": float(twist.linear.y),
                "vz": float(twist.linear.z),
                "wx": float(twist.angular.x),
                "wy": float(twist.angular.y),
                "wz": float(twist.angular.z),
                "tracking": self._tracking,
            })

        def on_path(msg: Path) -> None:
            poses = msg.poses[-self._path_max:]
            self._enqueue("slam_path", {
                "t": _stamp_seconds(msg.header.stamp),
                "frame_id": msg.header.frame_id,
                "poses": [
                    {
                        "t": _stamp_seconds(p.header.stamp),
                        "x": float(p.pose.position.x),
                        "y": float(p.pose.position.y),
                        "z": float(p.pose.position.z),
                    }
                    for p in poses
                ],
            })

        def on_status(msg: String) -> None:
            self._tracking = msg.data or "NO_LOCK"
            self._enqueue_status()

        def on_image(msg: Image, topic: str, source: str) -> None:
            now = time.monotonic()
            stats = self._stats[topic]
            if now - stats.last_sent < self._video_period:
                stats.dropped += 1
                return
            stats.last_sent = now

            try:
                image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            except (CvBridgeError, ValueError) as exc:
                log.warning("[ros_slam] image encode failed for %s: %s", source, exc)
                return
            if not ok:
                log.warning("[ros_slam] image encode failed for %s", source)
                return

            stats.sent += 1
            self._enqueue(topic, {
                "t": _stamp_seconds(msg.header.stamp),
                "frame_id": msg.header.frame_id,
                "source": source,
                "encoding": "jpeg",
                "width": int(msg.width),
                "height": int(msg.height),
                "seq": stats.sent,
                "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
            })

        node.create_subscription(PoseStamped, config.ROS_SLAM_POSE_TOPIC, on_pose, reliable_qos)
        node.create_subscription(Odometry, config.ROS_SLAM_ODOM_TOPIC, on_odom, reliable_qos)
        node.create_subscription(Path, config.ROS_SLAM_PATH_TOPIC, on_path, reliable_qos)
        node.create_subscription(String, config.ROS_SLAM_STATUS_TOPIC, on_status, reliable_qos)
        if config.ROS_SLAM_ENABLE_CAMERA:
            node.create_subscription(
                Image,
                config.ROS_SLAM_CAMERA_TOPIC,
                lambda msg: on_image(msg, "camera_frame", config.ROS_SLAM_CAMERA_TOPIC),
                image_qos,
            )
        if config.ROS_SLAM_ENABLE_ANNOTATED:
            node.create_subscription(
                Image,
                config.ROS_SLAM_ANNOTATED_TOPIC,
                lambda msg: on_image(msg, "slam_frame", config.ROS_SLAM_ANNOTATED_TOPIC),
                image_qos,
            )
        node.create_timer(1.0, self._enqueue_status)

        log.info(
            "[ros_slam] subscribed pose=%s camera=%s annotated=%s",
            config.ROS_SLAM_POSE_TOPIC,
            config.ROS_SLAM_CAMERA_TOPIC,
            config.ROS_SLAM_ANNOTATED_TOPIC,
        )

        try:
            while not self._stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()

    def _enqueue_status(self) -> None:
        self._enqueue("slam_status", {
            "t": time.time(),
            "tracking": self._tracking,
            "connected": True,
            "camera_frames": self._stats["camera_frame"].sent,
            "annotated_frames": self._stats["slam_frame"].sent,
            "dropped_frames": sum(s.dropped for s in self._stats.values()),
        })
        queue_depth = self._queue.qsize() if self._queue is not None else 0
        self._enqueue("slam_diagnostics", {
            "t": time.time(),
            "tracking": self._tracking,
            "dropped_frames": sum(s.dropped for s in self._stats.values()),
            "camera_frames": self._stats["camera_frame"].sent,
            "annotated_frames": self._stats["slam_frame"].sent,
            "queue_depth": queue_depth,
        })
