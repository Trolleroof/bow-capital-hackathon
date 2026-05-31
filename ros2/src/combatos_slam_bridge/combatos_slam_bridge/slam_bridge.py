from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import rclpy
import websockets
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


def stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def pose_payload(msg: PoseStamped, tracking: str) -> dict[str, Any]:
    pose = msg.pose
    return {
        "topic": "pose",
        "t": stamp_seconds(msg.header.stamp),
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "z": float(pose.position.z),
        "qx": float(pose.orientation.x),
        "qy": float(pose.orientation.y),
        "qz": float(pose.orientation.z),
        "qw": float(pose.orientation.w),
        "gps": False,
        "tracking": tracking,
    }


@dataclass
class StreamStats:
    last_sent: float = 0.0
    sent: int = 0
    dropped: int = 0


class CombatOSSlamBridge(Node):
    def __init__(self) -> None:
        super().__init__("combatos_slam_bridge")

        self.declare_parameter("orch_ws", "ws://localhost:8000")
        self.declare_parameter("pose_topic", "/slam/pose")
        self.declare_parameter("odom_topic", "/slam/odometry")
        self.declare_parameter("path_topic", "/slam/path")
        self.declare_parameter("status_topic", "/slam/status")
        self.declare_parameter("camera_topic", "/oak/left/image_rect")
        self.declare_parameter("right_camera_topic", "/oak/right/image_rect")
        self.declare_parameter("annotated_topic", "/slam/tracked_image")
        self.declare_parameter("enable_camera", True)
        self.declare_parameter("enable_right_camera", False)
        self.declare_parameter("enable_annotated", True)
        self.declare_parameter("video_fps", 8.0)
        self.declare_parameter("jpeg_quality", 70)
        self.declare_parameter("path_max_poses", 240)
        self.declare_parameter("queue_size", 96)

        self.orch_ws = self.get_parameter("orch_ws").value
        self.video_period = 1.0 / max(0.1, float(self.get_parameter("video_fps").value))
        self.jpeg_quality = max(20, min(95, int(self.get_parameter("jpeg_quality").value)))
        self.path_max_poses = max(1, int(self.get_parameter("path_max_poses").value))
        self.tracking = "NO_LOCK"
        self.cv_bridge = CvBridge()
        self.stats = {
            "camera_frame": StreamStats(),
            "camera_right_frame": StreamStats(),
            "slam_frame": StreamStats(),
        }

        queue_size = max(8, int(self.get_parameter("queue_size").value))
        self.loop = asyncio.new_event_loop()
        self.outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._run_loop, name="combatos_ws", daemon=True)
        self.thread.start()

        reliable_qos = QoSProfile(depth=10)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(PoseStamped, self.get_parameter("pose_topic").value, self._on_pose, reliable_qos)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, reliable_qos)
        self.create_subscription(Path, self.get_parameter("path_topic").value, self._on_path, reliable_qos)
        self.create_subscription(String, self.get_parameter("status_topic").value, self._on_status, reliable_qos)

        if bool(self.get_parameter("enable_camera").value):
            self.create_subscription(
                Image,
                self.get_parameter("camera_topic").value,
                lambda msg: self._on_image(msg, "camera_frame", self.get_parameter("camera_topic").value),
                image_qos,
            )
        if bool(self.get_parameter("enable_right_camera").value):
            self.create_subscription(
                Image,
                self.get_parameter("right_camera_topic").value,
                lambda msg: self._on_image(msg, "camera_right_frame", self.get_parameter("right_camera_topic").value),
                image_qos,
            )
        if bool(self.get_parameter("enable_annotated").value):
            self.create_subscription(
                Image,
                self.get_parameter("annotated_topic").value,
                lambda msg: self._on_image(msg, "slam_frame", self.get_parameter("annotated_topic").value),
                image_qos,
            )

        self.create_timer(1.0, self._publish_diagnostics)
        self.get_logger().info(f"CombatOS SLAM bridge publishing to {self.orch_ws}")

    def destroy_node(self) -> bool:
        self.loop.call_soon_threadsafe(self.loop.stop)
        return super().destroy_node()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._ws_sender())
        self.loop.run_forever()

    def _enqueue(self, payload: dict[str, Any]) -> None:
        def put() -> None:
            if self.outbox.full():
                try:
                    old = self.outbox.get_nowait()
                    topic = old.get("topic")
                    if topic in self.stats:
                        self.stats[topic].dropped += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                self.outbox.put_nowait(payload)
            except asyncio.QueueFull:
                topic = payload.get("topic")
                if topic in self.stats:
                    self.stats[topic].dropped += 1

        self.loop.call_soon_threadsafe(put)

    async def _ws_sender(self) -> None:
        while True:
            try:
                async with websockets.connect(self.orch_ws, ping_interval=10, ping_timeout=5) as ws:
                    self.get_logger().info("Connected to CombatOS orchestrator")
                    while True:
                        payload = await self.outbox.get()
                        await ws.send(json.dumps(payload, separators=(",", ":")))
            except Exception as exc:
                self.get_logger().warning(f"CombatOS connection unavailable: {exc}")
                await asyncio.sleep(1.5)

    def _on_pose(self, msg: PoseStamped) -> None:
        self._enqueue(pose_payload(msg, self.tracking))

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        twist = msg.twist.twist
        self._enqueue(
            {
                "topic": "slam_odometry",
                "t": stamp_seconds(msg.header.stamp),
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
                "tracking": self.tracking,
            }
        )

    def _on_path(self, msg: Path) -> None:
        poses = msg.poses[-self.path_max_poses :]
        self._enqueue(
            {
                "topic": "slam_path",
                "t": stamp_seconds(msg.header.stamp),
                "frame_id": msg.header.frame_id,
                "poses": [
                    {
                        "t": stamp_seconds(p.header.stamp),
                        "x": float(p.pose.position.x),
                        "y": float(p.pose.position.y),
                        "z": float(p.pose.position.z),
                    }
                    for p in poses
                ],
            }
        )

    def _on_status(self, msg: String) -> None:
        tracking = msg.data or "NO_LOCK"
        self.tracking = tracking
        self._enqueue(
            {
                "topic": "slam_status",
                "t": time.time(),
                "tracking": tracking,
                "connected": True,
                "camera_frames": self.stats["camera_frame"].sent,
                "annotated_frames": self.stats["slam_frame"].sent,
                "dropped_frames": sum(s.dropped for s in self.stats.values()),
            }
        )

    def _on_image(self, msg: Image, topic: str, source: str) -> None:
        now = time.monotonic()
        stats = self.stats[topic]
        if now - stats.last_sent < self.video_period:
            stats.dropped += 1
            return
        stats.last_sent = now

        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        except (CvBridgeError, ValueError) as exc:
            self.get_logger().warning(f"image encode failed for {source}: {exc}")
            return
        if not ok:
            self.get_logger().warning(f"image encode failed for {source}")
            return

        stats.sent += 1
        self._enqueue(
            {
                "topic": topic,
                "t": stamp_seconds(msg.header.stamp),
                "frame_id": msg.header.frame_id,
                "source": source,
                "encoding": "jpeg",
                "width": int(msg.width),
                "height": int(msg.height),
                "seq": stats.sent,
                "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
        )

    def _publish_diagnostics(self) -> None:
        dropped = sum(s.dropped for s in self.stats.values())
        self._enqueue(
            {
                "topic": "slam_diagnostics",
                "t": time.time(),
                "tracking": self.tracking,
                "dropped_frames": dropped,
                "camera_frames": self.stats["camera_frame"].sent,
                "annotated_frames": self.stats["slam_frame"].sent,
                "queue_depth": self.outbox.qsize(),
            }
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CombatOSSlamBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
