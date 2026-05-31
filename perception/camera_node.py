"""ROS2 camera capture node.

Reads frames from VIDEO_SOURCE (webcam index, file path, or RTSP URL) and
publishes them to /camera/image_raw (sensor_msgs/Image, BGR8 encoding).

Optional downscaling is applied here (PROC_WIDTH) so all downstream nodes
receive a consistently sized image.

Usage:
    source /opt/ros/<distro>/setup.bash   # e.g. jazzy or humble
    cd perception
    python camera_node.py           # uses .env / env vars for VIDEO_SOURCE
    VIDEO_SOURCE=/dev/video0 python camera_node.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import config


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_node")

        source = config.VIDEO_SOURCE
        self._cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self._cap.isOpened():
            self.get_logger().fatal(f"Cannot open video source: {source!r}")
            sys.exit(1)

        src_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        if config.PROC_WIDTH > 0 and config.PROC_WIDTH < src_w:
            self._proc_w = config.PROC_WIDTH
            self._proc_h = round(src_h * self._proc_w / src_w)
        else:
            self._proc_w = self._proc_h = 0  # no resize

        self.get_logger().info(
            f"Camera: {src_w}x{src_h} @ {fps:.1f} fps  source={source!r}"
            + (f"  -> resizing to {self._proc_w}x{self._proc_h}" if self._proc_w else "")
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub    = self.create_publisher(Image, config.ROS_CAMERA_TOPIC, qos)
        self._bridge = CvBridge()
        self._timer  = self.create_timer(1.0 / fps, self._capture)

        self.get_logger().info(f"Publishing to {config.ROS_CAMERA_TOPIC}")

    def _capture(self) -> None:
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warn("Camera read failed -- end of stream or error")
            return

        if self._proc_w:
            frame = cv2.resize(frame, (self._proc_w, self._proc_h), interpolation=cv2.INTER_LINEAR)

        msg = self._bridge.cv2_to_imgmsg(frame, "bgr8")
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self._pub.publish(msg)

    def destroy_node(self) -> None:
        self._cap.release()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
