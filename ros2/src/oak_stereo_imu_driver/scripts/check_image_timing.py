#!/usr/bin/env python3
"""
Verify stereo image message timing linearity by plotting timestamps and inter-message deltas.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
LEFT_TOPIC = "/oak/left/image_raw"
RIGHT_TOPIC = "/oak/right/image_raw"
NUM_MESSAGES = 300
# ---------------------


class ImageTimingChecker(Node):
    def __init__(self):
        super().__init__("image_timing_checker")
        self.left_timestamps = []
        self.right_timestamps = []
        self.get_logger().info(
            f"Subscribing to '{LEFT_TOPIC}' and '{RIGHT_TOPIC}', "
            f"collecting {NUM_MESSAGES} messages each..."
        )
        self.left_sub = self.create_subscription(Image, LEFT_TOPIC, self._left_cb, 10)
        self.right_sub = self.create_subscription(Image, RIGHT_TOPIC, self._right_cb, 10)

    def _left_cb(self, msg: Image):
        if len(self.left_timestamps) >= NUM_MESSAGES:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.left_timestamps.append(t)
        count = len(self.left_timestamps)
        if count % 50 == 0:
            self.get_logger().info(f"  Left: {count}/{NUM_MESSAGES}")
        if count >= NUM_MESSAGES and len(self.right_timestamps) >= NUM_MESSAGES:
            self._plot()
            rclpy.shutdown()

    def _right_cb(self, msg: Image):
        if len(self.right_timestamps) >= NUM_MESSAGES:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.right_timestamps.append(t)
        count = len(self.right_timestamps)
        if count % 50 == 0:
            self.get_logger().info(f"  Right: {count}/{NUM_MESSAGES}")
        if count >= NUM_MESSAGES and len(self.left_timestamps) >= NUM_MESSAGES:
            self._plot()
            rclpy.shutdown()

    def _plot(self):
        left_ts = np.array(self.left_timestamps)
        right_ts = np.array(self.right_timestamps)

        # Use left as time reference
        t0 = min(left_ts[0], right_ts[0])
        left_ts -= t0
        right_ts -= t0

        left_deltas_ms = np.diff(left_ts) * 1000.0
        right_deltas_ms = np.diff(right_ts) * 1000.0

        # Stereo sync: difference between left and right timestamps
        n = min(len(self.left_timestamps), len(self.right_timestamps))
        sync_diff_ms = (left_ts[:n] - right_ts[:n]) * 1000.0

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f"Stereo Image Timing Analysis ({NUM_MESSAGES} msgs)")

        # --- Top: timestamps vs sequence ---
        ax = axes[0]
        seq_l = np.arange(len(left_ts))
        seq_r = np.arange(len(right_ts))
        ax.plot(seq_l, left_ts, linewidth=1, label="Left")
        ax.plot(seq_r, right_ts, linewidth=1, label="Right", alpha=0.7)
        ax.set_xlabel("Message sequence number")
        ax.set_ylabel("Timestamp (s, relative)")
        ax.set_title("Timestamp linearity (ideal = straight line)")
        ax.legend()
        ax.grid(True)

        # --- Middle: delta time between consecutive messages ---
        ax = axes[1]
        ax.plot(seq_l[1:], left_deltas_ms, linewidth=0.8, label="Left Δt")
        ax.plot(seq_r[1:], right_deltas_ms, linewidth=0.8, label="Right Δt", alpha=0.7)
        expected_dt_ms = (left_ts[-1] / (len(left_ts) - 1)) * 1000.0
        ax.axhline(expected_dt_ms, color="r", linestyle="--", linewidth=1,
                    label=f"Expected {expected_dt_ms:.2f} ms")
        ax.set_xlabel("Message sequence number")
        ax.set_ylabel("Δt between messages (ms)")
        ax.set_title("Inter-message delta — spikes = dropped frames / jitter")
        ax.legend()
        ax.grid(True)

        # --- Bottom: stereo sync difference ---
        ax = axes[2]
        ax.plot(np.arange(n), sync_diff_ms, linewidth=0.8)
        ax.axhline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Message sequence number")
        ax.set_ylabel("Left − Right timestamp (ms)")
        ax.set_title("Stereo sync — ideal = 0 ms (left and right captured simultaneously)")
        ax.grid(True)

        plt.tight_layout()

        self.get_logger().info(
            f"Left  delta stats — mean: {left_deltas_ms.mean():.3f} ms, "
            f"std: {left_deltas_ms.std():.3f} ms, "
            f"max: {left_deltas_ms.max():.3f} ms, "
            f"min: {left_deltas_ms.min():.3f} ms"
        )
        self.get_logger().info(
            f"Right delta stats — mean: {right_deltas_ms.mean():.3f} ms, "
            f"std: {right_deltas_ms.std():.3f} ms, "
            f"max: {right_deltas_ms.max():.3f} ms, "
            f"min: {right_deltas_ms.min():.3f} ms"
        )
        self.get_logger().info(
            f"Stereo sync stats — mean: {sync_diff_ms.mean():.3f} ms, "
            f"std: {sync_diff_ms.std():.3f} ms, "
            f"max: {np.abs(sync_diff_ms).max():.3f} ms"
        )

        plt.show()


def main():
    rclpy.init()
    node = ImageTimingChecker()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
