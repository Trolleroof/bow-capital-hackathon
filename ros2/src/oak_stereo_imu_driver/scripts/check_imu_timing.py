#!/usr/bin/env python3
"""
Verify IMU message timing linearity by plotting timestamps and inter-message deltas.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
IMU_TOPIC = "/oak/imu/data"
NUM_MESSAGES = 1000
# ---------------------


class ImuTimingChecker(Node):
    def __init__(self):
        super().__init__("imu_timing_checker")
        self.timestamps = []
        self.get_logger().info(
            f"Subscribing to '{IMU_TOPIC}', collecting {NUM_MESSAGES} messages..."
        )
        self.sub = self.create_subscription(Imu, IMU_TOPIC, self._cb, 10)

    def _cb(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.timestamps.append(t)
        count = len(self.timestamps)
        if count % 100 == 0:
            self.get_logger().info(f"  Collected {count}/{NUM_MESSAGES}")
        if count >= NUM_MESSAGES:
            self.sub  # keep reference alive until plot
            self._plot()
            rclpy.shutdown()

    def _plot(self):
        ts = np.array(self.timestamps)
        ts -= ts[0]  # relative time starting at 0
        seq = np.arange(len(ts))
        deltas_ms = np.diff(ts) * 1000.0  # ms

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        fig.suptitle(f"IMU Timing Analysis — {IMU_TOPIC} ({NUM_MESSAGES} msgs)")

        # --- Top: timestamps vs sequence ---
        ax1.plot(seq, ts, linewidth=1)
        ax1.set_xlabel("Message sequence number")
        ax1.set_ylabel("Timestamp (s, relative)")
        ax1.set_title("Timestamp linearity (ideal = straight line)")
        ax1.grid(True)

        # --- Bottom: delta time between consecutive messages ---
        expected_dt_ms = (ts[-1] / (len(ts) - 1)) * 1000.0
        ax2.plot(seq[1:], deltas_ms, linewidth=0.8, label="Δt (ms)")
        ax2.axhline(expected_dt_ms, color="r", linestyle="--", linewidth=1,
                    label=f"Expected {expected_dt_ms:.2f} ms")
        ax2.set_xlabel("Message sequence number")
        ax2.set_ylabel("Δt between messages (ms)")
        ax2.set_title("Inter-message delta — spikes = dropped frames / jitter")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        self.get_logger().info(
            f"Delta stats — mean: {deltas_ms.mean():.3f} ms, "
            f"std: {deltas_ms.std():.3f} ms, "
            f"max: {deltas_ms.max():.3f} ms, "
            f"min: {deltas_ms.min():.3f} ms"
        )

        plt.show()


def main():
    rclpy.init()
    node = ImuTimingChecker()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
