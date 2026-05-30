#!/usr/bin/env python3

"""
OAK Camera Driver Node for ORB-SLAM3 Monocular Mode.

This node bridges the OAK-D camera (via depthai_ros_driver) to the mono_node_cpp
ORB-SLAM3 node for real-time monocular SLAM.

Usage:
    # Terminal 1 - Launch OAK camera
    ros2 launch depthai_ros_driver camera.launch.py

    # Terminal 2 - Launch ORB-SLAM3 mono node
    ros2 run ros2_orb_slam3 mono_node_cpp --ros-args -p node_name_arg:=mono_slam_cpp

    # Terminal 3 - Launch this driver
    ros2 run ros2_orb_slam3 oak_camera_driver.py --ros-args -p settings_name:=OAK

Parameters:
    settings_name: Name of the YAML config file (without .yaml extension)
    image_topic: Topic to subscribe for images (default: /oak/rgb/image_rect)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64
from cv_bridge import CvBridge, CvBridgeError


class OakCameraDriver(Node):
    def __init__(self, node_name="oak_camera_driver"):
        super().__init__(node_name)

        # Declare parameters
        self.declare_parameter("settings_name", "OAK")
        self.declare_parameter("image_topic", "/oak/rgb/image_rect")

        # Get parameter values
        self.settings_name = str(self.get_parameter('settings_name').value)
        self.image_topic = str(self.get_parameter('image_topic').value)

        self.get_logger().info("-------------- OAK Camera Driver --------------------------")
        self.get_logger().info(f"Settings name: {self.settings_name}")
        self.get_logger().info(f"Image topic: {self.image_topic}")

        # CvBridge for image conversion
        self.br = CvBridge()

        # State variables
        self.handshake_complete = False
        self.frame_count = 0

        # Topic names for mono_node_cpp (must match expected names)
        self.pub_exp_config_name = "/mono_py_driver/experiment_settings"
        self.sub_exp_ack_name = "/mono_py_driver/exp_settings_ack"
        self.pub_img_to_agent_name = "/mono_py_driver/img_msg"
        self.pub_timestep_to_agent_name = "/mono_py_driver/timestep_msg"

        # Publishers for mono_node_cpp
        self.publish_exp_config_ = self.create_publisher(
            String, self.pub_exp_config_name, 1)
        self.publish_img_msg_ = self.create_publisher(
            Image, self.pub_img_to_agent_name, 1)
        self.publish_timestep_msg_ = self.create_publisher(
            Float64, self.pub_timestep_to_agent_name, 1)

        # Subscriber for handshake acknowledgement
        self.subscribe_exp_ack_ = self.create_subscription(
            String,
            self.sub_exp_ack_name,
            self.ack_callback,
            10)

        # Image subscriber (created now, but only forwards after handshake)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            sensor_qos)

        # Timer for handshake - publishes config every 100ms until ACK received
        self.handshake_timer = self.create_timer(0.1, self.handshake_timer_callback)

        self.get_logger().info("Waiting for mono_node_cpp to acknowledge settings...")
        self.get_logger().info("Make sure mono_node_cpp is running!")

    def handshake_timer_callback(self):
        """Timer callback to send config until handshake is complete."""
        if not self.handshake_complete:
            msg = String()
            msg.data = self.settings_name
            self.publish_exp_config_.publish(msg)
        else:
            # Handshake done, cancel this timer
            self.handshake_timer.cancel()

    def ack_callback(self, msg):
        """Callback for handshake acknowledgement from mono_node_cpp."""
        self.get_logger().info(f"Got ack: {msg.data}")
        if msg.data == "ACK" and not self.handshake_complete:
            self.handshake_complete = True
            self.get_logger().info("Handshake complete! Starting image streaming...")

    def image_callback(self, msg):
        """
        Callback for incoming camera images.
        Only forwards images after handshake is complete.
        """
        # Skip if handshake not complete
        if not self.handshake_complete:
            return

        try:
            # Extract timestamp from the image message header
            # Convert ROS time to nanoseconds (similar to EuRoC format)
            timestamp_ns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec

            # Create and publish timestamp message
            timestep_msg = Float64()
            timestep_msg.data = float(timestamp_ns)

            # Publish timestamp first, then image (order matters!)
            self.publish_timestep_msg_.publish(timestep_msg)
            self.publish_img_msg_.publish(msg)

            self.frame_count += 1

            # Log periodically
            if self.frame_count % 100 == 0:
                self.get_logger().info(f"Published frame {self.frame_count}")

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = OakCameraDriver("oak_camera_driver")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
