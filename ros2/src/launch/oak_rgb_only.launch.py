#!/usr/bin/env python3

"""
Optimized OAK-D launch file for monocular SLAM
Only enables RGB camera with rectification to minimize PoE bandwidth usage
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch.conditions import IfCondition


def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration("name", default="oak").perform(context)
    namespace = LaunchConfiguration("namespace", default="").perform(context)

    # Parameters optimized for monocular SLAM over PoE
    camera_params = {
        "camera": {
            "i_pipeline_type": "RGB",  # RGB only - no stereo cameras
            "i_nn_type": "none",  # No neural network processing
            "i_enable_imu": False,  # Disable IMU
            "i_enable_ir": False,  # Disable IR
            "i_usb_speed": "SUPER_PLUS",  # Use maximum USB speed available
        },
        "rgb": {
            "i_publish_topic": True,
            "i_width": 1280,  # Can reduce to 640 if still slow
            "i_height": 720,  # Can reduce to 480 if still slow
            "i_fps": 30.0,  # Can reduce to 15 if needed
            "i_board_socket_id": 0,
            "i_resolution": "720p",  # Options: 1080p, 720p, 480p
            "i_set_isp_scale": False,  # Disable ISP scaling for better performance
            "i_interleaved": False,
            "i_keep_preview_aspect_ratio": True,
        },
    }

    return [
        # Main camera node container
        ComposableNodeContainer(
            name=f"{name}_container",
            namespace=namespace,
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                ComposableNode(
                    package="depthai_ros_driver",
                    plugin="depthai_ros_driver::Camera",
                    name=name,
                    namespace=namespace,
                    parameters=[camera_params],
                )
            ],
            output="both",
        ),
        # RGB rectification node
        LoadComposableNodes(
            target_container=f"{namespace}/{name}_container" if namespace else f"{name}_container",
            composable_node_descriptions=[
                ComposableNode(
                    package="image_proc",
                    plugin="image_proc::RectifyNode",
                    name="rectify_rgb_node",
                    namespace=namespace,
                    remappings=[
                        ("image", f"{name}/rgb/image_raw"),
                        ("camera_info", f"{name}/rgb/camera_info"),
                        ("image_rect", f"{name}/rgb/image_rect"),
                    ],
                )
            ],
        ),
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("name", default_value="oak"),
        DeclareLaunchArgument("namespace", default_value=""),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
