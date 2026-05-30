#!/usr/bin/env python3

"""
OAK-D launch file for camera driver only (no SLAM).

Launches:
  1. depthai_ros_driver camera container (stereo pipeline + IMU)
  2. Left and right image rectification nodes
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration("name", default="oak").perform(context)
    namespace = LaunchConfiguration("namespace", default="").perform(context)

    depthai_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "oak_stereo_imu.yaml"
    )

    camera_params = [depthai_config] if os.path.exists(depthai_config) else []

    container_name = f"{namespace}/{name}_container" if namespace else f"{name}_container"

    return [
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
                    parameters=camera_params,
                )
            ],
            output="both",
        ),
        LoadComposableNodes(
            target_container=container_name,
            composable_node_descriptions=[
                ComposableNode(
                    package="image_proc",
                    plugin="image_proc::RectifyNode",
                    name="rectify_left_node",
                    namespace=namespace,
                    remappings=[
                        ("image", f"{name}/left/image_raw"),
                        ("camera_info", f"{name}/left/camera_info"),
                        ("image_rect", f"{name}/left/image_rect"),
                    ],
                ),
                ComposableNode(
                    package="image_proc",
                    plugin="image_proc::RectifyNode",
                    name="rectify_right_node",
                    namespace=namespace,
                    remappings=[
                        ("image", f"{name}/right/image_raw"),
                        ("camera_info", f"{name}/right/camera_info"),
                        ("image_rect", f"{name}/right/image_rect"),
                    ],
                ),
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
