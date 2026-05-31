#!/usr/bin/env python3

"""
OAK-D Lite launch file — driver only (RGBD pipeline).

Topics published:
  /oak/left/image_raw          /oak/left/image_raw/compressed
  /oak/left/camera_info

  /oak/right/image_raw         /oak/right/image_raw/compressed
  /oak/right/camera_info

  /oak/rgb/image_raw           /oak/rgb/image_raw/compressed
  /oak/rgb/camera_info
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
        "config", "oak_d_lite.yaml"
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
                    package="image_transport",
                    plugin="image_transport::RepublishNode",
                    name="compress_left",
                    namespace=namespace,
                    parameters=[{"in_transport": "raw", "out_transport": "compressed"}],
                    remappings=[
                        ("in",  f"{name}/left/image_raw"),
                        ("out", f"{name}/left/image_raw"),
                    ],
                ),
                ComposableNode(
                    package="image_transport",
                    plugin="image_transport::RepublishNode",
                    name="compress_right",
                    namespace=namespace,
                    parameters=[{"in_transport": "raw", "out_transport": "compressed"}],
                    remappings=[
                        ("in",  f"{name}/right/image_raw"),
                        ("out", f"{name}/right/image_raw"),
                    ],
                ),
                ComposableNode(
                    package="image_transport",
                    plugin="image_transport::RepublishNode",
                    name="compress_rgb",
                    namespace=namespace,
                    parameters=[{"in_transport": "raw", "out_transport": "compressed"}],
                    remappings=[
                        ("in",  f"{name}/rgb/image_raw"),
                        ("out", f"{name}/rgb/image_raw"),
                    ],
                ),
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("name",      default_value="oak"),
        DeclareLaunchArgument("namespace", default_value=""),
        OpaqueFunction(function=launch_setup),
    ])
