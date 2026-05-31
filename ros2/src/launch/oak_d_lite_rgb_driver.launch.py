#!/usr/bin/env python3

"""
OAK-D Lite launch file — RGB camera only.

Topics published:
  /oak/rgb/image_raw
  /oak/rgb/image_raw/compressed
  /oak/rgb/camera_info
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration("name", default="oak").perform(context)
    namespace = LaunchConfiguration("namespace", default="").perform(context)
    fps = float(LaunchConfiguration("fps", default="30.0").perform(context))

    container_name = f"{namespace}/{name}_container" if namespace else f"{name}_container"

    camera_params = [{
        "camera": {
            "i_pipeline_type": "RGB",
            "i_nn_type": "none",
            "i_enable_imu": False,
            "i_enable_ir": False,
            "i_usb_speed": "SUPER_PLUS",
        },
        "rgb": {
            "i_publish_topic": True,
            "i_publish_compressed": False,
            "i_fps": fps,
            "i_resolution": "1080p",
            "i_set_isp_scale": True,
            "i_isp_scale_num": 1,
            "i_isp_scale_den": 3,
            "i_interleaved": False,
            "i_disable_node": False,
        },
    }]

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
        DeclareLaunchArgument("fps",       default_value="30.0"),
        OpaqueFunction(function=launch_setup),
    ])
