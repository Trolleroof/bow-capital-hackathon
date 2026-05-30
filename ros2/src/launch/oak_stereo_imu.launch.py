#!/usr/bin/env python3

"""
OAK-D launch file for Stereo-Inertial ORB-SLAM3

Launches:
  1. depthai_ros_driver camera container (stereo pipeline + IMU)
  2. Left and right image rectification nodes
  3. stereo_inertial_node_cpp ORB-SLAM3 node

Defaults are bandwidth-safe for PoE (15fps, 400p).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration("name", default="oak").perform(context)
    namespace = LaunchConfiguration("namespace", default="").perform(context)
    resolution = LaunchConfiguration("resolution", default="400p").perform(context)
    fps = float(LaunchConfiguration("fps", default="15.0").perform(context))
    settings_name = LaunchConfiguration("settings_name", default="OAK").perform(context)

    # Load depthai parameter file
    depthai_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "oak_stereo_imu.yaml"
    )

    # Override FPS from launch arg
    camera_params = {}
    if os.path.exists(depthai_config):
        camera_params = [depthai_config]
    else:
        # Inline fallback if config file not found
        camera_params = [{
            "camera": {
                "i_pipeline_type": "Stereo",
                "i_nn_type": "none",
            },
            "pipeline_gen": {
                "i_enable_imu": True,
                "i_enable_sync": False,
            },
            "imu": {
                "i_message_type": "IMU",
            },
            "left": {
                "i_publish_topic": True,
                "i_publish_compressed": False,
                "i_enable_nn": False,
                "i_fps": fps,
                "i_resolution": resolution,
            },
            "right": {
                "i_publish_topic": True,
                "i_publish_compressed": False,
                "i_enable_nn": False,
                "i_fps": fps,
                "i_resolution": resolution,
            },
            "rgb": {
                "i_disable_node": True,
            },
        }]

    container_name = f"{namespace}/{name}_container" if namespace else f"{name}_container"

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
                    parameters=camera_params,
                )
            ],
            output="both",
        ),
        # Left rectification node
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
        # ORB-SLAM3 Stereo-Inertial node
        Node(
            package="ros2_orb_slam3",
            executable="stereo_inertial_node_cpp",
            name="stereo_inertial_node",
            namespace=namespace,
            parameters=[{
                "left_image_topic": f"/{name}/left/image_rect" if not namespace else f"/{namespace}/{name}/left/image_rect",
                "right_image_topic": f"/{name}/right/image_rect" if not namespace else f"/{namespace}/{name}/right/image_rect",
                "imu_topic": f"/{name}/imu/data" if not namespace else f"/{namespace}/{name}/imu/data",
                "settings_name": settings_name,
            }],
            output="screen",
        ),
    ]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("name", default_value="oak"),
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("resolution", default_value="400p"),
        DeclareLaunchArgument("fps", default_value="15.0"),
        DeclareLaunchArgument("settings_name", default_value="OAK"),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
