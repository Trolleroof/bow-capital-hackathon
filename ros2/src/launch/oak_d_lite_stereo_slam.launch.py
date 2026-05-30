#!/usr/bin/env python3

"""
OAK-D Lite launch file for Stereo ORB-SLAM3 (no IMU).

Launches:
  1. depthai_ros_driver camera container (stereo pipeline)
  2. Left and right image rectification nodes
  3. stereo_node_cpp ORB-SLAM3 node with OAK_D_Lite settings
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    name = LaunchConfiguration("name", default="oak").perform(context)
    namespace = LaunchConfiguration("namespace", default="").perform(context)
    resolution = LaunchConfiguration("resolution", default="400p").perform(context)
    fps = float(LaunchConfiguration("fps", default="15.0").perform(context))
    settings_name = LaunchConfiguration("settings_name", default="OAK_D_Lite").perform(context)
    enable_combatos_bridge = LaunchConfiguration("enable_combatos_bridge", default="true")
    orch_ws = LaunchConfiguration("orch_ws", default="ws://localhost:8000").perform(context)
    video_fps = float(LaunchConfiguration("video_fps", default="8.0").perform(context))
    jpeg_quality = int(LaunchConfiguration("jpeg_quality", default="70").perform(context))
    enable_right_camera = LaunchConfiguration("enable_right_camera", default="false").perform(context).lower() in ("1", "true", "yes")

    depthai_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "oak_stereo_imu.yaml"
    )

    camera_params = [depthai_config] if os.path.exists(depthai_config) else [{
        "camera": {"i_pipeline_type": "Stereo", "i_nn_type": "none"},
        "pipeline_gen": {"i_enable_imu": False, "i_enable_sync": False},
        "left":  {"i_publish_topic": True, "i_publish_compressed": False, "i_enable_nn": False, "i_fps": fps, "i_resolution": resolution},
        "right": {"i_publish_topic": True, "i_publish_compressed": False, "i_enable_nn": False, "i_fps": fps, "i_resolution": resolution},
        "rgb":   {"i_disable_node": True},
    }]

    container_name = f"{namespace}/{name}_container" if namespace else f"{name}_container"

    prefix = f"/{namespace}/{name}" if namespace else f"/{name}"

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
                        ("image",       f"{name}/left/image_raw"),
                        ("camera_info", f"{name}/left/camera_info"),
                        ("image_rect",  f"{name}/left/image_rect"),
                    ],
                ),
                ComposableNode(
                    package="image_proc",
                    plugin="image_proc::RectifyNode",
                    name="rectify_right_node",
                    namespace=namespace,
                    remappings=[
                        ("image",       f"{name}/right/image_raw"),
                        ("camera_info", f"{name}/right/camera_info"),
                        ("image_rect",  f"{name}/right/image_rect"),
                    ],
                ),
            ],
        ),
        Node(
            package="ros2_orb_slam3",
            executable="stereo_node_cpp",
            name="stereo_node",
            namespace=namespace,
            parameters=[{
                "left_image_topic":  f"{prefix}/left/image_rect",
                "right_image_topic": f"{prefix}/right/image_rect",
                "settings_name":     settings_name,
            }],
            output="screen",
        ),
        Node(
            package="combatos_slam_bridge",
            executable="slam_bridge",
            name="combatos_slam_bridge",
            namespace=namespace,
            condition=IfCondition(enable_combatos_bridge),
            parameters=[{
                "orch_ws": orch_ws,
                "pose_topic": "/slam/pose",
                "odom_topic": "/slam/odometry",
                "path_topic": "/slam/path",
                "status_topic": "/slam/status",
                "camera_topic": f"{prefix}/left/image_rect",
                "right_camera_topic": f"{prefix}/right/image_rect",
                "annotated_topic": "/slam/tracked_image",
                "enable_camera": True,
                "enable_right_camera": enable_right_camera,
                "enable_annotated": True,
                "video_fps": video_fps,
                "jpeg_quality": jpeg_quality,
            }],
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("name",          default_value="oak"),
        DeclareLaunchArgument("namespace",     default_value=""),
        DeclareLaunchArgument("resolution",    default_value="400p"),
        DeclareLaunchArgument("fps",           default_value="15.0"),
        DeclareLaunchArgument("settings_name", default_value="OAK_D_Lite"),
        DeclareLaunchArgument("enable_combatos_bridge", default_value="true"),
        DeclareLaunchArgument("orch_ws", default_value="ws://localhost:8000"),
        DeclareLaunchArgument("video_fps", default_value="8.0"),
        DeclareLaunchArgument("jpeg_quality", default_value="70"),
        DeclareLaunchArgument("enable_right_camera", default_value="false"),
        OpaqueFunction(function=launch_setup),
    ])
