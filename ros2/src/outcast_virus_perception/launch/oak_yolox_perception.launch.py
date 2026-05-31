#!/usr/bin/env python3
"""Launch the Outcast Virus detector node against an existing OAK-D RGB stream."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("image_topic", default_value="/oak/rgb/image_rect"),
        DeclareLaunchArgument("detections_topic", default_value="/perception/detections"),
        DeclareLaunchArgument("annotated_topic", default_value="/perception/annotated_image"),
        DeclareLaunchArgument("model_type", default_value="ultralytics"),
        DeclareLaunchArgument("model_path", default_value="perception/yolo11n.pt"),
        DeclareLaunchArgument("yolox_exp_file", default_value=""),
        DeclareLaunchArgument("device", default_value="0"),
        DeclareLaunchArgument("confidence", default_value="0.40"),
        DeclareLaunchArgument("iou", default_value="0.45"),
        DeclareLaunchArgument("fp16", default_value="true"),
        DeclareLaunchArgument("max_fps", default_value="15.0"),
        Node(
            package="outcast_virus_perception",
            executable="yolox_node",
            name="outcast_virus_yolox",
            output="screen",
            parameters=[{
                "image_topic": LaunchConfiguration("image_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "annotated_topic": LaunchConfiguration("annotated_topic"),
                "model_type": LaunchConfiguration("model_type"),
                "model_path": LaunchConfiguration("model_path"),
                "yolox_exp_file": LaunchConfiguration("yolox_exp_file"),
                "device": LaunchConfiguration("device"),
                "confidence": LaunchConfiguration("confidence"),
                "iou": LaunchConfiguration("iou"),
                "fp16": LaunchConfiguration("fp16"),
                "max_fps": LaunchConfiguration("max_fps"),
            }],
        ),
    ])
