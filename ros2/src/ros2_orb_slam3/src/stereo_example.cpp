/*
 * Stereo ORB-SLAM3 ROS2 Node - Entry Point
 *
 * Author: Vikram
 * Date: 02/2026
 * Compatible for ROS2 Humble
 */

#include "ros2_orb_slam3/stereo.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<StereoMode>();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
