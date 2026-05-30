/*
 * Stereo-Inertial ORB-SLAM3 ROS2 Node - Entry Point
 *
 * Author: Vikram
 * Date: 02/2026
 * Compatible for ROS2 Humble
 */

#include "ros2_orb_slam3/stereo_inertial.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<StereoInertialMode>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
