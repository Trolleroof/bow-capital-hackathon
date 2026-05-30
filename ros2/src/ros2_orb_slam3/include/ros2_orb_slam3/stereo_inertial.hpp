#ifndef STEREO_INERTIAL_HPP
#define STEREO_INERTIAL_HPP

// C++ includes
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <sstream>

// ROS2 includes
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <std_msgs/msg/header.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"

// message_filters for stereo synchronization
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// Include Eigen
#include <Eigen/Dense>

// Include cv-bridge
#include <cv_bridge/cv_bridge.h>

// Include OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

// ORB-SLAM3 includes
#include "System.h"
#include "ImuTypes.h"

#define pass (void)0

class StereoInertialMode : public rclcpp::Node
{
    public:
        StereoInertialMode();
        ~StereoInertialMode();

    private:
        // Package and config paths
        std::string packageShareDir = "";
        std::string nodeName = "";
        std::string vocFilePath = "";
        std::string settingsFilePath = "";

        // Configurable topic parameters
        std::string leftImageTopic = "";
        std::string rightImageTopic = "";
        std::string imuTopic = "";
        std::string settingsName = "";

        // Stereo sync types
        using Image = sensor_msgs::msg::Image;
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;
        using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

        // Stereo image subscribers (message_filters)
        std::shared_ptr<message_filters::Subscriber<Image>> leftImgSub_;
        std::shared_ptr<message_filters::Subscriber<Image>> rightImgSub_;
        std::shared_ptr<Synchronizer> syncStereo_;

        // IMU subscriber (runs in its own callback group for concurrent execution)
        rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imuSub_;

        // SLAM output publishers
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr posePub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pathPub_;
        std::vector<geometry_msgs::msg::PoseStamped> poseHistory_;

        // IMU buffer (thread-safe)
        std::vector<ORB_SLAM3::IMU::Point> imuBuffer_;
        std::mutex imuMutex_;
        double lastImageTimestamp_ = -1.0;
        bool imuCoversTimestamp(double t_image);

        // ORB-SLAM3
        ORB_SLAM3::System* pAgent;
        ORB_SLAM3::System::eSensor sensorType;
        bool enablePangolinWindow = false;
        bool enableOpenCVWindow = false;

        // Callbacks
        void StereoImg_callback(const Image::ConstSharedPtr& leftMsg,
                                const Image::ConstSharedPtr& rightMsg);
        void Imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

        // Helpers
        std::vector<ORB_SLAM3::IMU::Point> getImuMeasurements(double t_image);
        void initializeVSLAM(std::string& configString);
        void publishSLAMOutput(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp);
};

#endif
