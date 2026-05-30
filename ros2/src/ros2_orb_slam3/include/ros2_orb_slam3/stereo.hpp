#ifndef STEREO_HPP
#define STEREO_HPP

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
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include <image_transport/image_transport.hpp>

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

#define pass (void)0

class StereoMode : public rclcpp::Node
{
    public:
        StereoMode();
        ~StereoMode();

    private:
        // Package and config paths
        std::string packageShareDir = "";
        std::string nodeName = "";
        std::string vocFilePath = "";
        std::string settingsFilePath = "";

        // Configurable topic parameters
        std::string leftImageTopic = "";
        std::string rightImageTopic = "";
        std::string settingsName = "";

        // Stereo sync types
        using Image = sensor_msgs::msg::Image;
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;
        using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

        // Stereo image subscribers (message_filters)
        std::shared_ptr<message_filters::Subscriber<Image>> leftImgSub_;
        std::shared_ptr<message_filters::Subscriber<Image>> rightImgSub_;
        std::shared_ptr<Synchronizer> syncStereo_;

        // ORB-SLAM3
        ORB_SLAM3::System* pAgent;
        ORB_SLAM3::System::eSensor sensorType;
        bool enablePangolinWindow = false;
        bool enableOpenCVWindow = false;

        // SLAM output publishers
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr posePub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pathPub_;
        image_transport::Publisher trackedImgPub_;
        std::vector<geometry_msgs::msg::PoseStamped> poseHistory_;

        // Callbacks
        void StereoImg_callback(const Image::ConstSharedPtr& leftMsg,
                                const Image::ConstSharedPtr& rightMsg);

        // Helpers
        void initializeVSLAM(std::string& configString);
        void publishSLAMOutput(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp);
};

#endif
