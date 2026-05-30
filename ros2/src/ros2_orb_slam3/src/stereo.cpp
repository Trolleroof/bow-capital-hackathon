/*
 * Stereo ORB-SLAM3 ROS2 Node
 *
 * Subscribes to synchronized stereo image pairs and calls TrackStereo()
 * with STEREO sensor type (no IMU).
 *
 * Author: Vikram
 * Date: 02/2026
 */

#include "ros2_orb_slam3/stereo.hpp"

//* Constructor
StereoMode::StereoMode() : Node("stereo_node_cpp")
{
    // Find package share directory dynamically
    try {
        packageShareDir = ament_index_cpp::get_package_share_directory("ros2_orb_slam3");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to find package share directory: %s", e.what());
        rclcpp::shutdown();
        return;
    }

    RCLCPP_INFO(this->get_logger(), "\nORB-SLAM3 STEREO NODE STARTED");
    RCLCPP_INFO(this->get_logger(), "Package share directory: %s", packageShareDir.c_str());

    // Declare parameters
    this->declare_parameter("node_name_arg", "not_given");
    this->declare_parameter("voc_file_arg", "file_not_set");
    this->declare_parameter("settings_file_path_arg", "file_path_not_set");
    this->declare_parameter("left_image_topic", "/oak/left/image_rect");
    this->declare_parameter("right_image_topic", "/oak/right/image_rect");
    this->declare_parameter("settings_name", "OAK");

    // Populate parameter values
    nodeName = this->get_parameter("node_name_arg").as_string();
    vocFilePath = this->get_parameter("voc_file_arg").as_string();
    settingsFilePath = this->get_parameter("settings_file_path_arg").as_string();
    leftImageTopic = this->get_parameter("left_image_topic").as_string();
    rightImageTopic = this->get_parameter("right_image_topic").as_string();
    settingsName = this->get_parameter("settings_name").as_string();

    // Set default paths using package share directory
    if (vocFilePath == "file_not_set" || settingsFilePath == "file_path_not_set")
    {
        vocFilePath = packageShareDir + "/orb_slam3/Vocabulary/ORBvoc.txt.bin";
        settingsFilePath = packageShareDir + "/orb_slam3/config/Stereo/";
    }

    RCLCPP_INFO(this->get_logger(), "nodeName: %s", nodeName.c_str());
    RCLCPP_INFO(this->get_logger(), "voc_file: %s", vocFilePath.c_str());
    RCLCPP_INFO(this->get_logger(), "Left image topic: %s", leftImageTopic.c_str());
    RCLCPP_INFO(this->get_logger(), "Right image topic: %s", rightImageTopic.c_str());
    RCLCPP_INFO(this->get_logger(), "Settings name: %s", settingsName.c_str());

    // Initialize VSLAM immediately
    initializeVSLAM(settingsName);

    // Set up stereo image subscribers using message_filters
    auto image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();

    leftImgSub_ = std::make_shared<message_filters::Subscriber<Image>>(
        this, leftImageTopic, image_qos.get_rmw_qos_profile());
    rightImgSub_ = std::make_shared<message_filters::Subscriber<Image>>(
        this, rightImageTopic, image_qos.get_rmw_qos_profile());

    // ApproximateTime synchronizer with queue size 10
    syncStereo_ = std::make_shared<Synchronizer>(SyncPolicy(10), *leftImgSub_, *rightImgSub_);
    syncStereo_->registerCallback(std::bind(
        &StereoMode::StereoImg_callback, this,
        std::placeholders::_1, std::placeholders::_2));

    // SLAM output publishers
    posePub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slam/pose", 10);
    odomPub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slam/odometry", 10);
    pathPub_ = this->create_publisher<nav_msgs::msg::Path>("/slam/path", 10);
    trackedImgPub_ = image_transport::create_publisher(this, "/slam/tracked_image");

    RCLCPP_INFO(this->get_logger(), "Stereo node initialized - Ready for SLAM!");
}

//* Destructor
StereoMode::~StereoMode()
{
    pAgent->Shutdown();
    pass;
}

//* Initialize ORB-SLAM3 with STEREO sensor type
void StereoMode::initializeVSLAM(std::string& configString)
{
    if (vocFilePath == "file_not_set" || settingsFilePath == "file_path_not_set")
    {
        RCLCPP_ERROR(get_logger(), "Please provide valid voc_file and settings_file paths");
        rclcpp::shutdown();
        return;
    }

    // Build .yaml file path
    settingsFilePath = settingsFilePath + configString + ".yaml";
    RCLCPP_INFO(this->get_logger(), "Path to settings file: %s", settingsFilePath.c_str());

    sensorType = ORB_SLAM3::System::STEREO;
    enablePangolinWindow = true;
    enableOpenCVWindow = true;

    pAgent = new ORB_SLAM3::System(vocFilePath, settingsFilePath, sensorType, enablePangolinWindow);
    RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 STEREO system initialized");
}

//* Stereo image callback - runs SLAM tracking
void StereoMode::StereoImg_callback(
    const Image::ConstSharedPtr& leftMsg,
    const Image::ConstSharedPtr& rightMsg)
{
    cv_bridge::CvImagePtr cv_left, cv_right;

    try
    {
        cv_left = cv_bridge::toCvCopy(leftMsg);
        cv_right = cv_bridge::toCvCopy(rightMsg);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Error converting stereo images: %s", e.what());
        return;
    }

    // Extract timestamp from left image header
    double timestamp = leftMsg->header.stamp.sec + leftMsg->header.stamp.nanosec * 1e-9;

    static int frame_count = 0;
    frame_count++;
    if (frame_count % 30 == 1)
    {
        RCLCPP_INFO(this->get_logger(), "Frame %d - timestamp: %.6f",
                    frame_count, timestamp);
    }

    // Run ORB-SLAM3 stereo tracking (no IMU data)
    Sophus::SE3f Tcw = pAgent->TrackStereo(cv_left->image, cv_right->image, timestamp);

    // Publish annotated frame (keypoints, tracking state overlay)
    cv::Mat tracked = pAgent->GetCurrentFrame();
    if (!tracked.empty() && trackedImgPub_.getNumSubscribers() > 0)
    {
        std_msgs::msg::Header hdr;
        hdr.stamp = leftMsg->header.stamp;
        hdr.frame_id = leftMsg->header.frame_id;
        trackedImgPub_.publish(cv_bridge::CvImage(hdr, "bgr8", tracked).toImageMsg());
    }

    publishSLAMOutput(Tcw, leftMsg->header.stamp);
}

//* Publish pose, odometry, and path from SLAM output
void StereoMode::publishSLAMOutput(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp)
{
    // Only publish when tracking is active (OK=2, RECENTLY_LOST=3, OK_KLT=5)
    int state = pAgent->GetTrackingState();
    if (state != 2 && state != 3 && state != 5)
        return;

    // Tcw is world-to-camera; invert to get camera position in world frame
    Sophus::SE3f Twc = Tcw.inverse();
    auto t = Twc.translation();
    auto q = Twc.unit_quaternion();

    // PoseStamped
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "map";
    pose_msg.pose.position.x = t.x();
    pose_msg.pose.position.y = t.y();
    pose_msg.pose.position.z = t.z();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    posePub_->publish(pose_msg);

    // Odometry
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header = pose_msg.header;
    odom_msg.child_frame_id = "camera";
    odom_msg.pose.pose = pose_msg.pose;
    odomPub_->publish(odom_msg);

    // Path
    poseHistory_.push_back(pose_msg);
    nav_msgs::msg::Path path_msg;
    path_msg.header = pose_msg.header;
    path_msg.poses = poseHistory_;
    pathPub_->publish(path_msg);
}
