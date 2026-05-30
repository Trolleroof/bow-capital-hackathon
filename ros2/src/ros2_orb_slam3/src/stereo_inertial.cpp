/*
 * Stereo-Inertial ORB-SLAM3 ROS2 Node
 *
 * Subscribes to synchronized stereo image pairs and IMU data,
 * buffers IMU measurements between frames, and calls TrackStereo()
 * with IMU_STEREO sensor type.
 *
 * Author: Vikram
 * Date: 02/2026
 */

#include "ros2_orb_slam3/stereo_inertial.hpp"

//* Constructor
StereoInertialMode::StereoInertialMode() : Node("stereo_inertial_node_cpp")
{
    // Find package share directory dynamically
    try {
        packageShareDir = ament_index_cpp::get_package_share_directory("ros2_orb_slam3");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to find package share directory: %s", e.what());
        rclcpp::shutdown();
        return;
    }

    RCLCPP_INFO(this->get_logger(), "\nORB-SLAM3 STEREO-INERTIAL NODE STARTED");
    RCLCPP_INFO(this->get_logger(), "Package share directory: %s", packageShareDir.c_str());

    // Declare parameters
    this->declare_parameter("node_name_arg", "not_given");
    this->declare_parameter("voc_file_arg", "file_not_set");
    this->declare_parameter("settings_file_path_arg", "file_path_not_set");
    this->declare_parameter("left_image_topic", "/oak/left/image_rect");
    this->declare_parameter("right_image_topic", "/oak/right/image_rect");
    this->declare_parameter("imu_topic", "/oak/imu/data");
    this->declare_parameter("settings_name", "OAK");

    // Populate parameter values
    nodeName = this->get_parameter("node_name_arg").as_string();
    vocFilePath = this->get_parameter("voc_file_arg").as_string();
    settingsFilePath = this->get_parameter("settings_file_path_arg").as_string();
    leftImageTopic = this->get_parameter("left_image_topic").as_string();
    rightImageTopic = this->get_parameter("right_image_topic").as_string();
    imuTopic = this->get_parameter("imu_topic").as_string();
    settingsName = this->get_parameter("settings_name").as_string();

    // Set default paths using package share directory
    if (vocFilePath == "file_not_set" || settingsFilePath == "file_path_not_set")
    {
        vocFilePath = packageShareDir + "/orb_slam3/Vocabulary/ORBvoc.txt.bin";
        settingsFilePath = packageShareDir + "/orb_slam3/config/Stereo-Inertial/";
    }

    RCLCPP_INFO(this->get_logger(), "nodeName: %s", nodeName.c_str());
    RCLCPP_INFO(this->get_logger(), "voc_file: %s", vocFilePath.c_str());
    RCLCPP_INFO(this->get_logger(), "Left image topic: %s", leftImageTopic.c_str());
    RCLCPP_INFO(this->get_logger(), "Right image topic: %s", rightImageTopic.c_str());
    RCLCPP_INFO(this->get_logger(), "IMU topic: %s", imuTopic.c_str());
    RCLCPP_INFO(this->get_logger(), "Settings name: %s", settingsName.c_str());

    // Initialize VSLAM immediately (no handshake needed for stereo-inertial)
    initializeVSLAM(settingsName);

    // Set up stereo image subscribers using message_filters
    auto image_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();

    leftImgSub_ = std::make_shared<message_filters::Subscriber<Image>>(
        this, leftImageTopic, image_qos.get_rmw_qos_profile());
    rightImgSub_ = std::make_shared<message_filters::Subscriber<Image>>(
        this, rightImageTopic, image_qos.get_rmw_qos_profile());

    // ApproximateTime synchronizer with queue size 10
    syncStereo_ = std::make_shared<Synchronizer>(SyncPolicy(20), *leftImgSub_, *rightImgSub_);
    syncStereo_->registerCallback(std::bind(
        &StereoInertialMode::StereoImg_callback, this,
        std::placeholders::_1, std::placeholders::_2));

    // IMU subscriber - dedicated callback group so it runs concurrently with SLAM tracking
    imu_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(2000)).reliable();
    rclcpp::SubscriptionOptions imu_opts;
    imu_opts.callback_group = imu_callback_group_;
    imuSub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imuTopic, imu_qos,
        std::bind(&StereoInertialMode::Imu_callback, this, std::placeholders::_1),
        imu_opts);

    // SLAM output publishers
    posePub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slam/pose", 10);
    odomPub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slam/odometry", 10);
    pathPub_ = this->create_publisher<nav_msgs::msg::Path>("/slam/path", 10);

    RCLCPP_INFO(this->get_logger(), "Stereo-Inertial node initialized - Ready for SLAM!");
}

//* Destructor
StereoInertialMode::~StereoInertialMode()
{
    pAgent->Shutdown();
}

//* Initialize ORB-SLAM3 with IMU_STEREO sensor type
void StereoInertialMode::initializeVSLAM(std::string& configString)
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

    sensorType = ORB_SLAM3::System::IMU_STEREO;
    enablePangolinWindow = true;
    enableOpenCVWindow = true;

    pAgent = new ORB_SLAM3::System(vocFilePath, settingsFilePath, sensorType, enablePangolinWindow);
    RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 IMU_STEREO system initialized");
}

//* IMU callback - buffers IMU measurements
void StereoInertialMode::Imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    // Extract accelerometer and gyroscope data
    float acc_x = msg->linear_acceleration.x;
    float acc_y = msg->linear_acceleration.y;
    float acc_z = msg->linear_acceleration.z;
    float gyro_x = msg->angular_velocity.x;
    float gyro_y = msg->angular_velocity.y;
    float gyro_z = msg->angular_velocity.z;

    ORB_SLAM3::IMU::Point imuPoint(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, t);

    std::lock_guard<std::mutex> lock(imuMutex_);

    // Only discard entries older than the last tracked frame — they'll never be needed again
    if (imuBuffer_.size() > 2000 && lastImageTimestamp_ > 0)
    {
        auto it = std::find_if(imuBuffer_.begin(), imuBuffer_.end(),
            [this](const ORB_SLAM3::IMU::Point& p) { return p.t > lastImageTimestamp_; });
        if (it != imuBuffer_.begin())
            imuBuffer_.erase(imuBuffer_.begin(), it);
    }

    imuBuffer_.push_back(imuPoint);
}

//* Check if the IMU buffer has data covering up to the image timestamp
bool StereoInertialMode::imuCoversTimestamp(double t_image)
{
    std::lock_guard<std::mutex> lock(imuMutex_);
    return !imuBuffer_.empty() && imuBuffer_.back().t >= t_image;
}

//* Extract IMU measurements between last image and current image timestamps
std::vector<ORB_SLAM3::IMU::Point> StereoInertialMode::getImuMeasurements(double t_image)
{
    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    std::lock_guard<std::mutex> lock(imuMutex_);

    if (imuBuffer_.empty())
    {
        return vImuMeas;
    }

    // Find the range of IMU measurements: lastImageTimestamp_ < t <= t_image
    auto it_start = imuBuffer_.begin();
    auto it_end = imuBuffer_.begin();

    for (auto it = imuBuffer_.begin(); it != imuBuffer_.end(); ++it)
    {
        if (it->t <= lastImageTimestamp_)
        {
            it_start = it + 1;
            continue;
        }
        if (it->t <= t_image)
        {
            vImuMeas.push_back(*it);
            it_end = it + 1;
        }
        else
        {
            break;
        }
    }

    // Erase consumed entries
    if (it_end != imuBuffer_.begin())
    {
        imuBuffer_.erase(imuBuffer_.begin(), it_end);
    }

    return vImuMeas;
}

//* Stereo image callback - runs SLAM tracking
void StereoInertialMode::StereoImg_callback(
    const Image::ConstSharedPtr& leftMsg,
    const Image::ConstSharedPtr& rightMsg)
{
    // Extract timestamp from left image header
    double timestamp = leftMsg->header.stamp.sec + leftMsg->header.stamp.nanosec * 1e-9;

    // Like the official impl: ensure IMU buffer has data at least as recent as the image.
    // This prevents feeding partial IMU data to ORB-SLAM3's preintegration.
    // TEMPORARILY DISABLED - remove comment below to re-enable
    // if (!imuCoversTimestamp(timestamp))
    // {
    //     static int skip_count = 0;
    //     if (skip_count++ % 30 == 0)
    //         RCLCPP_WARN(this->get_logger(),
    //             "Skipping frame - IMU data hasn't caught up to image timestamp %.6f", timestamp);
    //     return;
    // }

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

    // Get buffered IMU measurements between frames (safe — we know IMU covers this range)
    std::vector<ORB_SLAM3::IMU::Point> vImuMeas = getImuMeasurements(timestamp);

    static int frame_count = 0;
    frame_count++;
    if (frame_count % 30 == 1)
    {
        RCLCPP_INFO(this->get_logger(), "Frame %d - timestamp: %.6f, IMU measurements: %zu",
                    frame_count, timestamp, vImuMeas.size());
    }

    if (vImuMeas.empty())
    {
        RCLCPP_WARN(this->get_logger(), "Skipping frame %d - no IMU measurements in range", frame_count);
        return;
    }

    // Run ORB-SLAM3 stereo tracking with IMU
    Sophus::SE3f Tcw = pAgent->TrackStereo(cv_left->image, cv_right->image, timestamp, vImuMeas);

    // Update last image timestamp
    lastImageTimestamp_ = timestamp;

    publishSLAMOutput(Tcw, leftMsg->header.stamp);
}

//* Publish pose, odometry, and path from SLAM output
void StereoInertialMode::publishSLAMOutput(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp)
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
