/*

A bare-bones example node demonstrating the use of the Monocular mode in ORB-SLAM3

Author: Azmyin Md. Kamal
Date: 01/01/24

REQUIREMENTS
* Package paths are automatically discovered using ament_index

*/

//* Includes
#include "ros2_orb_slam3/common.hpp"

//* Constructor
MonocularMode::MonocularMode() :Node("mono_node_cpp")
{
    // Declare parameters to be passsed from command line
    // https://roboticsbackend.com/rclcpp-params-tutorial-get-set-ros2-params-with-cpp/
    
    //* Find package share directory dynamically
    try {
        packageShareDir = ament_index_cpp::get_package_share_directory("ros2_orb_slam3");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to find package share directory: %s", e.what());
        rclcpp::shutdown();
    }

    // std::cout<<"VLSAM NODE STARTED\n\n";
    RCLCPP_INFO(this->get_logger(), "\nORB-SLAM3-V1 NODE STARTED");
    RCLCPP_INFO(this->get_logger(), "Package share directory: %s", packageShareDir.c_str());

    this->declare_parameter("node_name_arg", "not_given"); // Name of this agent
    this->declare_parameter("voc_file_arg", "file_not_set"); // Needs to be overriden with appropriate name
    this->declare_parameter("settings_file_path_arg", "file_path_not_set"); // path to settings file
    this->declare_parameter("image_topic", ""); // Direct camera topic (e.g., /oak/rgb/image_rect)
    this->declare_parameter("settings_name", ""); // Settings name for direct mode (e.g., OAK)

    //* Watchdog, populate default values
    nodeName = "not_set";
    vocFilePath = "file_not_set";
    settingsFilePath = "file_not_set";

    //* Populate parameter values
    rclcpp::Parameter param1 = this->get_parameter("node_name_arg");
    nodeName = param1.as_string();

    rclcpp::Parameter param2 = this->get_parameter("voc_file_arg");
    vocFilePath = param2.as_string();

    rclcpp::Parameter param3 = this->get_parameter("settings_file_path_arg");
    settingsFilePath = param3.as_string();

    // Get direct subscription parameters
    directImageTopic = this->get_parameter("image_topic").as_string();
    directSettingsName = this->get_parameter("settings_name").as_string();

    // rclcpp::Parameter param4 = this->get_parameter("settings_file_name_arg");
    
  
    //* Set default paths using package share directory
    if (vocFilePath == "file_not_set" || settingsFilePath == "file_not_set")
    {
        pass;
        vocFilePath = packageShareDir + "/orb_slam3/Vocabulary/ORBvoc.txt.bin";
        settingsFilePath = packageShareDir + "/orb_slam3/config/Monocular/";
    }

    // std::cout<<"vocFilePath: "<<vocFilePath<<std::endl;
    // std::cout<<"settingsFilePath: "<<settingsFilePath<<std::endl;
    
    
    //* DEBUG print
    RCLCPP_INFO(this->get_logger(), "nodeName %s", nodeName.c_str());
    RCLCPP_INFO(this->get_logger(), "voc_file %s", vocFilePath.c_str());
    // RCLCPP_INFO(this->get_logger(), "settings_file_path %s", settingsFilePath.c_str());
    
    subexperimentconfigName = "/mono_py_driver/experiment_settings"; // topic that sends out some configuration parameters to the cpp ndoe
    pubconfigackName = "/mono_py_driver/exp_settings_ack"; // send an acknowledgement to the python node
    subImgMsgName = "/mono_py_driver/img_msg"; // topic to receive RGB image messages
    subTimestepMsgName = "/mono_py_driver/timestep_msg"; // topic to receive RGB image messages

    //* subscribe to python node to receive settings
    expConfig_subscription_ = this->create_subscription<std_msgs::msg::String>(subexperimentconfigName, 1, std::bind(&MonocularMode::experimentSetting_callback, this, _1));

    //* publisher to send out acknowledgement
    configAck_publisher_ = this->create_publisher<std_msgs::msg::String>(pubconfigackName, 10);

    //* SLAM output publishers
    posePub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slam/pose", 10);
    odomPub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slam/odometry", 10);
    pathPub_ = this->create_publisher<nav_msgs::msg::Path>("/slam/path", 10);

    //* subscrbite to the image messages coming from the Python driver node
    subImgMsg_subscription_= this->create_subscription<sensor_msgs::msg::Image>(subImgMsgName, 1, std::bind(&MonocularMode::Img_callback, this, _1));

    //* subscribe to receive the timestep
    subTimestepMsg_subscription_= this->create_subscription<std_msgs::msg::Float64>(subTimestepMsgName, 1, std::bind(&MonocularMode::Timestep_callback, this, _1));

    //* Check if direct camera subscription mode should be used
    if (!directImageTopic.empty() && !directSettingsName.empty())
    {
        useDirectSubscription = true;
        RCLCPP_INFO(this->get_logger(), "Direct subscription mode enabled");
        RCLCPP_INFO(this->get_logger(), "Image topic: %s", directImageTopic.c_str());
        RCLCPP_INFO(this->get_logger(), "Settings name: %s", directSettingsName.c_str());

        // Initialize VSLAM immediately with the provided settings
        initializeVSLAM(directSettingsName);

        // Create direct image subscription with sensor QoS (RELIABLE to match camera publisher)
        auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
        directImgMsg_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            directImageTopic, sensor_qos,
            std::bind(&MonocularMode::DirectImg_callback, this, _1));

        RCLCPP_INFO(this->get_logger(), "Subscribed to %s - Ready for SLAM!", directImageTopic.c_str());
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Waiting to finish handshake ......");
    }
    
}

//* Destructor
MonocularMode::~MonocularMode()
{   
    
    // Stop all threads
    // Call method to write the trajectory file
    // Release resources and cleanly shutdown
    pAgent->Shutdown();
    pass;

}

//* Callback which accepts experiment parameters from the Python node
void MonocularMode::experimentSetting_callback(const std_msgs::msg::String& msg){
    
    // std::cout<<"experimentSetting_callback"<<std::endl;
    bSettingsFromPython = true;
    experimentConfig = msg.data.c_str();
    // receivedConfig = experimentConfig; // Redundant
    
    RCLCPP_INFO(this->get_logger(), "Configuration YAML file name: %s", this->receivedConfig.c_str());

    //* Publish acknowledgement
    auto message = std_msgs::msg::String();
    message.data = "ACK";
    
    std::cout<<"Sent response: "<<message.data.c_str()<<std::endl;
    configAck_publisher_->publish(message);

    //* Wait to complete VSLAM initialization
    initializeVSLAM(experimentConfig);

}

//* Method to bind an initialized VSLAM framework to this node
void MonocularMode::initializeVSLAM(std::string& configString){
    
    // Watchdog, if the paths to vocabular and settings files are still not set
    if (vocFilePath == "file_not_set" || settingsFilePath == "file_not_set")
    {
        RCLCPP_ERROR(get_logger(), "Please provide valid voc_file and settings_file paths");       
        rclcpp::shutdown();
    } 
    
    //* Build .yaml`s file path
    
    settingsFilePath = settingsFilePath.append(configString);
    settingsFilePath = settingsFilePath.append(".yaml"); // Example ros2_ws/src/orb_slam3_ros2/orb_slam3/config/Monocular/TUM2.yaml

    RCLCPP_INFO(this->get_logger(), "Path to settings file: %s", settingsFilePath.c_str());
    
    // NOTE if you plan on passing other configuration parameters to ORB SLAM3 Systems class, do it here
    // NOTE you may also use a .yaml file here to set these values
    sensorType = ORB_SLAM3::System::MONOCULAR; 
    enablePangolinWindow = true; // Shows Pangolin window output
    enableOpenCVWindow = true; // Shows OpenCV window output
    
    pAgent = new ORB_SLAM3::System(vocFilePath, settingsFilePath, sensorType, enablePangolinWindow);
    std::cout << "MonocularMode node initialized" << std::endl; // TODO needs a better message
}

//* Callback that processes timestep sent over ROS
void MonocularMode::Timestep_callback(const std_msgs::msg::Float64& time_msg){
    // timeStep = 0; // Initialize
    timeStep = time_msg.data;
}

//* Callback to process image message and run SLAM node
void MonocularMode::Img_callback(const sensor_msgs::msg::Image& msg)
{
    // Initialize
    cv_bridge::CvImagePtr cv_ptr; //* Does not create a copy, memory efficient
    
    //* Convert ROS image to openCV image
    try
    {
        //cv::Mat im =  cv_bridge::toCvShare(msg.img, msg)->image;
        cv_ptr = cv_bridge::toCvCopy(msg); // Local scope
        
        // DEBUGGING, Show image
        // Update GUI Window
        // cv::imshow("test_window", cv_ptr->image);
        // cv::waitKey(3);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(),"Error reading image");
        return;
    }
    
    // std::cout<<std::fixed<<"Timestep: "<<timeStep<<std::endl; // Debug
    
    //* Perform all ORB-SLAM3 operations in Monocular mode
    //! Pose with respect to the camera coordinate frame not the world coordinate frame
    Sophus::SE3f Tcw = pAgent->TrackMonocular(cv_ptr->image, timeStep);

    publishSLAMOutput(Tcw, this->get_clock()->now());
}

//* Callback for direct camera subscription - extracts timestamp from image header
void MonocularMode::DirectImg_callback(const sensor_msgs::msg::Image& msg)
{
    // Initialize
    cv_bridge::CvImagePtr cv_ptr;

    // Debug: Log that callback is triggered
    static int frame_count = 0;
    frame_count++;
    if (frame_count % 30 == 1) {  // Log every 30th frame to avoid spam
        RCLCPP_INFO(this->get_logger(), "Received image %d - size: %dx%d, encoding: %s",
                    frame_count, msg.width, msg.height, msg.encoding.c_str());
    }

    //* Convert ROS image to openCV image
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(),"Error reading image: %s", e.what());
        return;
    }

    //* Extract timestamp from image header (convert to seconds as double)
    double timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;

    if (frame_count % 30 == 1) {
        RCLCPP_INFO(this->get_logger(), "Processing frame with timestamp: %.6f", timestamp);
    }

    //* Perform all ORB-SLAM3 operations in Monocular mode
    Sophus::SE3f Tcw = pAgent->TrackMonocular(cv_ptr->image, timestamp);

    publishSLAMOutput(Tcw, msg.header.stamp);
}

//* Publish pose, odometry, and path from SLAM output
void MonocularMode::publishSLAMOutput(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp)
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
