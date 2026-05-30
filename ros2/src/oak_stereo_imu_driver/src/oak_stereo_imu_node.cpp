#include <memory>

#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/datatype/IMUData.hpp"
#include "depthai/pipeline/datatype/ImgFrame.hpp"
#include "depthai/pipeline/node/IMU.hpp"
#include "depthai/pipeline/node/MonoCamera.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

class OakStereoImuNode : public rclcpp::Node {
   public:
    OakStereoImuNode() : Node("oak_stereo_imu_node") {
        // Declare parameters
        this->declare_parameter<double>("camera_fps", 30.0);
        this->declare_parameter<int>("camera_resolution", 400);
        double camera_fps = this->get_parameter("camera_fps").as_double();
        int camera_resolution = this->get_parameter("camera_resolution").as_int();

        // Publishers
        left_raw_pub_ = create_publisher<sensor_msgs::msg::Image>("/oak/left/image_raw", 10);
        right_raw_pub_ = create_publisher<sensor_msgs::msg::Image>("/oak/right/image_raw", 10);
        imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("/oak/imu/data", 10);

        // Build pipeline
        dai::Pipeline pipeline;
        pipeline.setXLinkChunkSize(0);  // Disable chunking — critical for PoE performance

        auto res = (camera_resolution == 400)
            ? dai::MonoCameraProperties::SensorResolution::THE_400_P
            : dai::MonoCameraProperties::SensorResolution::THE_800_P;

        auto left_cam = pipeline.create<dai::node::MonoCamera>();
        auto right_cam = pipeline.create<dai::node::MonoCamera>();
        left_cam->setBoardSocket(dai::CameraBoardSocket::CAM_B);
        right_cam->setBoardSocket(dai::CameraBoardSocket::CAM_C);
        left_cam->setResolution(res);
        right_cam->setResolution(res);
        left_cam->setFps(camera_fps);
        right_cam->setFps(camera_fps);

        auto xout_left_raw = pipeline.create<dai::node::XLinkOut>();
        auto xout_right_raw = pipeline.create<dai::node::XLinkOut>();
        xout_left_raw->setStreamName("left_raw");
        xout_right_raw->setStreamName("right_raw");
        left_cam->out.link(xout_left_raw->input);
        right_cam->out.link(xout_right_raw->input);

        // IMU: 400 Hz
        auto imu = pipeline.create<dai::node::IMU>();
        imu->enableIMUSensor(
            {dai::IMUSensor::ACCELEROMETER_RAW, dai::IMUSensor::GYROSCOPE_RAW}, 400);
        imu->setBatchReportThreshold(1);
        imu->setMaxBatchReports(10);

        auto xout_imu = pipeline.create<dai::node::XLinkOut>();
        xout_imu->setStreamName("imu");
        imu->out.link(xout_imu->input);

        // Open device and start pipeline
        device_ = std::make_shared<dai::Device>(pipeline);
        RCLCPP_INFO(get_logger(), "OAK device opened: %s", device_->getMxId().c_str());

        // Output queues
        left_raw_q_ = device_->getOutputQueue("left_raw", 4, false);
        right_raw_q_ = device_->getOutputQueue("right_raw", 4, false);
        imu_q_ = device_->getOutputQueue("imu", 50, false);

        // Register callbacks
        left_raw_q_->addCallback([this](const std::string&, std::shared_ptr<dai::ADatatype> data) {
            publishImage(left_raw_pub_, data, "oak_left_optical_frame");
        });
        right_raw_q_->addCallback([this](const std::string&, std::shared_ptr<dai::ADatatype> data) {
            publishImage(right_raw_pub_, data, "oak_right_optical_frame");
        });
        imu_q_->addCallback([this](const std::string&, std::shared_ptr<dai::ADatatype> data) {
            publishImu(data);
        });

        RCLCPP_INFO(get_logger(), "Streaming: stereo %dp @ %.1f FPS, IMU @ 400 Hz",
                    camera_resolution, camera_fps);
    }

    ~OakStereoImuNode() {
        if (left_raw_q_) left_raw_q_->close();
        if (right_raw_q_) right_raw_q_->close();
        if (imu_q_) imu_q_->close();
        if (device_) device_->close();
    }

   private:
    void publishImage(rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& pub,
                      const std::shared_ptr<dai::ADatatype>& data,
                      const std::string& frame_id) {
        auto frame = std::dynamic_pointer_cast<dai::ImgFrame>(data);
        if (!frame) return;

        sensor_msgs::msg::Image msg;
        auto ts = frame->getTimestampDevice();
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count();
        msg.header.stamp.sec = static_cast<int32_t>(ns / 1'000'000'000);
        msg.header.stamp.nanosec = static_cast<uint32_t>(ns % 1'000'000'000);
        msg.header.frame_id = frame_id;
        msg.height = frame->getHeight();
        msg.width = frame->getWidth();
        msg.encoding = "mono8";
        msg.step = frame->getWidth();
        msg.data = frame->getData();
        pub->publish(msg);
    }

    void publishImu(const std::shared_ptr<dai::ADatatype>& data) {
        auto imu_data = std::dynamic_pointer_cast<dai::IMUData>(data);
        if (!imu_data) return;

        for (const auto& packet : imu_data->packets) {
            sensor_msgs::msg::Imu msg;
            auto ts = packet.acceleroMeter.getTimestampDevice();
            int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count();
            msg.header.stamp.sec = static_cast<int32_t>(ns / 1'000'000'000);
            msg.header.stamp.nanosec = static_cast<uint32_t>(ns % 1'000'000'000);
            msg.header.frame_id = "oak_imu_frame";

            msg.linear_acceleration.x = packet.acceleroMeter.x;
            msg.linear_acceleration.y = packet.acceleroMeter.y;
            msg.linear_acceleration.z = packet.acceleroMeter.z;
            msg.angular_velocity.x = packet.gyroscope.x;
            msg.angular_velocity.y = packet.gyroscope.y;
            msg.angular_velocity.z = packet.gyroscope.z;

            msg.orientation_covariance[0] = -1.0;
            msg.linear_acceleration_covariance[0] = -1.0;
            msg.angular_velocity_covariance[0] = -1.0;

            imu_pub_->publish(msg);
        }
    }

    std::shared_ptr<dai::Device> device_;
    std::shared_ptr<dai::DataOutputQueue> left_raw_q_, right_raw_q_, imu_q_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_raw_pub_, right_raw_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OakStereoImuNode>());
    rclcpp::shutdown();
    return 0;
}
