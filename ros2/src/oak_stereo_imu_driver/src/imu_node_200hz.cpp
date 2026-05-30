#include "oak_stereo_imu_driver/imu_node_200hz.hpp"

#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/datatype/IMUData.hpp"
#include "depthai/device/Device.hpp"
#include "rclcpp/rclcpp.hpp"

namespace oak_stereo_imu_driver {

ImuNode200Hz::ImuNode200Hz(const std::string& daiNodeName,
                           std::shared_ptr<rclcpp::Node> node,
                           std::shared_ptr<dai::Pipeline> pipeline)
    : BaseNode(daiNodeName, node, pipeline) {

    // Create the depthai IMU node
    imuNode_ = pipeline->create<dai::node::IMU>();

    // Luxonis recommendation: 200 Hz RAW accel + gyro
    imuNode_->enableIMUSensor(
        {dai::IMUSensor::ACCELEROMETER_RAW, dai::IMUSensor::GYROSCOPE_RAW},
        200);
    imuNode_->setBatchReportThreshold(1);
    imuNode_->setMaxBatchReports(10);

    setNames();
    setXinXout(pipeline);

    imuPub_ = node->create_publisher<sensor_msgs::msg::Imu>("~/" + getName() + "/data", 10);

    RCLCPP_INFO(getLogger(), "ImuNode200Hz: configured at 200 Hz (batch threshold=1, max=10)");
}

ImuNode200Hz::~ImuNode200Hz() {
    closeQueues();
}

void ImuNode200Hz::setNames() {
    qName_ = getName() + "_imu";
}

void ImuNode200Hz::setXinXout(std::shared_ptr<dai::Pipeline> pipeline) {
    xout_ = setupXout(pipeline, qName_);
    imuNode_->out.link(xout_->input);
}

void ImuNode200Hz::setupQueues(std::shared_ptr<dai::Device> device) {
    imuQ_ = device->getOutputQueue(qName_, 50, false);
    imuQ_->addCallback(std::bind(&ImuNode200Hz::imuCallback, this,
                                 std::placeholders::_1, std::placeholders::_2));
}

void ImuNode200Hz::closeQueues() {
    if (imuQ_) {
        imuQ_->close();
    }
}

void ImuNode200Hz::updateParams(const std::vector<rclcpp::Parameter>& /*params*/) {}

void ImuNode200Hz::imuCallback(const std::string& /*name*/,
                                const std::shared_ptr<dai::ADatatype>& data) {
    auto imuData = std::dynamic_pointer_cast<dai::IMUData>(data);
    if (!imuData) return;

    auto rosNode = getROSNode();

    for (const auto& packet : imuData->packets) {
        sensor_msgs::msg::Imu msg;

        // Use accelerometer timestamp as the message stamp
        auto ts = packet.acceleroMeter.getTimestampDevice();
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count();
        msg.header.stamp.sec = static_cast<int32_t>(ns / 1'000'000'000);
        msg.header.stamp.nanosec = static_cast<uint32_t>(ns % 1'000'000'000);
        msg.header.frame_id = getName() + "_frame";

        msg.linear_acceleration.x = packet.acceleroMeter.x;
        msg.linear_acceleration.y = packet.acceleroMeter.y;
        msg.linear_acceleration.z = packet.acceleroMeter.z;

        msg.angular_velocity.x = packet.gyroscope.x;
        msg.angular_velocity.y = packet.gyroscope.y;
        msg.angular_velocity.z = packet.gyroscope.z;

        // Covariance unknown — fill with -1 sentinel
        msg.orientation_covariance[0] = -1.0;
        msg.linear_acceleration_covariance[0] = -1.0;
        msg.angular_velocity_covariance[0] = -1.0;

        imuPub_->publish(msg);
    }
}

}  // namespace oak_stereo_imu_driver
