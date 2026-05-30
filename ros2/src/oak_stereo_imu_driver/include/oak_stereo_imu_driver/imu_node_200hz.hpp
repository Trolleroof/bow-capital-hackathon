#pragma once

#include <memory>
#include <string>

#include "depthai/pipeline/datatype/IMUData.hpp"
#include "depthai/pipeline/node/IMU.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"
#include "depthai_ros_driver/dai_nodes/base_node.hpp"
#include "rclcpp/publisher.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace dai {
class Pipeline;
class Device;
class DataOutputQueue;
}  // namespace dai

namespace rclcpp {
class Node;
class Parameter;
}  // namespace rclcpp

namespace oak_stereo_imu_driver {

class ImuNode200Hz : public depthai_ros_driver::dai_nodes::BaseNode {
   public:
    ImuNode200Hz(const std::string& daiNodeName,
                 std::shared_ptr<rclcpp::Node> node,
                 std::shared_ptr<dai::Pipeline> pipeline);
    ~ImuNode200Hz();

    void setNames() override;
    void setXinXout(std::shared_ptr<dai::Pipeline> pipeline) override;
    void setupQueues(std::shared_ptr<dai::Device> device) override;
    void closeQueues() override;
    void updateParams(const std::vector<rclcpp::Parameter>& params) override;

   private:
    void imuCallback(const std::string& name, const std::shared_ptr<dai::ADatatype>& data);

    std::shared_ptr<dai::node::IMU> imuNode_;
    std::shared_ptr<dai::node::XLinkOut> xout_;
    std::shared_ptr<dai::DataOutputQueue> imuQ_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imuPub_;
    std::string qName_;
};

}  // namespace oak_stereo_imu_driver
