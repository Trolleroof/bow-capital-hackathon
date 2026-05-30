#include <memory>
#include <string>
#include <vector>

#include "depthai-shared/common/CameraBoardSocket.hpp"
#include "depthai_ros_driver/dai_nodes/base_node.hpp"
#include "depthai_ros_driver/dai_nodes/sensors/sensor_wrapper.hpp"
#include "depthai_ros_driver/pipeline/base_pipeline.hpp"
#include "oak_stereo_imu_driver/imu_node_200hz.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace oak_stereo_imu_driver {

class OakPipeline : public depthai_ros_driver::pipeline_gen::BasePipeline {
   public:
    std::vector<std::unique_ptr<depthai_ros_driver::dai_nodes::BaseNode>> createPipeline(
        std::shared_ptr<rclcpp::Node> node,
        std::shared_ptr<dai::Device> device,
        std::shared_ptr<dai::Pipeline> pipeline,
        const std::string& /*nnType*/) override {

        namespace dai_nodes = depthai_ros_driver::dai_nodes;
        std::vector<std::unique_ptr<dai_nodes::BaseNode>> nodes;

        // Stereo pair
        auto left = std::make_unique<dai_nodes::SensorWrapper>(
            "left", node, pipeline, device, dai::CameraBoardSocket::CAM_B);
        auto right = std::make_unique<dai_nodes::SensorWrapper>(
            "right", node, pipeline, device, dai::CameraBoardSocket::CAM_C);

        // IMU at 200 Hz — configured directly in code, no YAML param handler
        auto imu = std::make_unique<ImuNode200Hz>("imu", node, pipeline);

        nodes.push_back(std::move(left));
        nodes.push_back(std::move(right));
        nodes.push_back(std::move(imu));

        return nodes;
    }
};

}  // namespace oak_stereo_imu_driver

PLUGINLIB_EXPORT_CLASS(oak_stereo_imu_driver::OakPipeline,
                       depthai_ros_driver::pipeline_gen::BasePipeline)
