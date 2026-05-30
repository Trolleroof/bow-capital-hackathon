cmake_minimum_required(VERSION 3.8)
project(oak_stereo_imu_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(depthai_ros_driver REQUIRED)
find_package(pluginlib REQUIRED)
find_package(depthai_bridge REQUIRED)

find_package(depthai REQUIRED)
find_package(OpenCV REQUIRED)

# Plugin shared library (imu node + pipeline plugin)
add_library(${PROJECT_NAME} SHARED
  src/imu_node_200hz.cpp
  src/oak_pipeline.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  depthai_ros_driver
  pluginlib
  depthai_bridge
  depthai
)

pluginlib_export_plugin_description_file(depthai_ros_driver plugins.xml)

# Standalone node
add_executable(oak_stereo_imu_node src/oak_stereo_imu_node.cpp)
ament_target_dependencies(oak_stereo_imu_node rclcpp sensor_msgs depthai_ros_driver)
target_include_directories(oak_stereo_imu_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(oak_stereo_imu_node ${OpenCV_LIBS})

install(TARGETS ${PROJECT_NAME} oak_stereo_imu_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/check_imu_timing.py
  scripts/check_image_timing.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
