cmake_minimum_required(VERSION 3.8)
project(jetank_perception)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

add_executable(stereo_camera_node 
  src/stereo_camera_node.cpp
  src/camera_interface.cpp
  src/stereo_processing_strategy.cpp)

ament_target_dependencies(stereo_camera_node 
  rclcpp
  sensor_msgs
  cv_bridge
  camera_info_manager
  pcl_conversions
  OpenCV)

target_include_directories(stereo_camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GST_INCLUDE_DIRS})

target_link_libraries(stereo_camera_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${GST_LIBRARIES})

install(TARGETS stereo_camera_node
  DESTINATION lib/${PROJECT_NAME})

# install(DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}/)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/)
