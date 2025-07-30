cmake_minimum_required(VERSION 3.8)
project(jetank_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

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

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(stereo_camera_node 
  src/stereo_camera_node.cpp
  src/camera_interface.cpp
  src/stereo_processing_strategy.cpp)

add_executable(camera_node 
  src/single_camera.cpp
  src/camera_interface.cpp)

ament_target_dependencies(stereo_camera_node 
  rclcpp
  sensor_msgs
  cv_bridge
  camera_info_manager
  pcl_conversions
  OpenCV)

ament_target_dependencies(camera_node 
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

target_include_directories(camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GST_INCLUDE_DIRS})

target_link_libraries(camera_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${GST_LIBRARIES})

install(TARGETS stereo_camera_node
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS camera_node
  DESTINATION lib/${PROJECT_NAME})


# Install launch files
# install(DIRECTORY launch/
#     DESTINATION share/${PROJECT_NAME}/launch/
#     FILES_MATCHING PATTERN "*.py"
# )

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


# # Optional: Install config files (like RViz configs)
# install(DIRECTORY config/
#     DESTINATION share/${PROJECT_NAME}/config/
#     FILES_MATCHING PATTERN "*.rviz"
# )

# Optional: Install any header files if this becomes a library
install(DIRECTORY include/
    DESTINATION include/
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/)

ament_package()