cmake_minimum_required(VERSION 3.8)
project(jetank_perception)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set policy for find_package to avoid warnings
cmake_policy(SET CMP0074 NEW)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# Add filesystem for C++17
find_package(Threads REQUIRED)

# Link directories
link_directories(${PCL_LIBRARY_DIRS})

# Add compile definitions
add_definitions(${PCL_DEFINITIONS})

# Check if OpenCV has CUDA support
if(OpenCV_CUDA_VERSION)
    message(STATUS "OpenCV compiled with CUDA support")
    add_definitions(-DOPENCV_ENABLE_NONFREE)
else()
    message(STATUS "OpenCV compiled without CUDA support")
endif()

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

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

# Create executables
add_executable(stereo_camera_node 
  src/stereo_camera_node.cpp)

add_executable(camera_node 
  src/single_camera.cpp)

# Set target dependencies
ament_target_dependencies(stereo_camera_node 
  rclcpp
  sensor_msgs
  stereo_msgs
  std_srvs
  cv_bridge
  camera_info_manager
  pcl_conversions
  OpenCV)

ament_target_dependencies(camera_node 
  rclcpp
  sensor_msgs
  stereo_msgs
  std_srvs
  cv_bridge
  camera_info_manager
  pcl_conversions
  OpenCV)

# Configure stereo_camera_node
target_include_directories(stereo_camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GST_INCLUDE_DIRS})

target_compile_features(stereo_camera_node PUBLIC cxx_std_17)

target_link_libraries(stereo_camera_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${GST_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  Threads::Threads
  stdc++fs)

# Configure camera_node
target_include_directories(camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GST_INCLUDE_DIRS})

target_compile_features(camera_node PUBLIC cxx_std_17)

target_link_libraries(camera_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${GST_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  Threads::Threads
  stdc++fs)

# Install targets
install(TARGETS stereo_camera_node
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS camera_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install header files
install(DIRECTORY include/
    DESTINATION include/
    FILES_MATCHING PATTERN "*.hpp"
)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/)

ament_package()