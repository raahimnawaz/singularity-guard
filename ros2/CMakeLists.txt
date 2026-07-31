cmake_minimum_required(VERSION 3.16)
project(armctl_ros2_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pluginlib REQUIRED)

# armctl itself lives in ../cpp/include (Phase 5) -- reused as-is, not copied.
set(ARMCTL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/include)

add_library(armctl_ros2_control SHARED
  src/singularity_controller.cpp
)
target_include_directories(armctl_ros2_control PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${ARMCTL_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(armctl_ros2_control PUBLIC cxx_std_17)
ament_target_dependencies(armctl_ros2_control
  controller_interface
  hardware_interface
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
  pluginlib
)

pluginlib_export_plugin_description_file(controller_interface armctl_ros2_control_plugin.xml)

install(
  TARGETS armctl_ros2_control
  EXPORT export_armctl_ros2_control
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY ${ARMCTL_INCLUDE_DIR}/armctl DESTINATION include)
install(DIRECTORY urdf worlds config launch DESTINATION share/${PROJECT_NAME})
# Installed as a program (not into share/) so `ros2 run armctl_ros2_control
# record_run.py` finds it on the libexec path with its exec bit intact.
install(PROGRAMS scripts/record_run.py DESTINATION lib/${PROJECT_NAME})
install(FILES armctl_ros2_control_plugin.xml DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(armctl_ros2_control)
ament_export_dependencies(
  controller_interface
  hardware_interface
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
  pluginlib
)
ament_export_targets(export_armctl_ros2_control)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
