cmake_minimum_required(VERSION 3.8)
project(perception_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MarkerPose.msg"
  "msg/MarkerPoseArray.msg"
  "msg/ObjectPose.msg"     
  "msg/ObjectPoseArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
