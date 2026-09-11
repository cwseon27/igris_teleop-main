cmake_minimum_required(VERSION 3.8)
project(igris_c_sdk)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Header.msg"
  "msg/IMUState.msg"
  "msg/MotorCmd.msg"
  "msg/MotorState.msg"
  "msg/JointState.msg"
  "msg/LowCmd.msg"
  "msg/LowState.msg"
  "msg/BmsState.msg"
  "msg/RobotState.msg"
  "msg/ServiceResponse.msg"
  "msg/WalkTwist.msg"
  "srv/BmsInitCmd.srv"
  "srv/TorqueCmd.srv"
  "srv/ControlModeCommandRequest.srv"
  "srv/HandInitRequest.srv"
  "srv/MujocoSimCmd.srv"
  DEPENDENCIES geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
