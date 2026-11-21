cmake_minimum_required(VERSION 3.8)
project(team_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ControlState.msg"
  "msg/MissionState.msg"
  "msg/NodeStatus.msg"
  "msg/ServerState.msg"
  "msg/ServerCommand.msg"
  "msg/UavFrame.msg"
  "msg/FieldRegion.msg"
  "msg/FieldFace.msg"
  "msg/FieldVehicle.msg"
  "msg/FieldMap.msg"
  "msg/Observation.msg"
  "msg/ObservationArray.msg"
  "msg/ObjectCount.msg"
  "msg/SectorCount.msg"
  "msg/ParsedCommand.msg"
  "msg/ThreatAssessment.msg"
  "msg/Report.msg"
  "msg/Picture.msg"
  "msg/UgvState.msg"
  "msg/SensorSpot.msg"
  "srv/SetReady.srv"
  "srv/SubmitReport.srv"
  "srv/ClassifyImage.srv"
  "srv/ReadUgvs.srv"
  "srv/LocateSensors.srv"
  "action/Navigate.action"
  "action/TrackTarget.action"
  "action/Analyze.action"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs builtin_interfaces action_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
