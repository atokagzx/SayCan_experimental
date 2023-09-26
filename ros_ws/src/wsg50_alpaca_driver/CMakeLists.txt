cmake_minimum_required(VERSION 3.0.2)
project(wsg50_alpaca_driver)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  control_msgs
  std_msgs
  controller_manager_msgs
)
include_directories(
  ${catkin_INCLUDE_DIRS}
)
