cmake_minimum_required(VERSION 3.0.2)
project(planner)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)