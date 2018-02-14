cmake_minimum_required(VERSION 2.8.3)
project(apriltag_kinect2)

set(CMAKE_CXX_STANDARD 11)

# Set up the ROS Catkin package settings.
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  std_msgs
  sensor_msgs
  image_transport
  roscpp
  cmake_modules
  message_generation
  geometry_msgs
  pcl_ros
  pcl_conversions
)
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)

# apriltag cpp
find_package(apriltagscpp REQUIRED)

# Import the yaml-cpp libraries.
include(FindPkgConfig)
pkg_check_modules(Yaml REQUIRED yaml-cpp)

add_message_files(DIRECTORY msg FILES
    AprilTagDetection.msg
    AprilTagDetections.msg
    AprilKinectDetection.msg
    AprilKinectDetections.msg
)

generate_messages(DEPENDENCIES
    std_msgs
    geometry_msgs
)

# Set up the ROS Catkin package settings
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cv_bridge
                 std_msgs
                 sensor_msgs
                 image_transport
                 roscpp
                 geometry_msgs
                 pcl_ros
                 pcl_conversions
)

# CGAL requires that -frounding-math be set.
add_definitions(-frounding-math)

include_directories(
    include/
    ${catkin_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Yaml_INCLUDE_DIRS}
    ${apriltagscpp_INCLUDE_DIR}/apriltagscpp
)

add_executable(apriltag_kinect2 src/apriltags.cpp)
target_link_libraries(apriltag_kinect2 ${catkin_LIBRARIES}
                                ${Eigen_LIBRARIES}
                                ${OpenCV_LIBRARIES}
                                ${Yaml_LIBRARIES}    
                                ${apriltagscpp_LIBRARY})
add_dependencies(apriltag_kinect2 ${PROJECT_NAME}_gencpp)

install(TARGETS apriltag_kinect2
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION} 
)

#install(FILES ${apriltagscpp_LIBRARY}
#  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#)

