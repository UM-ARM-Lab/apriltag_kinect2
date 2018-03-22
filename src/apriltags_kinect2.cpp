/***********************************************************************

Copyright (c) 2014, Carnegie Mellon University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************/

#include "apriltag_kinect2/apriltags.h"
#include "apriltag_kinect2/kinect_utilities.hpp"
#include "apriltag_kinect2/AprilKinectDetections.h"

#include <iostream>
#include <chrono>

#include <opencv/cv.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>


using namespace std;

void broadcast_pose(geometry_msgs::Pose& pose){
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "/kinect2_victor_head_rgb_optical_frame";
    transformStamped.child_frame_id = "apriltag_kinect_test";
    transformStamped.transform.translation.x = pose.position.x;
    transformStamped.transform.translation.y = pose.position.y;
    transformStamped.transform.translation.z = pose.position.z;
    transformStamped.transform.rotation.x = pose.orientation.x;
    transformStamped.transform.rotation.y = pose.orientation.y;
    transformStamped.transform.rotation.z = pose.orientation.z;
    transformStamped.transform.rotation.w = pose.orientation.w;

    transform_broadcaster_->sendTransform(transformStamped);
}


// Functions


// Draw a line with an arrow head
// This function's argument list is designed to match the cv::line() function
void ArrowLine(cv::Mat& image,
               const cv::Point& pt1, const cv::Point& pt2, const cv::Scalar& color,
               const int thickness=1, const int line_type=8, const int shift=0,
               const double tip_length=0.1)
{
    // Normalize the size of the tip depending on the length of the arrow
    const double tip_size = norm(pt1-pt2)*tip_length;

    cv::line(image, pt1, pt2, color, thickness, line_type, shift);
    const double angle = atan2(double(pt1.y - pt2.y), double(pt1.x - pt2.x));
    cv::Point p(cvRound(pt2.x + tip_size * cos(angle + CV_PI / 4.0)),
    cvRound(pt2.y + tip_size * sin(angle + CV_PI / 4.0)));
    cv::line(image, p, pt2, color, thickness, line_type, shift);
    p.x = cvRound(pt2.x + tip_size * cos(angle - CV_PI / 4.0));
    p.y = cvRound(pt2.y + tip_size * sin(angle - CV_PI / 4.0));
    cv::line(image, p, pt2, color, thickness, line_type, shift);
}

// Draw the marker's axes with red/green/blue lines
void DrawMarkerAxes(const cv::Matx33f& intrinsic_matrix, const cv::Vec4f& distortion_coeffs,
                    const cv::Mat& rvec, const cv::Mat& tvec, const float length, const bool use_arrows,
                    cv::Mat& image)
{
    std::vector<cv::Point3f> axis_points;
    axis_points.push_back(cv::Point3f(0, 0, 0));
    axis_points.push_back(cv::Point3f(length, 0, 0));
    axis_points.push_back(cv::Point3f(0, length, 0));
    axis_points.push_back(cv::Point3f(0, 0, length));
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(axis_points, rvec, tvec, intrinsic_matrix, distortion_coeffs, image_points);

    // Draw axis lines
    const int thickness = 2;
    if (use_arrows)
    {
        ArrowLine(image, image_points[0], image_points[1], cv::Scalar(0, 0, 255), thickness);
        ArrowLine(image, image_points[0], image_points[2], cv::Scalar(0, 255, 0), thickness);
        ArrowLine(image, image_points[0], image_points[3], cv::Scalar(255, 0, 0), thickness);
    }
    else
    {
        cv::line(image, image_points[0], image_points[1], cv::Scalar(0, 0, 255), thickness);
        cv::line(image, image_points[0], image_points[2], cv::Scalar(0, 255, 0), thickness);
        cv::line(image, image_points[0], image_points[3], cv::Scalar(255, 0, 0), thickness);
    }
}

// Draw the outline of the square marker in a single color,
// with a mark on the first corner
void DrawMarkerOutline(const TagDetection& detection, const cv::Scalar outline_color, cv::Mat& image)
{
    // Draw outline
    const int outline_thickness = 2;
    for(int i = 0; i < 4; i++) {
        cv::Point2f p0(detection.p[i].x, detection.p[i].y);
        cv::Point2f p1(detection.p[(i + 1) % 4].x, detection.p[(i + 1) % 4].y);
        cv::line(image, p0, p1, outline_color, outline_thickness);
    }
    // Indicate first corner with a small rectangle
    const int width = 6;
    const int rect_thickness = 1;
    cv::Point2f p0(detection.p[0].x - width/2, detection.p[0].y - width/2);
    cv::Point2f p1(detection.p[0].x + width/2, detection.p[0].y + width/2);
    cv::rectangle(image, p0, p1, outline_color, rect_thickness, CV_AA); // anti-aliased
}

// Draw the four edges of the marker in separate colors
// to show the ordering of corner points
void DrawMarkerEdges(const TagDetection& detection, cv::Mat& image)
{
    // Draw edges
    std::vector<cv::Scalar> colors;

    colors.push_back(cv::Scalar(0,0,255)); // red (BGR ordering)
    colors.push_back(cv::Scalar(0,255,0)); // green
    colors.push_back(cv::Scalar(255,0,0)); // blue
    colors.push_back(cv::Scalar(0,204,255)); // yellow

    const int edge_thickness = 2;
    for(int i = 0; i < 4; i++) {
        cv::Point2f p0(detection.p[i].x, detection.p[i].y);
        cv::Point2f p1(detection.p[(i + 1) % 4].x, detection.p[(i + 1) % 4].y);
        cv::line(image, p0, p1, colors[i], edge_thickness);
    }
}

// Draw the marker's ID number
void DrawMarkerID(const TagDetection& detection, const cv::Scalar text_color, cv::Mat& image)
{
    cv::Point2f center(0,0);
    for(int i = 0; i < 4; i++)
    {
        center += cv::Point2f(detection.p[i].x, detection.p[i].y);
    }
    center.x = center.x / 4.0f;
    center.y = center.y / 4.0f;
    std::stringstream s;
    s << "  " << detection.id; // move label away from origin
    const double font_scale = 0.5;
    const int thickness = 2;
    cv::putText(image, s.str(), center, cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
}

// Callback for camera info
void InfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info)
{
    camera_info_ = (*camera_info);
    has_camera_info_ = true;
}


// Callback for point cloud
void getPointCloudCallback (const sensor_msgs::PointCloud2ConstPtr &pc_msg)
{
    auto begin = std::chrono::steady_clock::now();

    //extract pointcloud to pcl format
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_native_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    sensor_msgs::ImagePtr ros_image_msg(new sensor_msgs::Image);

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*pc_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, *pcl_native_cloud);

    pcl::PCLImage pcl_image;
    pcl::toPCLPointCloud2(pcl_pc2, pcl_image);
    pcl_conversions::moveFromPCL(pcl_image, *ros_image_msg);

    ros_image_msg->header.stamp = pc_msg->header.stamp;
    ros_image_msg->header.frame_id = pc_msg->header.frame_id;

    if(!has_camera_info_){
        ROS_WARN("No Camera Info Received Yet");
        return;
    }

    // Get the image
    cv_bridge::CvImagePtr subscribed_ptr;
    try
    {
        subscribed_ptr = cv_bridge::toCvCopy(ros_image_msg, "mono8");
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat subscribed_gray = subscribed_ptr->image;
    cv::Point2d opticalCenter;

    if ((camera_info_.K[2] > 1.0) && (camera_info_.K[5] > 1.0))
    {
        // cx,cy from intrinsic matric look reasonable, so we'll use that
        opticalCenter = cv::Point2d(camera_info_.K[5], camera_info_.K[2]);
    }
    else
    {
        opticalCenter = cv::Point2d(0.5*subscribed_gray.rows, 0.5*subscribed_gray.cols);
    }

    // Detect AprilTag markers in the image
    TagDetectionArray detections;
    detector_->process(subscribed_gray, opticalCenter, detections);


    KinectPoseImprovement improvement_obj(9, pcl_native_cloud);


    // After detection, send over message
    visualization_msgs::MarkerArray marker_array;
    apriltag_kinect2::AprilKinectDetections aprilkinect_detections;
    aprilkinect_detections.header.frame_id = ros_image_msg->header.frame_id;
    aprilkinect_detections.header.stamp = ros_image_msg->header.stamp;

    cv_bridge::CvImagePtr subscribed_color_ptr;
    if ((viewer_) || (publish_detections_image_))
    {
        try
        {
            subscribed_color_ptr = cv_bridge::toCvCopy(ros_image_msg, "bgr8");
        }
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        if (display_marker_overlay_)
        {
            // Overlay a black&white marker for each detection
            subscribed_color_ptr->image = family_->superimposeDetections(subscribed_color_ptr->image, detections);
        }
    }

    for(size_t i = 0; i < detections.size(); ++i)
    {
        // skip bad detections
        if(!detections[i].good)
        {
            continue;
        }

//        Eigen::Matrix4d pose;
//        cv::Mat rvec;
//        cv::Mat tvec;
//        GetMarkerTransformUsingOpenCV(detections[i], pose, rvec, tvec);
//
//        // Get this info from earlier code, don't extract it again
//        Eigen::Matrix3d R = pose.block<3,3>(0,0);
//        Eigen::Quaternion<double> q(R);

        double tag_size = GetTagSize(int(detections[i].id));

        // Fill in MarkerArray msg
        visualization_msgs::Marker marker;
        marker.header.frame_id = ros_image_msg->header.frame_id;
        marker.header.stamp = ros_image_msg->header.stamp;

        // Only publish marker for 0.5 seconds after it
        // was last seen
        marker.lifetime = ros::Duration(1.0);

        stringstream convert;
        convert << "tag" << detections[i].id;
        marker.ns = convert.str().c_str();
        marker.id = int(detections[i].id);
        if(display_type_ == "ARROW")
        {
            marker.type = visualization_msgs::Marker::ARROW;
            marker.scale.x = tag_size; // arrow length
            marker.scale.y = tag_size/10.0; // diameter
            marker.scale.z = tag_size/10.0; // diameter
        }
        else if(display_type_ == "CUBE")
        {
            marker.type = visualization_msgs::Marker::CUBE;
            marker.scale.x = tag_size;
            marker.scale.y = tag_size;
            marker.scale.z = marker_thickness_;
        }
        marker.action = visualization_msgs::Marker::ADD;
        improvement_obj.localize(detections[i], marker.pose);

        if(broadcast_tf_)
        {
            if(int(detections[i].id) == tf_marker_id_)
            {
                broadcast_pose(marker.pose);
            }
        }
//        marker_transform.pose.position.x = pose(0,3);
//        marker_transform.pose.position.y = pose(1,3);
//        marker_transform.pose.position.z = pose(2,3);
//        marker_transform.pose.orientation.x = q.x();
//        marker_transform.pose.orientation.y = q.y();
//        marker_transform.pose.orientation.z = q.z();
//        marker_transform.pose.orientation.w = q.w();

        marker.color.r = 1.0f;
        marker.color.g = 0.0f;
        marker.color.b = 1.0f;
        marker.color.a = 1.0f;
        marker_array.markers.push_back(marker);

        const TagDetection &det = detections[i];

        // Fill in AprilTag detection.
        apriltag_kinect2::AprilKinectDetection aprilkinect_det;
        aprilkinect_det.header = marker.header;
        aprilkinect_det.id = int(marker.id);
        aprilkinect_det.tag_size = float(tag_size);
        aprilkinect_det.hammingDistance = det.hammingDistance;
        aprilkinect_det.pose = marker.pose;

        aprilkinect_detections.detections.push_back(aprilkinect_det);

        if ((viewer_) || (publish_detections_image_))
        {
            if (display_marker_outline_)
            {
                cv::Scalar outline_color(0, 0, 255); // blue (BGR ordering)
                DrawMarkerOutline(detections[i], outline_color, subscribed_color_ptr->image);
            }

            if (display_marker_id_)
            {
                cv::Scalar text_color(255, 255, 0); // light-blue (BGR ordering)
                DrawMarkerID(detections[i], text_color, subscribed_color_ptr->image);
            }

            if (display_marker_edges_)
            {
                DrawMarkerEdges(detections[i], subscribed_color_ptr->image);
            }

        }
    }

    marker_publisher_.publish(marker_array);
    apriltag_publisher_.publish(aprilkinect_detections);

    if(publish_detections_image_)
    {
        image_publisher_.publish(subscribed_color_ptr->toImageMsg());
    }

    if(viewer_)
    {
        cv::imshow("AprilTags", subscribed_color_ptr->image);
    }


    auto end = std::chrono::steady_clock::now();
    auto dur = end - begin;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    ROS_DEBUG("aprilTags_kinect2 Callback Duration: %d milliseconds.", int(ms));
    std::cout << "Callback Duration: " << ms << " milliseconds \n";
}


void ConnectCallback(const ros::SingleSubscriberPublisher& info)
{
    // Check for subscribers.
    uint32_t subscribers = marker_publisher_.getNumSubscribers()
                           + apriltag_publisher_.getNumSubscribers();
    ROS_DEBUG("Subscription detected! (%d subscribers)", subscribers);

    if(subscribers && !running_)
    {
        ROS_DEBUG("New Subscribers, Connecting to Input Image Topic.");
        cloud_subscriber = (*node_).subscribe(DEFAULT_IMAGE_TOPIC, 1, &getPointCloudCallback);
        info_subscriber = (*node_).subscribe(
                DEFAULT_CAMERA_INFO_TOPIC, 10, &InfoCallback);
        running_ = true;
    }
}

void DisconnectHandler()
{
}

void DisconnectCallback(const ros::SingleSubscriberPublisher& info)
{
    // Check for subscribers.
    uint32_t subscribers = marker_publisher_.getNumSubscribers()
                           + apriltag_publisher_.getNumSubscribers();
    ROS_DEBUG("Unsubscription detected! (%d subscribers)", subscribers);
    
    if(!subscribers && running_)
    {
        ROS_DEBUG("No Subscribers, Disconnecting from Input Image Topic.");
        cloud_subscriber.shutdown();
        info_subscriber.shutdown();
        running_ = false;
    }
}

void GetParameterValues()
{
    // Load node-wide configuration values.
    node_->param("tag_family", tag_family_name_, DEFAULT_TAG_FAMILY);
    node_->param("default_tag_size", default_tag_size_, DEFAULT_TAG_SIZE);
    node_->param("display_type", display_type_, DEFAULT_DISPLAY_TYPE);
    node_->param("marker_thickness", marker_thickness_, 0.01);

    node_->param("viewer", viewer_, false);
    node_->param("publish_detections_image", publish_detections_image_, false);
    node_->param("display_marker_overlay", display_marker_overlay_, true);
    node_->param("display_marker_outline", display_marker_outline_, false);
    node_->param("display_marker_id", display_marker_id_, false);
    node_->param("display_marker_edges", display_marker_edges_, false);
    node_->param("display_marker_axes", display_marker_axes_, false);

    node_->param("broadcast_tf", broadcast_tf_, false);
    node_->param("tf_marker_id", tf_marker_id_, 0);

    ROS_INFO("Tag Family: %s", tag_family_name_.c_str());

    // Load tag specific configuration values.
    XmlRpc::XmlRpcValue tag_data;
    node_->param("tag_data", tag_data, tag_data);

    // Iterate through each tag in the configuration.
    XmlRpc::XmlRpcValue::ValueStruct::iterator it;
    for (it = tag_data.begin(); it != tag_data.end(); ++it)
    {
        // Retrieve the settings for the next tag.
        int tag_id = boost::lexical_cast<int>(it->first);
        XmlRpc::XmlRpcValue tag_values = it->second;

        // Load all the settings for this tag.
        if (tag_values.hasMember("size")) 
        {
            tag_sizes_[tag_id] = static_cast<double>(tag_values["size"]);
            ROS_DEBUG("Setting tag%d to size %f m.", tag_id, tag_sizes_[tag_id]);
        }
    }
}

void SetupPublisher()
{    
    ros::SubscriberStatusCallback connect_callback = &ConnectCallback;
    ros::SubscriberStatusCallback disconnect_callback = &DisconnectCallback;
    
    // Publisher
    marker_publisher_ = node_->advertise<visualization_msgs::MarkerArray>(
            DEFAULT_MARKER_TOPIC, 1, connect_callback,
            disconnect_callback);
    apriltag_publisher_ = node_->advertise<apriltag_kinect2::AprilKinectDetections>(
            DEFAULT_DETECTIONS_TOPIC, 1, connect_callback, disconnect_callback);

    if(publish_detections_image_)
    {
        image_publisher_ = (*image_).advertise(DEFAULT_DETECTIONS_IMAGE_TOPIC, 1);
    }
}

void InitializeTags()
{
    tag_params.newQuadAlgorithm = 1;
    family_ = new TagFamily(tag_family_name_);
    detector_ = new TagDetector(*family_, tag_params);
}

void InitializeROSNode(int argc, char **argv)
{
    ros::init(argc, argv, "apriltags");
    node_ =  boost::make_shared<ros::NodeHandle>("~");
    image_ = boost::make_shared<image_transport::ImageTransport>(*node_);
    transform_broadcaster_ = boost::make_shared<tf2_ros::TransformBroadcaster>();
}

int main(int argc, char **argv)
{
    InitializeROSNode(argc,argv);
    GetParameterValues();
    SetupPublisher();
    InitializeTags();

    if(viewer_){
        cvNamedWindow("AprilTags");
        cvStartWindowThread();
    }

    ROS_INFO("AprilTags node started.");
    running_ = false;
    has_camera_info_ = false;
//    ros::spin(); // original single threaded spinner
    ros::MultiThreadedSpinner spinner(0); // open one thread per processor
    spinner.spin();

    ROS_INFO("AprilTags node stopped.");

    //Destroying Stuff
    cvDestroyWindow("AprilTags");
    delete detector_;
    delete family_;

    return EXIT_SUCCESS;
}
