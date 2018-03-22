#include <string>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <Eigen/Core>

#include <apriltagscpp/TagDetector.h>
#include <apriltagscpp/TagDetection.h>
#include <apriltagscpp/TagFamily.h>

#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

const double SMALL_TAG_SIZE = 0.0358968;
const double MED_TAG_SIZE = 0.06096;
const double PAGE_TAG_SIZE = 0.165;

const std::string DEFAULT_TAG_FAMILY = "Tag36h11";
const std::string DEFAULT_IMAGE_TOPIC = "image";
const std::string DEFAULT_CAMERA_INFO_TOPIC = "camera_info";
const std::string DEFAULT_MARKER_TOPIC = "marker_array";
const std::string DEFAULT_DETECTIONS_TOPIC = "detections";
const std::string DEFAULT_DETECTIONS_IMAGE_TOPIC = "detections_image";
const double DEFAULT_TAG_SIZE = MED_TAG_SIZE;
const std::string DEFAULT_DISPLAY_TYPE = "CUBE";

// ROS parts
ros::NodeHandlePtr node_;
boost::shared_ptr<image_transport::ImageTransport> image_;
sensor_msgs::CameraInfo camera_info_;

boost::shared_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;

ros::Publisher marker_publisher_;
ros::Publisher apriltag_publisher_;
image_transport::Publisher image_publisher_;

ros::Subscriber info_subscriber;
ros::Subscriber cloud_subscriber;

// AprilTag parts
TagFamily* family_;
TagDetector* detector_;

TagDetectorParams tag_params;
std::string tag_data;
std::string tag_family_name_;

// Settings and local information
bool viewer_;
bool publish_detections_image_;
double default_tag_size_;
double marker_thickness_;
boost::unordered_map<size_t, double> tag_sizes_;
bool running_;
bool has_camera_info_;
std::string display_type_;

bool display_marker_overlay_;
bool display_marker_outline_;
bool display_marker_id_;
bool display_marker_edges_;
bool display_marker_axes_;

bool broadcast_tf_;
int tf_marker_id_;

void GetMarkerTransformUsingOpenCV(const TagDetection& detection, Eigen::Matrix4d& transform, cv::Mat& rvec, cv::Mat& tvec);

void ArrowLine(cv::Mat& image, const cv::Point& pt1, const cv::Point& pt2, const cv::Scalar& color,
               const int thickness, const int line_type, const int shift, const double tip_length);
void DrawMarkerAxes(const cv::Matx33f& intrinsic_matrix, const cv::Vec4f& distortion_coeffs,
                    const cv::Mat& rvec, const cv::Mat& tvec, const float length, const bool use_arrows,
                    cv::Mat& image);
void DrawMarkerOutline(const TagDetection& detection, const cv::Scalar outline_color, cv::Mat& image);
void DrawMarkerEdges(const TagDetection& detection, cv::Mat& image);
void DrawMarkerID(const TagDetection& detection, const cv::Scalar text_color, cv::Mat& image);

void InfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info);
void InfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info);
void ConnectCallback(const ros::SingleSubscriberPublisher& info);
void DisconnectCallback(const ros::SingleSubscriberPublisher& info);
void DisconnectHandler();
void GetParameterValues();
void SetupPublisher();
void InitializeTags();
void InitializeROSNode();
