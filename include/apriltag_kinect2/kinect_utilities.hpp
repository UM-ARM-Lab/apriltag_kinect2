//#include <opencv/cv.h>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/calib3d.hpp>

#include <Eigen/Core>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <apriltagscpp/TagDetection.h>
#include <apriltagscpp/AprilTypes.h>

#include <geometry_msgs/Pose.h>


double GetTagSize(int tag_id)
{
    boost::unordered_map<size_t, double>::iterator tag_sizes_it =
            tag_sizes_.find(tag_id);
    if(tag_sizes_it != tag_sizes_.end()) {
        return tag_sizes_it->second;
    } else {
        return default_tag_size_;
    }
}

/*
void GetMarkerTransformUsingOpenCV(const TagDetection& detection, Eigen::Matrix4d& transform, cv::Mat& rvec, cv::Mat& tvec)
{
    // Check if fx,fy or cx,cy are not set
    if ((camera_info_.K[0] == 0.0) || (camera_info_.K[4] == 0.0) || (camera_info_.K[2] == 0.0) || (camera_info_.K[5] == 0.0))
    {
        ROS_WARN("Warning: Camera intrinsic matrix K is not set, can't recover 3D pose");
    }

    double tag_size = GetTagSize(detection.id);

    std::vector<cv::Point3f> object_pts;
    std::vector<cv::Point2f> image_pts;
    double tag_radius = tag_size/2.;

    object_pts.push_back(cv::Point3f(-tag_radius, -tag_radius, 0));
    object_pts.push_back(cv::Point3f( tag_radius, -tag_radius, 0));
    object_pts.push_back(cv::Point3f( tag_radius,  tag_radius, 0));
    object_pts.push_back(cv::Point3f(-tag_radius,  tag_radius, 0));

    image_pts.push_back(detection.p[0]);
    image_pts.push_back(detection.p[1]);
    image_pts.push_back(detection.p[2]);
    image_pts.push_back(detection.p[3]);

    cv::Matx33f intrinsics(camera_info_.K[0], 0, camera_info_.K[2],
                           0, camera_info_.K[4], camera_info_.K[5],
                           0, 0, 1);

    cv::Vec4f distortion_coeff(camera_info_.D[0], camera_info_.D[1], camera_info_.D[2], camera_info_.D[3]);

    // Estimate 3D pose of tag
    // Methods:
    //   CV_ITERATIVE
    //     Iterative method based on Levenberg-Marquardt optimization.
    //     Finds the pose that minimizes reprojection error, being the sum of squared distances
    //     between the observed projections (image_points) and the projected points (object_pts).
    //   CV_P3P
    //     Based on: Gao et al, "Complete Solution Classification for the Perspective-Three-Point Problem"
    //     Requires exactly four object and image points.
    //   CV_EPNP
    //     Moreno-Noguer, Lepetit & Fua, "EPnP: Efficient Perspective-n-Point Camera Pose Estimation"
    int method = CV_ITERATIVE;
    bool use_extrinsic_guess = false; // only used for ITERATIVE method
    cv::solvePnP(object_pts, image_pts, intrinsics, distortion_coeff, rvec, tvec, use_extrinsic_guess, method);

    cv::Matx33d r;
    cv::Rodrigues(rvec, r);
    Eigen::Matrix3d rot;
    rot << r(0,0), r(0,1), r(0,2),
            r(1,0), r(1,1), r(1,2),
            r(2,0), r(2,1), r(2,2);

    Eigen::Matrix4d T;
    T.topLeftCorner(3,3) = rot;
    T.col(3).head(3) <<
                     tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
    T.row(3) << 0,0,0,1;

    transform = T;
}
*/

class KinectPoseImprovement{
private:
    size_t m_num_samples;//recommended 9
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr m_cloud;
    at::Mat m_tag_space_samples;// 3 by n matrix stores n points in each row

public:
    KinectPoseImprovement(
            size_t num_tag_samples,
            pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr input_cloud
    ):
            m_num_samples(num_tag_samples),
            m_cloud(input_cloud)
    {
        gen_tag_samples();
    }


    void localize_2d(TagDetection& detection, geometry_msgs::Pose& out_pose){
        Eigen::Matrix4d pose;
        cv::Mat rvec;
        cv::Mat tvec;
        GetMarkerTransformUsingOpenCV(detection, pose, rvec, tvec);

        // Get this info from earlier code, don't extract it again
        Eigen::Matrix3d R = pose.block<3,3>(0,0);
        Eigen::Quaternion<double> q(R);

        out_pose.position.x = pose(0,3);
        out_pose.position.y = pose(1,3);
        out_pose.position.z = pose(2,3);
        out_pose.orientation.x = q.x();
        out_pose.orientation.y = q.y();
        out_pose.orientation.z = q.z();
        out_pose.orientation.w = q.w();
    }

    void localize(TagDetection& detection, geometry_msgs::Pose& out_pose){
        // sample(segment) a tag corresponding to this detection.
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tag_sample_cloud = sample_cloud(detection);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr corners_3D = extract_corners(detection);

        //fit plane on sampled cloud
        pcl::PointIndices::Ptr inlier_idxs=boost::make_shared<pcl::PointIndices>();
        pcl::ModelCoefficients coeffs;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr inliers(new pcl::PointCloud<pcl::PointXYZRGB>());

        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.005);

        seg.setInputCloud(tag_sample_cloud);
        seg.segment(*inlier_idxs, coeffs);

        pcl::ExtractIndices<pcl::PointXYZRGB> extracter;
        extracter.setInputCloud(tag_sample_cloud);
        extracter.setIndices(inlier_idxs);
        extracter.setNegative(false);
        extracter.filter(*inliers);

        //set center as the mean of inliers
        out_pose.position = centroid(*inliers);

        //get orientation
        Eigen::Matrix3d R;
        extractFrame(coeffs, *corners_3D, R);
        Eigen::Quaternion<double> q(R);
        q.normalize();

        out_pose.orientation.x = q.x();
        out_pose.orientation.y = q.y();
        out_pose.orientation.z = q.z();
        out_pose.orientation.w = q.w();
    }

private:
    void gen_tag_samples(){
        m_tag_space_samples = at::Mat::zeros(3, int(m_num_samples*m_num_samples));
        at::real step = 2.0/ (m_num_samples - 1);
        for(size_t y=0; y < m_num_samples; y++){
            for(size_t x=0; x < m_num_samples; x++){
                size_t idx = y*m_num_samples + x;
                m_tag_space_samples[0][idx] = at::real(x) * step - 1.0;
                m_tag_space_samples[1][idx] = at::real(y) * step - 1.0;
                m_tag_space_samples[2][idx] = 1.0;
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sample_cloud(TagDetection& detection){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        // calcuate points in image coordinate.
        at::Mat img_idx_mat = detection.homography * m_tag_space_samples;
        size_t x = 0; // col id in image(0,0) at center of image(according to pcl_conversions.h)
        size_t y = 0; // row id in image
        for(size_t i = 0; i < m_num_samples * m_num_samples; i++){
            x = size_t(img_idx_mat[0][i]/img_idx_mat[2][i] + detection.hxy.x);
            y = size_t(img_idx_mat[1][i]/img_idx_mat[2][i] + detection.hxy.y);

            const pcl::PointXYZRGB& pt = (*m_cloud)(x, y);

            //check if pt is NaN
            if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)){
                //ROS_INFO("    Skipping (%.4f, %.4f, %.4f)", pt.x, pt.y, pt.z);
            }else{
                out_cloud->points.push_back(pt);
            }
        }
        return out_cloud;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_corners(TagDetection& detection){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for(size_t i = 0; i < 4; i++){
            out_cloud->points.push_back((*m_cloud)(detection.p[i].x, detection.p[i].y));
        }
        return out_cloud;
    }

    geometry_msgs::Point centroid (const pcl::PointCloud<pcl::PointXYZRGB>& points)
    {
        // find the mean of all point coordinate.
        geometry_msgs::Point sum;
        sum.x = 0;
        sum.y = 0;
        sum.z = 0;
        //for (const Point& p : points)
        for(size_t i=0; i<points.size(); i++)
        {
            sum.x += points[i].x;
            sum.y += points[i].y;
            sum.z += points[i].z;
        }

        geometry_msgs::Point center;
        const size_t n = points.size();
        center.x = sum.x/n;
        center.y = sum.y/n;
        center.z = sum.z/n;
        return center;
    }

    Eigen::Vector3d project(const pcl::PointXYZRGB& p, const double a, const double b,
                         const double c, const double d)
    {
        const double t = a*p.x + b*p.y + c*p.z + d;
        return Eigen::Vector3d(p.x-t*a, p.y-t*b, p.z-t*c);
    }

    int getCoeffs (const pcl::ModelCoefficients& coeffs, double& a, double& b,
                   double& c, double& d)
    {
        if(coeffs.values.size() != 4)
            return -1;
        const double s = coeffs.values[0]*coeffs.values[0] +
                         coeffs.values[1]*coeffs.values[1] + coeffs.values[2]*coeffs.values[2];
        if(fabs(s) < 1e-6)
            return -1;
        a = coeffs.values[0]/s;
        b = coeffs.values[1]/s;
        c = coeffs.values[2]/s;
        d = coeffs.values[3]/s;
        return 0;
    }

    int extractFrame (const pcl::ModelCoefficients& coeffs,
                      pcl::PointCloud<pcl::PointXYZRGB>& corners,
                      Eigen::Matrix3d &retmat){
        double a=0, b=0, c=0, d=0;
        if(getCoeffs(coeffs, a, b, c, d) < 0){
            return -1;
        }

        const Eigen::Vector3d q1 = project(corners.points[0], a, b, c, d);
        const Eigen::Vector3d q2 = project(corners.points[1], a, b, c, d);
        const Eigen::Vector3d q3 = project(corners.points[0], a, b, c, d);
        const Eigen::Vector3d q4 = project(corners.points[3], a, b, c, d);

        const Eigen::Vector3d v = (q2-q1).normalized();
        const Eigen::Vector3d n(a, b, c);
        const Eigen::Vector3d w = -v.cross(n);
        Eigen::Matrix3d m;
        m << v[0], v[1], v[2],
             w[0], w[1], w[2],
             n[0], n[1], n[2];

        retmat = m.inverse();

        return 0;
    }
};




