#include <opencv/cv.h>
#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "TagDetection.h"
#include "AprilTypes.h"

#include <geometry_msgs/Pose.h>

using namespace pcl;

class KinectPoseImprovement{
private:
    size_t m_num_samples;//recommended 9
    PointCloud<PointXYZRGB>::Ptr m_cloud;
    at::Mat m_tag_space_samples;// 3 by n matrix stores n points in each row

public:
    KinectPoseImprovement(
            size_t num_tag_samples,
            PointCloud<PointXYZRGB>::Ptr input_cloud
    ):
            m_num_samples(num_tag_samples),
            m_cloud(input_cloud)
    {
        gen_tag_samples();
    }


    int localize_2d(TagDetection& detection, geometry_msgs::Pose& out_pose){
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

    int localize(TagDetection& detection, geometry_msgs::Pose& out_pose){
        std::cout << "localize started" << std::endl;

        // sample(segment) a tag corresponding to this detection.
        PointCloud<PointXYZRGB>::Ptr tag_sample_cloud = sample_cloud(detection);
        std::cout << "finishied sample cloud" << std::endl;
        PointCloud<PointXYZRGB>::Ptr corners_3D = extract_corners(detection);
        std::cout << "finished extract corners" << std::endl;

        //fit plane on sampled cloud
        pcl::PointIndices::Ptr inlier_idxs=boost::make_shared<pcl::PointIndices>();
        pcl::ModelCoefficients coeffs;
        PointCloud<PointXYZRGB>::Ptr inliers(new PointCloud<PointXYZRGB>());

        pcl::SACSegmentation<PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.005);

        seg.setInputCloud(tag_sample_cloud);
        seg.segment(*inlier_idxs, coeffs);

        pcl::ExtractIndices<PointXYZRGB> extracter;
        extracter.setInputCloud(tag_sample_cloud);
        extracter.setIndices(inlier_idxs);
        extracter.setNegative(false);
        extracter.filter(*inliers);

        std::cout << "finished RANSAC" << std::endl;

        //set center as the mean of inliers
        out_pose.position = centroid(*inliers);

        std::cout << "finished centeroid" << std::endl;

        //get orientation
        Eigen::Matrix3d R;
        extractFrame(coeffs, *corners_3D, R);
        Eigen::Quaternion<double> q(R);

        std::cout << "finished extractFrame" << std::endl;

        out_pose.orientation.x = q.x();
        out_pose.orientation.y = q.y();
        out_pose.orientation.z = q.z();
        out_pose.orientation.w = q.w();


        //debug
//        size_t i = 8;
//        out_pose.position.x = tag_sample_cloud->points[i].x;
//        out_pose.position.y = tag_sample_cloud->points[i].y;
//        out_pose.position.z = tag_sample_cloud->points[i].z;

        return 0;
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

    PointCloud<PointXYZRGB>::Ptr sample_cloud(TagDetection& detection){
        PointCloud<PointXYZRGB>::Ptr out_cloud(new PointCloud<PointXYZRGB>());

        // calcuate points in image coordinate.
        at::Mat img_idx_mat = detection.homography * m_tag_space_samples;
        size_t x = 0; // col id in image(0,0) at center of image(according to pcl_conversions.h)
        size_t y = 0; // row id in image
        for(size_t i = 0; i < m_num_samples * m_num_samples; i++){
            x = size_t(img_idx_mat[0][i]/img_idx_mat[2][i] + detection.hxy.x);
            y = size_t(img_idx_mat[1][i]/img_idx_mat[2][i] + detection.hxy.y);

            const PointXYZRGB& pt = (*m_cloud)(x, y);

            //check if pt is NaN
            if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)){
                //ROS_INFO("    Skipping (%.4f, %.4f, %.4f)", pt.x, pt.y, pt.z);
            }else{
                out_cloud->points.push_back(pt);
            }
        }
        return out_cloud;
    }

    PointCloud<PointXYZRGB>::Ptr extract_corners(TagDetection& detection){
        PointCloud<PointXYZRGB>::Ptr out_cloud(new PointCloud<PointXYZRGB>());
        for(size_t i = 0; i < 4; i++){
            out_cloud->points.push_back((*m_cloud)(detection.p[i].x, detection.p[i].y));
            std::cout << i << std::endl;
        }
        return out_cloud;
    }

    geometry_msgs::Point centroid (const PointCloud<PointXYZRGB>& points)
    {
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

    Eigen::Vector3d project(const PointXYZRGB& p, const double a, const double b,
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
                      PointCloud<PointXYZRGB>& corners,
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

//        m << v[0], v[1], v[2],
//                w[0], w[1], w[2],
//                n[0], n[1], n[2];


        retmat = m.inverse();

        return 0;
    }
};




