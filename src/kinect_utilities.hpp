#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>

#include "TagDetection.h"
#include "AprilTypes.h"

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

    int improve(TagDetectionArray& detections){
        int status = 0;
        for(size_t i = 0; i < detections.size(); i++){
            int succ = improve(detections[i]);
            if(succ < 0) status = -1;
        }
        return status;
    }

    int improve(TagDetection& detection){
        // sample(segment) a tag corresponding to this detection.
        PointCloud<PointXYZRGB>::Ptr tag_sample_cloud = sample_cloud(detection);

        //get marker pose
        Eigen::Matrix4d pose;
        cv::Mat rvec;
        cv::Mat tvec;
        GetMarkerTransformUsingOpenCV(detection, pose, rvec, tvec);

        return 0;
    }

private:
    void gen_tag_samples(){
        m_tag_space_samples = at::Mat::zeros(3, int(m_num_samples*m_num_samples));
        at::real step = 2.0/m_num_samples;
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
        for(size_t i = 0; i < m_num_samples; i++){
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
};




