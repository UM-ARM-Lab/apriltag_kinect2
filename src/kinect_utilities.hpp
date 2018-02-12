#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>

#include "TagDetection.h"


static constexpr int IMAGE_WIDTH = 1920;
static constexpr int IMAGE_HEIGHT = 1080;
cv::Mat test_matrix_;

using namespace pcl;

void mat_init(){
    test_matrix_.create(IMAGE_WIDTH * IMAGE_HEIGHT, 3, CV_32FC1);
    for (int x_ind = 0; x_ind < IMAGE_WIDTH; ++x_ind)
    {
        for (int y_ind = 0; y_ind < IMAGE_HEIGHT; ++y_ind)
        {
            test_matrix_.at<float>(x_ind * IMAGE_HEIGHT + y_ind, 0) = (float)x_ind;
            test_matrix_.at<float>(x_ind * IMAGE_HEIGHT + y_ind, 1) = (float)y_ind;
            test_matrix_.at<float>(x_ind * IMAGE_HEIGHT + y_ind, 2) = 1.0f;
        }
    }
}

cv::Mat getRegionMask(const TagDetection& tag)
{
    cv::Mat valid_region_mask(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255));

    // Iterate through each line, finding the valid region of the image for each one
    for (size_t first_corner_ind = 0; first_corner_ind < 4; ++first_corner_ind)
    {
        const size_t second_corner_ind = (first_corner_ind + 1) % 4;
        const size_t opposite_corner_ind = (first_corner_ind + 2) % 4;

        // Find the nullspace of A to determine the equation for the line
        Eigen::MatrixXf A(2, 3);
        A << tag.p[first_corner_ind].x,  tag.p[first_corner_ind].y,  1.0f,
                tag.p[second_corner_ind].x, tag.p[second_corner_ind].y, 1.0f;
        const auto lu_decomp = Eigen::FullPivLU<Eigen::MatrixXf>(A);
        const Eigen::Vector3f nullspace_eigen = lu_decomp.kernel();

//            std::cout << "A:\n" << A << std::endl << std::endl;
//            std::cout << "nullspace_eigen:" << nullspace_eigen.transpose() << std::endl << std::endl;

        cv::Mat abc(3, 1, CV_32FC1);
        abc.at<float>(0, 0) = nullspace_eigen(0);
        abc.at<float>(1, 0) = nullspace_eigen(1);
        abc.at<float>(2, 0) = nullspace_eigen(2);

//            std::cout << "Eigen: " << nullspace_eigen.transpose() << std::endl;
//            std::cout << "Eigen manual: " << nullspace_eigen(0) << " " << nullspace_eigen(1) << " " << nullspace_eigen(2) << std::endl;
//            std::cout << "CV:    " << abc.t() << std::endl;

        const cv::Mat raw_math_vals = test_matrix_ * abc;

//            std::cout << "Raw math vals: " << raw_math_vals(cv::Rect(0, 0, 1, 5)).t() << std::endl;

        // Determine which side of the line we should count as "in"
        const float value =
                abc.at<float>(0, 0) * tag.p[opposite_corner_ind].x +
                abc.at<float>(1, 0) * tag.p[opposite_corner_ind].y +
                abc.at<float>(2, 0);
        cv::Mat valid_region;
        if (value < 0.0f)
        {
            valid_region = raw_math_vals <= 0.0f;
        }
        else
        {
            valid_region = raw_math_vals >= 0.0f;
        }

        const cv::Mat tmp = valid_region.reshape(0, IMAGE_WIDTH).t();
        valid_region_mask &= tmp;

//            std::cout << "First 20 elements of valid region:\n" << valid_region(cv::Rect(0, 0, 1, 20)) << std::endl;
//            std::cout << "First 20 elements of valid region mask:\n" << valid_region_mask(cv::Rect(0, 0, 1, 20)) << std::endl;

//            cv::imshow("Display window", valid_region_mask);         // Show our image inside it.
//            cv::waitKey(1000);                                       // Wait for a keystroke in the window

//            std::cout << valid_region << std::endl << std::endl << std::endl << std::endl;

    }

    return valid_region_mask;
}

PointCloud<PointXYZRGB>::Ptr extractPoints(const PointCloud<PointXYZRGB>::ConstPtr& input_cloud, const cv::Mat& region_mask)
{
    // build the indices that we will extract
    PointIndices::Ptr inliers (new PointIndices());
    for (int x_ind = 0; x_ind < IMAGE_WIDTH; ++x_ind)
    {
        for (int y_ind = 0; y_ind < IMAGE_HEIGHT; ++y_ind)
        {
            if (region_mask.at<uchar>(y_ind, x_ind) != 0)
            {
                inliers->indices.push_back(y_ind * IMAGE_WIDTH + x_ind);
            }
        }
    }

    // Perform the extraction itself
    PointCloud<PointXYZRGB>::Ptr extracted_cloud(new PointCloud<PointXYZRGB>());
    ExtractIndices<PointXYZRGB> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.filter(*extracted_cloud);

    // Filter out any NaN values
    PointCloud<PointXYZRGB>::Ptr output_cloud(new PointCloud<PointXYZRGB>());
    std::vector<int> index; // just used to satisfy the interface, not actually consumed
    pcl::removeNaNFromPointCloud(*extracted_cloud, *output_cloud, index);

    return output_cloud;
}

PointCloud<PointXYZRGB>::Ptr filter_tag(const TagDetection& tag, const PointCloud<PointXYZRGB>::ConstPtr& input_cloud){
    mat_init();
    cv::Mat region_mask;
    region_mask = getRegionMask(tag);
    return extractPoints(input_cloud, region_mask);
}


int PlaneFitPoseImprovement(int id, const ARCloud &corners_3D, ARCloud::Ptr selected_points, const ARCloud &cloud, Pose &p){

    ata::PlaneFitResult res = ata::fitPlane(selected_points);
    gm::PoseStamped pose;
    pose.header.stamp = pcl_conversions::fromPCL(cloud.header).stamp;
    pose.header.frame_id = cloud.header.frame_id;
    pose.pose.position = ata::centroid(*res.inliers);

    draw3dPoints(selected_points, cloud.header.frame_id, 1, id, 0.005);

    //Get 2 points that point forward in marker x direction
    int i1,i2;
    if(isnan(corners_3D[0].x) || isnan(corners_3D[0].y) || isnan(corners_3D[0].z) ||
       isnan(corners_3D[3].x) || isnan(corners_3D[3].y) || isnan(corners_3D[3].z))
    {
        if(isnan(corners_3D[1].x) || isnan(corners_3D[1].y) || isnan(corners_3D[1].z) ||
           isnan(corners_3D[2].x) || isnan(corners_3D[2].y) || isnan(corners_3D[2].z))
        {
            return -1;
        }
        else{
            i1 = 1;
            i2 = 2;
        }
    }
    else{
        i1 = 0;
        i2 = 3;
    }

    //Get 2 points the point forward in marker y direction
    int i3,i4;
    if(isnan(corners_3D[0].x) || isnan(corners_3D[0].y) || isnan(corners_3D[0].z) ||
       isnan(corners_3D[1].x) || isnan(corners_3D[1].y) || isnan(corners_3D[1].z))
    {
        if(isnan(corners_3D[3].x) || isnan(corners_3D[3].y) || isnan(corners_3D[3].z) ||
           isnan(corners_3D[2].x) || isnan(corners_3D[2].y) || isnan(corners_3D[2].z))
        {
            return -1;
        }
        else{
            i3 = 2;
            i4 = 3;
        }
    }
    else{
        i3 = 1;
        i4 = 0;
    }

    ARCloud::Ptr orient_points(new ARCloud());
    orient_points->points.push_back(corners_3D[i1]);
    draw3dPoints(orient_points, cloud.header.frame_id, 3, id+1000, 0.008);

    orient_points->clear();
    orient_points->points.push_back(corners_3D[i2]);
    draw3dPoints(orient_points, cloud.header.frame_id, 2, id+2000, 0.008);

    int succ;
    succ = ata::extractOrientation(res.coeffs, corners_3D[i1], corners_3D[i2], corners_3D[i3], corners_3D[i4], pose.pose.orientation);
    if(succ < 0) return -1;

    tf::Matrix3x3 mat;
    succ = ata::extractFrame(res.coeffs, corners_3D[i1], corners_3D[i2], corners_3D[i3], corners_3D[i4], mat);
    if(succ < 0) return -1;

    drawArrow(pose.pose.position, mat, cloud.header.frame_id, 1, id);

    p.translation[0] = pose.pose.position.x * 100.0;
    p.translation[1] = pose.pose.position.y * 100.0;
    p.translation[2] = pose.pose.position.z * 100.0;
    p.quaternion[1] = pose.pose.orientation.x;
    p.quaternion[2] = pose.pose.orientation.y;
    p.quaternion[3] = pose.pose.orientation.z;
    p.quaternion[0] = pose.pose.orientation.w;

    return 0;
}


