#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>

namespace jetson_stereo_camera {

// Metric structures with data only - methods implemented in .cpp
struct ImageQualityMetrics {
    double blur_score = 0.0;
    double brightness_mean = 0.0;
    double brightness_std = 0.0;
    double contrast = 0.0;
    int total_pixels = 0;

    void log(rclcpp::Logger logger, const std::string& image_name) const;
};

struct RectificationQualityMetrics {
    double mean_epipolar_error = 0.0;
    double max_epipolar_error = 0.0;
    int samples_checked = 0;
    bool quality_acceptable = false;

    void log(rclcpp::Logger logger) const;
};

struct DisparityQualityMetrics {
    int total_pixels = 0;
    int valid_pixels = 0;
    int invalid_pixels = 0;
    double valid_ratio = 0.0;
    double mean_disparity = 0.0;
    double std_disparity = 0.0;
    double min_disparity = 0.0;
    double max_disparity = 0.0;
    int histogram[256] = {0};

    void log(rclcpp::Logger logger) const;
};

struct PointCloudQualityMetrics {
    int total_points = 0;
    int finite_points = 0;
    int infinite_or_nan_points = 0;
    double density = 0.0;
    double mean_x = 0.0, mean_y = 0.0, mean_z = 0.0;
    double std_x = 0.0, std_y = 0.0, std_z = 0.0;
    double min_z = 0.0, max_z = 0.0;
    int outliers_detected = 0;
    double outlier_ratio = 0.0;

    void log(rclcpp::Logger logger, const std::string& stage) const;
};

// Quality analysis functions
ImageQualityMetrics analyze_image_quality(const cv::Mat& image);

RectificationQualityMetrics analyze_rectification_quality(const cv::Mat& left_rect,
                                                           const cv::Mat& right_rect,
                                                           int num_samples = 50);

DisparityQualityMetrics analyze_disparity_quality(const cv::Mat& disparity);

PointCloudQualityMetrics analyze_pointcloud_quality(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

// Visualization utilities
cv::Mat create_disparity_colored(const cv::Mat& disparity, double min_disp, double max_disp);

cv::Mat create_depth_uncertainty_map(const cv::Mat& disparity, const cv::Mat& Q_matrix);

} // namespace jetson_stereo_camera
