#include "jetank_perception/stereo_processing_strategy.hpp"

#include <iostream>
#include <chrono>

namespace jetson_stereo_camera {

PointCloudFilter::PointCloudFilter(const PointCloudConfig& config)
    : config_(config) {}

void PointCloudFilter::filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    if (config_.apply_voxel_filter) {
        apply_voxel_filter(cloud);
    }
    if (config_.apply_statistical_filter) {
        apply_statistical_filter(cloud);
    }
    apply_range_filter(cloud);
}

void PointCloudFilter::update_config(const PointCloudConfig& config) {
    config_ = config;
}

void PointCloudFilter::apply_range_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Placeholder: filter points outside min/max range
}

void PointCloudFilter::apply_statistical_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Placeholder: apply statistical outlier removal
}

void PointCloudFilter::apply_voxel_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Placeholder: apply voxel grid downsampling
}

}  // namespace jetson_stereo_camera