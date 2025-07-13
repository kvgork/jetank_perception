#include "jetank_perception/stereo_processing_strategy.hpp"

#include <iostream>
#include <chrono>

namespace jetson_stereo_camera {
// -------------------- GPUStereoStrategy --------------------

GPUStereoStrategy::GPUStereoStrategy()
    : last_processing_time_(0.0)
{}

GPUStereoStrategy::~GPUStereoStrategy() = default;

bool GPUStereoStrategy::initialize(const StereoConfig& config, const cv::Size& image_size) {
    config_ = config;
    try {
        stereo_matcher_ = cv::cuda::createStereoBM(config.num_disparities, config.block_size);
        setup_gpu_memory(image_size);
        optimize_for_jetson();
    } catch (const cv::Exception& e) {
        std::cerr << "GPU StereoBM initialization failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

cv::Mat GPUStereoStrategy::compute_disparity(const cv::Mat& left_rectified, const cv::Mat& right_rectified) {
    auto start = std::chrono::high_resolution_clock::now();
    
    left_gpu_.upload(left_rectified);
    right_gpu_.upload(right_rectified);
    
    stereo_matcher_->compute(left_gpu_, right_gpu_, disparity_gpu_);
    
    cv::Mat disparity;
    disparity_gpu_.download(disparity);
    
    auto end = std::chrono::high_resolution_clock::now();
    last_processing_time_ = std::chrono::duration<double, std::milli>(end - start).count();
    
    return disparity;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr GPUStereoStrategy::generate_pointcloud(
    const cv::Mat& disparity, const cv::Mat& Q_matrix) {
    // Placeholder: Implement disparity-to-pointcloud conversion here
    return pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
}

void GPUStereoStrategy::update_config(const StereoConfig& config) {
    config_ = config;
}

void GPUStereoStrategy::setup_gpu_memory(const cv::Size& image_size) {
    left_gpu_.create(image_size, CV_8UC1);
    right_gpu_.create(image_size, CV_8UC1);
    disparity_gpu_.create(image_size, CV_16S);
}

void GPUStereoStrategy::optimize_for_jetson() {
    // Jetson-specific optimizations (optional)
}

// -------------------- CPUStereoStrategy --------------------

CPUStereoStrategy::CPUStereoStrategy()
    : last_processing_time_(0.0)
{}

bool CPUStereoStrategy::initialize(const StereoConfig& config, const cv::Size& image_size) {
    config_ = config;
    try {
        stereo_matcher_ = cv::StereoBM::create(config.num_disparities, config.block_size);
    } catch (const cv::Exception& e) {
        std::cerr << "CPU StereoBM initialization failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

cv::Mat CPUStereoStrategy::compute_disparity(const cv::Mat& left_rectified, const cv::Mat& right_rectified) {
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat disparity;
    stereo_matcher_->compute(left_rectified, right_rectified, disparity);
    
    auto end = std::chrono::high_resolution_clock::now();
    last_processing_time_ = std::chrono::duration<double, std::milli>(end - start).count();
    
    return disparity;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CPUStereoStrategy::generate_pointcloud(
    const cv::Mat& disparity, const cv::Mat& Q_matrix) {
    // Placeholder: Implement disparity-to-pointcloud conversion here
    return pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
}

void CPUStereoStrategy::update_config(const StereoConfig& config) {
    config_ = config;
}

// -------------------- SGBMStereoStrategy --------------------

SGBMStereoStrategy::SGBMStereoStrategy(bool use_gpu)
    : use_gpu_(use_gpu), last_processing_time_(0.0)
{}

bool SGBMStereoStrategy::initialize(const StereoConfig& config, const cv::Size& image_size) {
    config_ = config;
    try {
        if (use_gpu_) {
            gpu_stereo_matcher_ = cv::cuda::createStereoSGM(
                config.min_disparity, config.num_disparities, config.P1, config.P2);
            // Set additional parameters here as needed
        } else {
            cpu_stereo_matcher_ = cv::StereoSGBM::create(
                config.min_disparity, config.num_disparities, config.block_size);
            // Set additional parameters here as needed
        }
    } catch (const cv::Exception& e) {
        std::cerr << "SGBM initialization failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

cv::Mat SGBMStereoStrategy::compute_disparity(const cv::Mat& left_rectified, const cv::Mat& right_rectified) {
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat disparity;
    if (use_gpu_) {
        left_gpu_.upload(left_rectified);
        right_gpu_.upload(right_rectified);
        gpu_stereo_matcher_->compute(left_gpu_, right_gpu_, disparity_gpu_);
        disparity_gpu_.download(disparity);
    } else {
        cpu_stereo_matcher_->compute(left_rectified, right_rectified, disparity);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    last_processing_time_ = std::chrono::duration<double, std::milli>(end - start).count();
    
    return disparity;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr SGBMStereoStrategy::generate_pointcloud(
    const cv::Mat& disparity, const cv::Mat& Q_matrix) {
    // Placeholder: Implement disparity-to-pointcloud conversion here
    return pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
}

void SGBMStereoStrategy::update_config(const StereoConfig& config) {
    config_ = config;
}

std::unique_ptr<StereoProcessingStrategy> StereoProcessingFactory::create_strategy(StrategyType type) {
    switch (type) {
        case StrategyType::GPU_BM:
            return std::make_unique<GPUStereoStrategy>();
        case StrategyType::CPU_BM:
            return std::make_unique<CPUStereoStrategy>();
        case StrategyType::GPU_SGBM:
            return std::make_unique<SGBMStereoStrategy>(true);
        case StrategyType::CPU_SGBM:
            return std::make_unique<SGBMStereoStrategy>(false);
        default:
            return nullptr;
    }
}

StereoProcessingFactory::StrategyType StereoProcessingFactory::get_optimal_strategy_for_platform() {
#ifdef __JETSON__
    return StrategyType::GPU_SGBM;
#else
    return StrategyType::CPU_SGBM;
#endif
}

}