// src/stereo_processing_strategy.cpp
#include "jetank_perception/stereo_processing_strategy.hpp"
#include <chrono>
#include <iostream>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace jetson_stereo_camera {

// =============================================================================
// GPUStereoStrategy Implementation
// =============================================================================

GPUStereoStrategy::GPUStereoStrategy() : last_processing_time_(0.0) {}

GPUStereoStrategy::~GPUStereoStrategy() = default;

bool GPUStereoStrategy::initialize(const StereoConfig& config, const cv::Size& image_size) {
    try {
        config_ = config;
        
        // Check if CUDA is available
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
            std::cerr << "No CUDA-enabled devices found. Cannot use GPU strategy." << std::endl;
            return false;
        }
        
        // Create GPU stereo matcher
        stereo_matcher_ = cv::cuda::createStereoBM(config_.max_disparity, config_.block_size);
        
        // Configure stereo matcher parameters
        stereo_matcher_->setPreFilterSize(config_.prefilter_size);
        stereo_matcher_->setPreFilterCap(config_.prefilter_cap);
        stereo_matcher_->setTextureThreshold(config_.texture_threshold);
        stereo_matcher_->setUniquenessRatio(config_.uniqueness_ratio);
        stereo_matcher_->setSpeckleWindowSize(config_.speckle_window_size);
        stereo_matcher_->setSpeckleRange(config_.speckle_range);
        stereo_matcher_->setDisp12MaxDiff(config_.disp12_max_diff);
        
        // Setup GPU memory
        setup_gpu_memory(image_size);
        
        // Apply Jetson-specific optimizations
        optimize_for_jetson();
        
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in GPUStereoStrategy::initialize: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error in GPUStereoStrategy::initialize: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat GPUStereoStrategy::compute_disparity(const cv::Mat& left_rectified, 
                                           const cv::Mat& right_rectified) {
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        // Upload images to GPU
        left_gpu_.upload(left_rectified);
        right_gpu_.upload(right_rectified);
        
        // Compute disparity on GPU
        stereo_matcher_->compute(left_gpu_, right_gpu_, disparity_gpu_, stream_);
        
        // Download result from GPU
        cv::Mat disparity;
        disparity_gpu_.download(disparity);
        
        // Wait for GPU operations to complete
        stream_.waitForCompletion();
        
        auto end = std::chrono::high_resolution_clock::now();
        last_processing_time_ = std::chrono::duration<double, std::milli>(end - start).count();
        
        return disparity;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in compute_disparity: " << e.what() << std::endl;
        return cv::Mat();
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr GPUStereoStrategy::generate_pointcloud(
    const cv::Mat& disparity, const cv::Mat& Q_matrix) {
    
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    
    try {
        // Convert disparity to 3D points
        cv::Mat points_3d;
        cv::reprojectImageTo3D(disparity, points_3d, Q_matrix);
        
        // Convert to PCL point cloud
        cloud->width = disparity.cols;
        cloud->height = disparity.rows;
        cloud->is_dense = false;
        cloud->points.resize(cloud->width * cloud->height);
        
        int point_idx = 0;
        for (int y = 0; y < disparity.rows; ++y) {
            for (int x = 0; x < disparity.cols; ++x) {
                const cv::Vec3f& pt = points_3d.at<cv::Vec3f>(y, x);
                
                pcl::PointXYZ& pcl_pt = cloud->points[point_idx++];
                pcl_pt.x = pt[0];
                pcl_pt.y = pt[1];
                pcl_pt.z = pt[2];
                
                // Check for invalid points
                if (!std::isfinite(pcl_pt.x) || !std::isfinite(pcl_pt.y) || !std::isfinite(pcl_pt.z)) {
                    pcl_pt.x = pcl_pt.y = pcl_pt.z = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating point cloud: " << e.what() << std::endl;
        return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }
    
    return cloud;
}

void GPUStereoStrategy::update_config(const StereoConfig& config) {
    config_ = config;
    
    if (stereo_matcher_) {
        stereo_matcher_->setNumDisparities(config_.num_disparities);
        stereo_matcher_->setBlockSize(config_.block_size);
        stereo_matcher_->setPreFilterSize(config_.prefilter_size);
        stereo_matcher_->setPreFilterCap(config_.prefilter_cap);
        stereo_matcher_->setTextureThreshold(config_.texture_threshold);
        stereo_matcher_->setUniquenessRatio(config_.uniqueness_ratio);
        stereo_matcher_->setSpeckleWindowSize(config_.speckle_window_size);
        stereo_matcher_->setSpeckleRange(config_.speckle_range);
        stereo_matcher_->setDisp12MaxDiff(config_.disp12_max_diff);
    }
}

void GPUStereoStrategy::setup_gpu_memory(const cv::Size& image_size) {
    // Pre-allocate GPU memory for better performance
    left_gpu_.create(image_size, CV_8UC1);
    right_gpu_.create(image_size, CV_8UC1);
    disparity_gpu_.create(image_size, CV_16SC1);
}

void GPUStereoStrategy::optimize_for_jetson() {
    // Jetson-specific optimizations
    // Set optimal CUDA stream priority for real-time processing
    stream_ = cv::cuda::Stream();
    
    // Enable GPU memory pooling for better performance
    cv::cuda::setBufferPoolUsage(true);
    cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), 1024 * 1024 * 64, 2);
}

// =============================================================================
// CPUStereoStrategy Implementation
// =============================================================================

CPUStereoStrategy::CPUStereoStrategy() : last_processing_time_(0.0) {}

bool CPUStereoStrategy::initialize(const StereoConfig& config, const cv::Size& image_size) {
    try {
        config_ = config;
        
        // Create CPU stereo matcher
        stereo_matcher_ = cv::StereoBM::create(config_.num_disparities, config_.block_size);
        
        // Configure parameters
        stereo_matcher_->setPreFilterSize(config_.prefilter_size);
        stereo_matcher_->setPreFilterCap(config_.prefilter_cap);
        stereo_matcher_->setTextureThreshold(config_.texture_threshold);
        stereo_matcher_->setUniquenessRatio(config_.uniqueness_ratio);
        stereo_matcher_->setSpeckleWindowSize(config_.speckle_window_size);
        stereo_matcher_->setSpeckleRange(config_.speckle_range);
        stereo_matcher_->setDisp12MaxDiff(config_.disp12_max_diff);
        
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in CPUStereoStrategy::initialize: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat CPUStereoStrategy::compute_disparity(const cv::Mat& left_rectified, 
                                           const cv::Mat& right_rectified) {
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat disparity;
    try {
        stereo_matcher_->compute(left_rectified, right_rectified, disparity);
        
        auto end = std::chrono::high_resolution_clock::now();
        last_processing_time_ = std::chrono::duration<double, std::milli>(end - start).count();
        
        return disparity;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in CPU compute_disparity: " << e.what() << std::endl;
        return cv::Mat();
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CPUStereoStrategy::generate_pointcloud(
    const cv::Mat& disparity, const cv::Mat& Q_matrix) {
    
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    
    try {
        cv::Mat points_3d;
        cv::reprojectImageTo3D(disparity, points_3d, Q_matrix);
        
        cloud->width = disparity.cols;
        cloud->height = disparity.rows;
        cloud->is_dense = false;
        cloud->points.resize(cloud->width * cloud->height);
        
        int point_idx = 0;
        for (int y = 0; y < disparity.rows; ++y) {
            for (int x = 0; x < disparity.cols; ++x) {
                const cv::Vec3f& pt = points_3d.at<cv::Vec3f>(y, x);
                
                pcl::PointXYZ& pcl_pt = cloud->points[point_idx++];
                pcl_pt.x = pt[0];
                pcl_pt.y = pt[1];
                pcl_pt.z = pt[2];
                
                if (!std::isfinite(pcl_pt.x) || !std::isfinite(pcl_pt.y) || !std::isfinite(pcl_pt.z)) {
                    pcl_pt.x = pcl_pt.y = pcl_pt.z = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating point cloud: " << e.what() << std::endl;
        return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }
    
    return cloud;
}

void CPUStereoStrategy::update_config(const StereoConfig& config) {
    config_ = config;
    
    if (stereo_matcher_) {
        stereo_matcher_->setNumDisparities(config_.num_disparities);
        stereo_matcher_->setBlockSize(config_.block_size);
        stereo_matcher_->setPreFilterSize(config_.prefilter_size);
        stereo_matcher_->setPreFilterCap(config_.prefilter_cap);
        stereo_matcher_->setTextureThreshold(config_.texture_threshold);
        stereo_matcher_->setUniquenessRatio(config_.uniqueness_ratio);
        stereo_matcher_->setSpeckleWindowSize(config_.speckle_window_size);
        stereo_matcher_->setSpeckleRange(config_.speckle_range);
        stereo_matcher_->setDisp12MaxDiff(config_.disp12_max_diff);
    }
}

// =============================================================================
// SGBMStereoStrategy Implementation
// =============================================================================

SGBMStereoStrategy::SGBMStereoStrategy(bool use_gpu) : use_gpu_(use_gpu), last_processing_time_(0.0) {}

bool SGBMStereoStrategy::initialize(const StereoConfig& config, const cv::Size& image_size) {
    try {
        config_ = config;
        
        if (use_gpu_) {
            // Check CUDA availability
            if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
                std::cerr << "No CUDA devices available, falling back to CPU SGBM" << std::endl;
                use_gpu_ = false;
            } else {
                gpu_stereo_matcher_ = cv::cuda::createStereoSGM(config_.min_disparity, 
                                                               config_.num_disparities, 
                                                               config_.block_size);
                
                // Setup GPU memory
                left_gpu_.create(image_size, CV_8UC1);
                right_gpu_.create(image_size, CV_8UC1);
                disparity_gpu_.create(image_size, CV_16SC1);
            }
        }
        
        if (!use_gpu_) {
            cpu_stereo_matcher_ = cv::StereoSGBM::create(config_.min_disparity,
                                                        config_.num_disparities,
                                                        config_.block_size);
            
            // Configure SGBM parameters
            cpu_stereo_matcher_->setP1(8 * config_.block_size * config_.block_size);
            cpu_stereo_matcher_->setP2(32 * config_.block_size * config_.block_size);
            cpu_stereo_matcher_->setPreFilterCap(config_.prefilter_cap);
            cpu_stereo_matcher_->setUniquenessRatio(config_.uniqueness_ratio);
            cpu_stereo_matcher_->setSpeckleWindowSize(config_.speckle_window_size);
            cpu_stereo_matcher_->setSpeckleRange(config_.speckle_range);
            cpu_stereo_matcher_->setDisp12MaxDiff(config_.disp12_max_diff);
        }
        
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in SGBMStereoStrategy::initialize: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat SGBMStereoStrategy::compute_disparity(const cv::Mat& left_rectified, 
                                            const cv::Mat& right_rectified) {
    auto start = std::chrono::high_resolution_clock::now();
    
    cv::Mat disparity;
    try {
        if (use_gpu_ && gpu_stereo_matcher_) {
            left_gpu_.upload(left_rectified);
            right_gpu_.upload(right_rectified);
            
            gpu_stereo_matcher_->compute(left_gpu_, right_gpu_, disparity_gpu_);
            disparity_gpu_.download(disparity);
        } else if (cpu_stereo_matcher_) {
            cpu_stereo_matcher_->compute(left_rectified, right_rectified, disparity);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        last_processing_time_ = std::chrono::duration<double, std::milli>(end - start).count();
        
        return disparity;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in SGBM compute_disparity: " << e.what() << std::endl;
        return cv::Mat();
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr SGBMStereoStrategy::generate_pointcloud(
    const cv::Mat& disparity, const cv::Mat& Q_matrix) {
    
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    
    try {
        cv::Mat points_3d;
        cv::reprojectImageTo3D(disparity, points_3d, Q_matrix);
        
        cloud->width = disparity.cols;
        cloud->height = disparity.rows;
        cloud->is_dense = false;
        cloud->points.resize(cloud->width * cloud->height);
        
        int point_idx = 0;
        for (int y = 0; y < disparity.rows; ++y) {
            for (int x = 0; x < disparity.cols; ++x) {
                const cv::Vec3f& pt = points_3d.at<cv::Vec3f>(y, x);
                
                pcl::PointXYZ& pcl_pt = cloud->points[point_idx++];
                pcl_pt.x = pt[0];
                pcl_pt.y = pt[1];
                pcl_pt.z = pt[2];
                
                if (!std::isfinite(pcl_pt.x) || !std::isfinite(pcl_pt.y) || !std::isfinite(pcl_pt.z)) {
                    pcl_pt.x = pcl_pt.y = pcl_pt.z = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating point cloud: " << e.what() << std::endl;
        return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }
    
    return cloud;
}

void SGBMStereoStrategy::update_config(const StereoConfig& config) {
    config_ = config;
    
    if (use_gpu_ && gpu_stereo_matcher_) {
        // GPU SGBM has limited parameter updating
        // Would need to recreate matcher for full parameter changes
    } else if (cpu_stereo_matcher_) {
        cpu_stereo_matcher_->setMinDisparity(config_.min_disparity);
        cpu_stereo_matcher_->setNumDisparities(config_.num_disparities);
        cpu_stereo_matcher_->setBlockSize(config_.block_size);
        cpu_stereo_matcher_->setP1(8 * config_.block_size * config_.block_size);
        cpu_stereo_matcher_->setP2(32 * config_.block_size * config_.block_size);
        cpu_stereo_matcher_->setPreFilterCap(config_.prefilter_cap);
        cpu_stereo_matcher_->setUniquenessRatio(config_.uniqueness_ratio);
        cpu_stereo_matcher_->setSpeckleWindowSize(config_.speckle_window_size);
        cpu_stereo_matcher_->setSpeckleRange(config_.speckle_range);
        cpu_stereo_matcher_->setDisp12MaxDiff(config_.disp12_max_diff);
    }
}

// =============================================================================
// StereoProcessingFactory Implementation
// =============================================================================

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
    // Check if CUDA is available
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        // For Jetson platforms, GPU BM is typically faster for real-time applications
        return StrategyType::GPU_BM;
    } else {
        // Fall back to CPU SGBM for better quality on CPU-only systems
        return StrategyType::CPU_SGBM;
    }
}

// =============================================================================
// PointCloudFilter Implementation
// =============================================================================

PointCloudFilter::PointCloudFilter(const PointCloudConfig& config) : config_(config) {}

void PointCloudFilter::filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    if (!cloud || cloud->empty()) {
        return;
    }
    
    try {
        // Apply range filter first to remove obviously bad points
        apply_range_filter(cloud);
        
        // Apply voxel filter to downsample
        if (config_.apply_voxel_filter) {
            apply_voxel_filter(cloud);
        }
        
        // Apply statistical filter to remove outliers
        if (config_.apply_statistical_filter) {
            apply_statistical_filter(cloud);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in point cloud filtering: " << e.what() << std::endl;
    }
}

void PointCloudFilter::update_config(const PointCloudConfig& config) {
    config_ = config;
}

void PointCloudFilter::apply_range_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(config_.min_range, config_.max_range);
    pass.filter(*cloud);
}

void PointCloudFilter::apply_statistical_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(config_.statistical_filter_k);
    sor.setStddevMulThresh(config_.statistical_filter_stddev);
    sor.filter(*cloud);
}

void PointCloudFilter::apply_voxel_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::VoxelGrid<pcl::PointXYZ> vox;
    vox.setInputCloud(cloud);
    vox.setLeafSize(config_.voxel_leaf_size, config_.voxel_leaf_size, config_.voxel_leaf_size);
    vox.filter(*cloud);
}

} // namespace jetson_stereo_camera