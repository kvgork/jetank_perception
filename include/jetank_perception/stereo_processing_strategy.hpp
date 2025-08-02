#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <memory>
#include <string>
#include <chrono>
#include <iostream>
#include <limits>
#include <atomic>

// Include CUDA headers conditionally
#ifdef OPENCV_ENABLE_NONFREE
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#endif

namespace jetson_stereo_camera {

// Configuration structures
struct StereoConfig {
    int num_disparities = 64;
    int block_size = 15;
    int min_disparity = 0;
    int max_disparity = 64;
    int prefilter_size = 9;
    int prefilter_cap = 31;
    int texture_threshold = 10;
    int uniqueness_ratio = 10;
    int speckle_window_size = 100;
    int speckle_range = 32;
    int disp12_max_diff = 1;
    bool use_gpu = true;
};

struct PointCloudConfig {
    bool enable_voxel_filter = true;
    double voxel_leaf_size = 0.01;
    bool enable_statistical_filter = true;
    int statistical_filter_k = 50;
    double statistical_filter_stddev = 1.0;
    bool enable_range_filter = true;
    double min_range = 0.1;
    double max_range = 10.0;
    int max_threads = 4;
    int downsample_factor = 1;
    
    // Alternative names for compatibility
    bool apply_voxel_filter = true;
    bool apply_statistical_filter = true;
    bool apply_range_filter = true;
};

// =============================================================================
// BASE STEREO PROCESSING STRATEGY
// =============================================================================

class StereoProcessingStrategy {
public:
    virtual ~StereoProcessingStrategy() = default;
    
    virtual bool initialize(const StereoConfig& config, const cv::Size& image_size) = 0;
    virtual cv::Mat compute_disparity(const cv::Mat& left_rectified, const cv::Mat& right_rectified) = 0;
    virtual pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(const cv::Mat& disparity, const cv::Mat& Q_matrix) = 0;
    virtual void update_config(const StereoConfig& config) = 0;
    virtual std::string get_strategy_name() const = 0;
    
    // Performance monitoring
    virtual void get_processing_stats(double& avg_time_ms, double& fps) const {
        avg_time_ms = last_processing_time_;
        fps = (last_processing_time_ > 0) ? 1000.0 / last_processing_time_ : 0.0;
    }

protected:
    StereoConfig config_;
    double last_processing_time_ = 0.0;
};

// =============================================================================
// GPU STEREO STRATEGY IMPLEMENTATION (Header-Only)
// =============================================================================

class GPUStereoStrategy : public StereoProcessingStrategy {
private:
#ifdef OPENCV_ENABLE_NONFREE
    cv::Ptr<cv::cuda::StereoBM> stereo_matcher_;
    cv::cuda::GpuMat left_gpu_, right_gpu_, disparity_gpu_;
    cv::cuda::Stream stream_;
#endif

public:
    GPUStereoStrategy() = default;
    ~GPUStereoStrategy() = default;

    bool initialize(const StereoConfig& config, const cv::Size& image_size) override {
        try {
            config_ = config;
            
#ifdef OPENCV_ENABLE_NONFREE
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
#else
            std::cerr << "OpenCV not compiled with CUDA support. Cannot use GPU strategy." << std::endl;
            return false;
#endif
            
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in GPUStereoStrategy::initialize: " << e.what() << std::endl;
            return false;
        } catch (const std::exception& e) {
            std::cerr << "Error in GPUStereoStrategy::initialize: " << e.what() << std::endl;
            return false;
        }
    }

    cv::Mat compute_disparity(const cv::Mat& left_rectified, const cv::Mat& right_rectified) override {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
#ifdef OPENCV_ENABLE_NONFREE
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
#else
            std::cerr << "GPU processing not available" << std::endl;
            return cv::Mat();
#endif
            
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error in compute_disparity: " << e.what() << std::endl;
            return cv::Mat();
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(const cv::Mat& disparity, const cv::Mat& Q_matrix) override {
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

    void update_config(const StereoConfig& config) override {
        config_ = config;
        
#ifdef OPENCV_ENABLE_NONFREE
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
#endif
    }

    std::string get_strategy_name() const override {
        return "GPU Block Matching";
    }

private:
    void setup_gpu_memory([[maybe_unused]] const cv::Size& image_size) {
#ifdef OPENCV_ENABLE_NONFREE
        // Pre-allocate GPU memory for better performance
        left_gpu_.create(image_size, CV_8UC1);
        right_gpu_.create(image_size, CV_8UC1);
        disparity_gpu_.create(image_size, CV_16SC1);
#endif
    }

    void optimize_for_jetson() {
#ifdef OPENCV_ENABLE_NONFREE
        // Jetson-specific optimizations
        // Set optimal CUDA stream priority for real-time processing
        stream_ = cv::cuda::Stream();
        
        // Enable GPU memory pooling for better performance
        cv::cuda::setBufferPoolUsage(true);
        cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), 1024 * 1024 * 64, 2);
#endif
    }
};

// =============================================================================
// CPU STEREO STRATEGY IMPLEMENTATION (Header-Only)
// =============================================================================

class CPUStereoStrategy : public StereoProcessingStrategy {
private:
    cv::Ptr<cv::StereoBM> stereo_matcher_;

public:
    CPUStereoStrategy() = default;
    ~CPUStereoStrategy() = default;

    bool initialize(const StereoConfig& config, const cv::Size& image_size) override {
        try {
            (void)image_size; // Suppress unused parameter warning
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

    cv::Mat compute_disparity(const cv::Mat& left_rectified, const cv::Mat& right_rectified) override {
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(const cv::Mat& disparity, const cv::Mat& Q_matrix) override {
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

    void update_config(const StereoConfig& config) override {
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

    std::string get_strategy_name() const override {
        return "CPU Block Matching";
    }
};

// =============================================================================
// SGBM STEREO STRATEGY IMPLEMENTATION (Header-Only)
// =============================================================================

class SGBMStereoStrategy : public StereoProcessingStrategy {
private:
    bool use_gpu_;
    cv::Ptr<cv::StereoSGBM> cpu_stereo_matcher_;
    
#ifdef OPENCV_ENABLE_NONFREE
    cv::Ptr<cv::cuda::StereoSGM> gpu_stereo_matcher_;
    cv::cuda::GpuMat left_gpu_, right_gpu_, disparity_gpu_;
#endif

public:
    explicit SGBMStereoStrategy(bool use_gpu = false) : use_gpu_(use_gpu) {}
    ~SGBMStereoStrategy() = default;

    bool initialize(const StereoConfig& config, const cv::Size& image_size) override {
        try {
            config_ = config;
            
            if (use_gpu_) {
#ifdef OPENCV_ENABLE_NONFREE
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
#else
                std::cerr << "OpenCV not compiled with CUDA support, using CPU SGBM" << std::endl;
                use_gpu_ = false;
#endif
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

    cv::Mat compute_disparity(const cv::Mat& left_rectified, const cv::Mat& right_rectified) override {
        auto start = std::chrono::high_resolution_clock::now();
        
        cv::Mat disparity;
        try {
            if (use_gpu_) {
#ifdef OPENCV_ENABLE_NONFREE
                if (gpu_stereo_matcher_) {
                    left_gpu_.upload(left_rectified);
                    right_gpu_.upload(right_rectified);
                    
                    gpu_stereo_matcher_->compute(left_gpu_, right_gpu_, disparity_gpu_);
                    disparity_gpu_.download(disparity);
                }
#endif
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(const cv::Mat& disparity, const cv::Mat& Q_matrix) override {
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

    void update_config(const StereoConfig& config) override {
        config_ = config;
        
        if (use_gpu_) {
#ifdef OPENCV_ENABLE_NONFREE
            // GPU SGBM has limited parameter updating
            // Would need to recreate matcher for full parameter changes
#endif
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

    std::string get_strategy_name() const override {
        return use_gpu_ ? "GPU Semi-Global Block Matching" : "CPU Semi-Global Block Matching";
    }
};

// =============================================================================
// CONCRETE STRATEGY CLASSES (for compatibility with your existing code)
// =============================================================================

class CPUBlockMatchingStereo : public CPUStereoStrategy {
public:
    CPUBlockMatchingStereo() = default;
    std::string get_strategy_name() const override {
        return "CPU Block Matching Stereo";
    }
};

class GPUSGBMStereo : public SGBMStereoStrategy {
public:
    GPUSGBMStereo() : SGBMStereoStrategy(true) {}
    std::string get_strategy_name() const override {
        return "GPU SGBM Stereo";
    }
};

class CPUSGBMStereo : public SGBMStereoStrategy {
public:
    CPUSGBMStereo() : SGBMStereoStrategy(false) {}
    std::string get_strategy_name() const override {
        return "CPU SGBM Stereo";
    }
};

// =============================================================================
// STEREO PROCESSING FACTORY (Header-Only)
// =============================================================================

class StereoProcessingFactory {
public:
    enum class StrategyType {
        GPU_BM,
        CPU_BM,
        GPU_SGBM,
        CPU_SGBM
    };

    static std::unique_ptr<StereoProcessingStrategy> create_strategy(StrategyType type) {
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

    static StrategyType get_optimal_strategy_for_platform() {
        // Check if CUDA is available
#ifdef OPENCV_ENABLE_NONFREE
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            // For Jetson platforms, GPU BM is typically faster for real-time applications
            return StrategyType::GPU_BM;
        }
#endif
        // Fall back to CPU SGBM for better quality on CPU-only systems
        return StrategyType::CPU_SGBM;
    }
};

// =============================================================================
// POINT CLOUD FILTER (Header-Only)
// =============================================================================

class PointCloudFilter {
private:
    PointCloudConfig config_;

public:
    explicit PointCloudFilter(const PointCloudConfig& config) : config_(config) {}

    void filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        if (!cloud || cloud->empty()) {
            return;
        }
        
        try {
            // Apply range filter first to remove obviously bad points
            if (config_.enable_range_filter || config_.apply_range_filter) {
                apply_range_filter(cloud);
            }
            
            // Apply voxel filter to downsample
            if (config_.enable_voxel_filter || config_.apply_voxel_filter) {
                apply_voxel_filter(cloud);
            }
            
            // Apply statistical filter to remove outliers
            if (config_.enable_statistical_filter || config_.apply_statistical_filter) {
                apply_statistical_filter(cloud);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error in point cloud filtering: " << e.what() << std::endl;
        }
    }

    void update_config(const PointCloudConfig& config) {
        config_ = config;
    }

private:
    void apply_range_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(config_.min_range, config_.max_range);
        pass.filter(*cloud);
    }

    void apply_statistical_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(config_.statistical_filter_k);
        sor.setStddevMulThresh(config_.statistical_filter_stddev);
        sor.filter(*cloud);
    }

    void apply_voxel_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        pcl::VoxelGrid<pcl::PointXYZ> vox;
        vox.setInputCloud(cloud);
        vox.setLeafSize(config_.voxel_leaf_size, config_.voxel_leaf_size, config_.voxel_leaf_size);
        vox.filter(*cloud);
    }
};

} // namespace jetson_stereo_camera