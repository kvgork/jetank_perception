#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>

namespace jetson_stereo_camera {

struct StereoConfig {
    int num_disparities;
    int block_size;
    int min_disparity;
    int max_disparity;
    int prefilter_size;
    int prefilter_cap;
    int texture_threshold;
    int uniqueness_ratio;
    int speckle_window_size;
    int speckle_range;
    int disp12_max_diff;
    bool use_gpu;
    
    StereoConfig() : num_disparities(64), block_size(15), min_disparity(0), max_disparity(64),
                     prefilter_size(9), prefilter_cap(31), texture_threshold(10),
                     uniqueness_ratio(15), speckle_window_size(100), speckle_range(32),
                     disp12_max_diff(1), use_gpu(true) {}
};

struct PointCloudConfig {
    double voxel_leaf_size;
    int statistical_filter_k;
    double statistical_filter_stddev;
    double min_range;
    double max_range;
    bool apply_statistical_filter;
    bool apply_voxel_filter;
    
    PointCloudConfig() : voxel_leaf_size(0.01), statistical_filter_k(50),
                        statistical_filter_stddev(1.0), min_range(0.1), max_range(5.0),
                        apply_statistical_filter(true), apply_voxel_filter(true) {}
};

class StereoProcessingStrategy {
public:
    virtual ~StereoProcessingStrategy() = default;
    
    // Core processing methods
    virtual bool initialize(const StereoConfig& config, const cv::Size& image_size) = 0;
    virtual cv::Mat compute_disparity(const cv::Mat& left_rectified, 
                                     const cv::Mat& right_rectified) = 0;
    
    // Point cloud generation
    virtual pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(
        const cv::Mat& disparity, const cv::Mat& Q_matrix) = 0;
    
    // Configuration
    virtual void update_config(const StereoConfig& config) = 0;
    virtual StereoConfig get_config() const = 0;
    virtual std::string get_strategy_name() const = 0;
    
    // Performance metrics
    virtual double get_last_processing_time() const = 0;
    virtual bool supports_gpu() const = 0;
};

// GPU-accelerated stereo processing strategy
class GPUStereoStrategy : public StereoProcessingStrategy {
public:
    GPUStereoStrategy();
    ~GPUStereoStrategy() override;
    
    bool initialize(const StereoConfig& config, const cv::Size& image_size) override;
    cv::Mat compute_disparity(const cv::Mat& left_rectified, 
                             const cv::Mat& right_rectified) override;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(
        const cv::Mat& disparity, const cv::Mat& Q_matrix) override;
    
    void update_config(const StereoConfig& config) override;
    StereoConfig get_config() const override { return config_; }
    std::string get_strategy_name() const override { return "GPU Stereo BM"; }
    
    double get_last_processing_time() const override { return last_processing_time_; }
    bool supports_gpu() const override { return true; }

private:
    StereoConfig config_;
    cv::Ptr<cv::cuda::StereoBM> stereo_matcher_;
    cv::cuda::GpuMat left_gpu_, right_gpu_, disparity_gpu_;
    cv::cuda::Stream stream_;
    double last_processing_time_;
    
    void setup_gpu_memory(const cv::Size& image_size);
    void optimize_for_jetson();
};

// CPU-based stereo processing strategy
class CPUStereoStrategy : public StereoProcessingStrategy {
public:
    CPUStereoStrategy();
    ~CPUStereoStrategy() override = default;
    
    bool initialize(const StereoConfig& config, const cv::Size& image_size) override;
    cv::Mat compute_disparity(const cv::Mat& left_rectified, 
                             const cv::Mat& right_rectified) override;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(
        const cv::Mat& disparity, const cv::Mat& Q_matrix) override;
    
    void update_config(const StereoConfig& config) override;
    StereoConfig get_config() const override { return config_; }
    std::string get_strategy_name() const override { return "CPU Stereo BM"; }
    
    double get_last_processing_time() const override { return last_processing_time_; }
    bool supports_gpu() const override { return false; }

private:
    StereoConfig config_;
    cv::Ptr<cv::StereoBM> stereo_matcher_;
    double last_processing_time_;
};

// SGBM strategy for better quality (CPU/GPU)
class SGBMStereoStrategy : public StereoProcessingStrategy {
public:
    SGBMStereoStrategy(bool use_gpu = true);
    ~SGBMStereoStrategy() override = default;
    
    bool initialize(const StereoConfig& config, const cv::Size& image_size) override;
    cv::Mat compute_disparity(const cv::Mat& left_rectified, 
                             const cv::Mat& right_rectified) override;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(
        const cv::Mat& disparity, const cv::Mat& Q_matrix) override;
    
    void update_config(const StereoConfig& config) override;
    StereoConfig get_config() const override { return config_; }
    std::string get_strategy_name() const override { 
        return use_gpu_ ? "GPU SGBM" : "CPU SGBM"; 
    }
    
    double get_last_processing_time() const override { return last_processing_time_; }
    bool supports_gpu() const override { return use_gpu_; }

private:
    StereoConfig config_;
    bool use_gpu_;
    cv::Ptr<cv::StereoSGBM> cpu_stereo_matcher_;
    cv::Ptr<cv::cuda::StereoSGM> gpu_stereo_matcher_;
    cv::cuda::GpuMat left_gpu_, right_gpu_, disparity_gpu_;
    double last_processing_time_;
};

// Factory for creating stereo processing strategies
class StereoProcessingFactory {
public:
    enum class StrategyType {
        GPU_BM,
        CPU_BM,
        GPU_SGBM,
        CPU_SGBM
    };
    
    static std::unique_ptr<StereoProcessingStrategy> create_strategy(StrategyType type);
    static StrategyType get_optimal_strategy_for_platform();
};

// Point cloud filtering strategy
class PointCloudFilter {
public:
    explicit PointCloudFilter(const PointCloudConfig& config);
    
    void filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void update_config(const PointCloudConfig& config);
    
private:
    PointCloudConfig config_;
    
    void apply_range_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void apply_statistical_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void apply_voxel_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
};

}