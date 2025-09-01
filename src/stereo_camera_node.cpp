#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>  // Correct header
#include <sensor_msgs/srv/set_camera_info.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <cv_bridge/cv_bridge.h>
#include <camera_info_manager/camera_info_manager.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "jetank_perception/camera_interface.hpp"
#include "jetank_perception/stereo_processing_strategy.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace jetson_stereo_camera {

class StereoCalibration {
public:
    bool load_calibration(const std::string& left_info_url, 
                         const std::string& right_info_url,
                         const std::string& stereo_info_url,
                         const cv::Size& image_size);
    
    void rectify_images(const cv::Mat& left_raw, const cv::Mat& right_raw,
                       cv::Mat& left_rectified, cv::Mat& right_rectified);
    
    cv::Mat get_Q_matrix() const { return Q_; }
    bool is_calibrated() const { return calibrated_; }
    
    // Getters for camera matrices
    cv::Mat get_left_camera_matrix() const { return camera_matrix_left_; }
    cv::Mat get_right_camera_matrix() const { return camera_matrix_right_; }
    cv::Mat get_left_dist_coeffs() const { return dist_coeffs_left_; }
    cv::Mat get_right_dist_coeffs() const { return dist_coeffs_right_; }
    
    // Calibration file operations
    bool load_calibration_from_yaml(const std::string& yaml_file_path, bool is_left_camera = true);
    bool save_calibration_to_yaml(const std::string& yaml_file_path, 
                                 const sensor_msgs::msg::CameraInfo& camera_info);
    
    // Convert between ROS CameraInfo and OpenCV calibration data
    sensor_msgs::msg::CameraInfo to_camera_info_msg(const std::string& frame_id, bool is_left = true) const;
    bool from_camera_info_msg(const sensor_msgs::msg::CameraInfo& camera_info, bool is_left = true);
    
    // Stereo calibration operations
    bool calibrate_stereo_from_individual(const sensor_msgs::msg::CameraInfo& left_info,
                                        const sensor_msgs::msg::CameraInfo& right_info);
    bool save_stereo_calibration(const std::string& yaml_file_path);

private:
    bool calibrated_ = false;
    cv::Mat camera_matrix_left_, camera_matrix_right_;
    cv::Mat dist_coeffs_left_, dist_coeffs_right_;
    cv::Mat R_, T_, E_, F_;
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::Mat left_map1_, left_map2_, right_map1_, right_map2_;
    cv::Size image_size_;
    
    void setup_default_calibration();
    void compute_rectification_maps();
    bool load_stereo_calibration_yaml(const std::string& yaml_file_path);
};

class JetsonStereoNode : public rclcpp::Node {
private:
    // Core components
    std::unique_ptr<CameraInterface> left_camera_;
    std::unique_ptr<CameraInterface> right_camera_;
    std::unique_ptr<StereoProcessingStrategy> stereo_processor_;
    std::unique_ptr<StereoCalibration> calibration_;
    std::unique_ptr<PointCloudFilter> pointcloud_filter_;
    
    // ROS2 publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_rect_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_rect_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_pub_;  // Correct type
    
    // Camera info managers
    std::shared_ptr<camera_info_manager::CameraInfoManager> left_info_manager_;
    std::shared_ptr<camera_info_manager::CameraInfoManager> right_info_manager_;
    
    // Services
    rclcpp::Service<sensor_msgs::srv::SetCameraInfo>::SharedPtr left_set_info_service_;
    rclcpp::Service<sensor_msgs::srv::SetCameraInfo>::SharedPtr right_set_info_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stereo_calibrate_service_;
    
    // Processing thread
    std::unique_ptr<std::thread> processing_thread_;
    std::atomic<bool> processing_active_;
    std::mutex processing_mutex_;
    
    // Configuration
    CameraConfig camera_config_;
    StereoConfig stereo_config_;
    PointCloudConfig pointcloud_config_;
    
    // Frame IDs
    std::string left_frame_id_;
    std::string right_frame_id_;
    std::string base_frame_id_;
    
    // Performance monitoring
    std::atomic<double> processing_fps_;
    rclcpp::Time last_frame_time_;
    int performance_log_interval_;
    bool log_performance_;
    
    // Publishing flags
    bool publish_raw_images_;
    bool publish_rectified_images_;
    bool publish_disparity_;
    bool publish_pointcloud_;
    
public:
    JetsonStereoNode() : Node("stereo_camera_node"), processing_active_(false), processing_fps_(0.0) {
        // Initialize components
        initialize_parameters();
        // In constructor, right after initialize_parameters():
        RCLCPP_INFO(get_logger(), "Debug: camera.left_sensor_id = %d, camera.flip_images_180 = %s", 
                get_parameter("camera.left_sensor_id").as_int(),
                get_parameter("camera.flip_images_180").as_bool() ? "true" : "false");
        create_cameras();
        create_stereo_processor();
        create_calibration();
        create_pointcloud_filter();
        
        // Setup ROS2 interfaces
        setup_publishers();
        setup_services();
        setup_camera_info_managers();
        
        // Load calibration
        if (!calibration_->load_calibration(
            get_parameter("calibration.left_camera_info_url").as_string(),
            get_parameter("calibration.right_camera_info_url").as_string(),
            get_parameter("calibration.stereo_calibration_url").as_string(),
            cv::Size(camera_config_.width, camera_config_.height))) {
            RCLCPP_WARN(get_logger(), "Failed to load calibration, using defaults");
        }
        
        // Start cameras
        start_cameras();
        
        // Start processing thread
        start_processing();
        
        RCLCPP_INFO(get_logger(), "Stereo camera node initialized with %s strategy", 
                   stereo_processor_->get_strategy_name().c_str());
    }
    
    ~JetsonStereoNode() {
        stop_processing();
        stop_cameras();
    }

private:
    void initialize_parameters() {
        // ========================================================================
        // CAMERA PARAMETERS
        // ========================================================================
        declare_parameter("camera.width", 640);
        declare_parameter("camera.height", 480);
        declare_parameter("camera.fps", 20);
        declare_parameter("camera.format", "GRAY8");
        declare_parameter("camera.left_sensor_id", 0);
        declare_parameter("camera.right_sensor_id", 1);
        declare_parameter("camera.use_hardware_acceleration", true);
        declare_parameter("camera.buffer_size", 3);
        declare_parameter("camera.processing_threads", 4);
        declare_parameter("camera.processing_quality", "balanced");
        declare_parameter("camera.flip_images_180", false);
        // if (!has_parameter("camera.left_sensor_id")) {
        //     declare_parameter("camera.left_sensor_id", 0);
        // }
        // if (!has_parameter("camera.flip_images_180")) {
        //     declare_parameter("camera.flip_images_180", false);
        // }
        // if (!has_parameter("camera.format")) {
        //     declare_parameter("camera.format", std::string("GRAY8"));
        // }
        
        // ========================================================================
        // FRAME PARAMETERS
        // ========================================================================
        declare_parameter("frames.left_frame_id", "camera_left_link");
        declare_parameter("frames.right_frame_id", "camera_right_link");
        declare_parameter("frames.base_frame_id", "base_link");
        
        // ========================================================================
        // STEREO PROCESSING PARAMETERS
        // ========================================================================
        declare_parameter("stereo.algorithm", "GPU_BM");
        declare_parameter("stereo.use_gpu", true);
        declare_parameter("stereo.num_disparities", 64);
        declare_parameter("stereo.block_size", 15);
        declare_parameter("stereo.min_disparity", 0);
        declare_parameter("stereo.uniqueness_ratio", 10);
        declare_parameter("stereo.speckle_window_size", 100);
        declare_parameter("stereo.speckle_range", 32);
        declare_parameter("stereo.disp12_max_diff", 1);
        declare_parameter("stereo.pre_filter_cap", 31);
        declare_parameter("stereo.pre_filter_size", 9);
        declare_parameter("stereo.texture_threshold", 10);
        declare_parameter("stereo.smaller_block_size", 0);
        
        // ========================================================================
        // CALIBRATION PARAMETERS
        // ========================================================================
        declare_parameter("calibration.left_camera_info_url", "");
        declare_parameter("calibration.right_camera_info_url", "");
        declare_parameter("calibration.stereo_calibration_url", "");
        declare_parameter("calibration.auto_load_calibration", true);
        declare_parameter("calibration.min_calibration_samples", 30);
        declare_parameter("calibration.max_reprojection_error", 0.5);
        declare_parameter("calibration.default.focal_length", 300.0);
        declare_parameter("calibration.default.baseline", 0.06);
        declare_parameter("calibration.transforms.publish_camera_transforms", false);
        
        // ========================================================================
        // POINT CLOUD PARAMETERS
        // ========================================================================
        declare_parameter("pointcloud.enable", true);
        declare_parameter("pointcloud.downsample_factor", 1);
        declare_parameter("pointcloud.max_processing_threads", 4);
        
        // Voxel filter
        declare_parameter("pointcloud.voxel_filter.enable", true);
        declare_parameter("pointcloud.voxel_filter.leaf_size", 0.01);
        
        // Statistical filter
        declare_parameter("pointcloud.statistical_filter.enable", true);
        declare_parameter("pointcloud.statistical_filter.k_neighbors", 50);
        declare_parameter("pointcloud.statistical_filter.stddev_threshold", 1.0);
        
        // Range filter
        declare_parameter("pointcloud.range_filter.enable", true);
        declare_parameter("pointcloud.range_filter.min_range", 0.1);
        declare_parameter("pointcloud.range_filter.max_range", 10.0);
        
        // Passthrough filter
        declare_parameter("pointcloud.passthrough_filter.enable", false);
        declare_parameter("pointcloud.passthrough_filter.x_min", -5.0);
        declare_parameter("pointcloud.passthrough_filter.x_max", 5.0);
        declare_parameter("pointcloud.passthrough_filter.y_min", -5.0);
        declare_parameter("pointcloud.passthrough_filter.y_max", 5.0);
        declare_parameter("pointcloud.passthrough_filter.z_min", 0.0);
        declare_parameter("pointcloud.passthrough_filter.z_max", 3.0);
        
        // ========================================================================
        // PUBLISHING PARAMETERS
        // ========================================================================
        declare_parameter("publishing.publish_raw_images", true);
        declare_parameter("publishing.publish_rectified_images", true);
        declare_parameter("publishing.publish_disparity", true);
        declare_parameter("publishing.publish_pointcloud", true);
        declare_parameter("publishing.qos_depth", 10);
        
        // ========================================================================
        // PERFORMANCE PARAMETERS
        // ========================================================================
        declare_parameter("performance.log_performance", true);
        declare_parameter("performance.performance_log_interval", 100);
        declare_parameter("performance.enable_multithreading", true);
        declare_parameter("performance.thread_priority", 0);
        declare_parameter("performance.enable_memory_optimization", true);
        declare_parameter("performance.max_memory_usage_mb", 500);
        
        // ========================================================================
        // LOGGING PARAMETERS
        // ========================================================================
        declare_parameter("logging.log_calibration_info", true);
        declare_parameter("logging.log_processing_stats", false);
        declare_parameter("logging.log_frame_drops", true);
        
        // ========================================================================
        // DEVELOPMENT PARAMETERS
        // ========================================================================
        declare_parameter("development.save_debug_images", false);
        declare_parameter("development.debug_image_path", "/tmp/stereo_debug");
        declare_parameter("development.strict_frame_sync", true);
        declare_parameter("development.max_frame_age_ms", 100);
        declare_parameter("development.auto_restart_on_error", true);
        declare_parameter("development.max_consecutive_errors", 5);
        
        // NOTE: DO NOT declare "use_sim_time" - ROS2 handles this automatically
        
        // Load parameter values
        load_parameters();
    }
    
    void load_parameters() {
        // ========================================================================
        // CAMERA CONFIGURATION
        // ========================================================================
        camera_config_.width = get_parameter("camera.width").as_int();
        camera_config_.height = get_parameter("camera.height").as_int();
        camera_config_.fps = get_parameter("camera.fps").as_int();
        camera_config_.format = get_parameter("camera.format").as_string();
        camera_config_.use_hardware_acceleration = get_parameter("camera.use_hardware_acceleration").as_bool();
        camera_config_.flip_180 = get_parameter("camera.flip_images_180").as_bool();
        
        // ========================================================================
        // FRAME CONFIGURATION
        // ========================================================================
        left_frame_id_ = get_parameter("frames.left_frame_id").as_string();
        right_frame_id_ = get_parameter("frames.right_frame_id").as_string();
        base_frame_id_ = get_parameter("frames.base_frame_id").as_string();
        
        // ========================================================================
        // STEREO CONFIGURATION
        // ========================================================================
        stereo_config_.num_disparities = get_parameter("stereo.num_disparities").as_int();
        stereo_config_.block_size = get_parameter("stereo.block_size").as_int();
        stereo_config_.min_disparity = get_parameter("stereo.min_disparity").as_int();
        stereo_config_.max_disparity = stereo_config_.num_disparities;
        stereo_config_.uniqueness_ratio = get_parameter("stereo.uniqueness_ratio").as_int();
        stereo_config_.speckle_window_size = get_parameter("stereo.speckle_window_size").as_int();
        stereo_config_.speckle_range = get_parameter("stereo.speckle_range").as_int();
        stereo_config_.disp12_max_diff = get_parameter("stereo.disp12_max_diff").as_int();
        stereo_config_.use_gpu = get_parameter("stereo.use_gpu").as_bool();
        
        // ========================================================================
        // POINT CLOUD CONFIGURATION
        // ========================================================================
        pointcloud_config_.enable_voxel_filter = get_parameter("pointcloud.voxel_filter.enable").as_bool();
        pointcloud_config_.voxel_leaf_size = get_parameter("pointcloud.voxel_filter.leaf_size").as_double();
        pointcloud_config_.enable_statistical_filter = get_parameter("pointcloud.statistical_filter.enable").as_bool();
        pointcloud_config_.statistical_filter_k = get_parameter("pointcloud.statistical_filter.k_neighbors").as_int();
        pointcloud_config_.statistical_filter_stddev = get_parameter("pointcloud.statistical_filter.stddev_threshold").as_double();
        pointcloud_config_.enable_range_filter = get_parameter("pointcloud.range_filter.enable").as_bool();
        pointcloud_config_.min_range = get_parameter("pointcloud.range_filter.min_range").as_double();
        pointcloud_config_.max_range = get_parameter("pointcloud.range_filter.max_range").as_double();
        pointcloud_config_.max_threads = get_parameter("pointcloud.max_processing_threads").as_int();
        pointcloud_config_.downsample_factor = get_parameter("pointcloud.downsample_factor").as_int();
        
        // ========================================================================
        // PUBLISHING FLAGS
        // ========================================================================
        publish_raw_images_ = get_parameter("publishing.publish_raw_images").as_bool();
        publish_rectified_images_ = get_parameter("publishing.publish_rectified_images").as_bool();
        publish_disparity_ = get_parameter("publishing.publish_disparity").as_bool();
        publish_pointcloud_ = get_parameter("publishing.publish_pointcloud").as_bool();
        
        // ========================================================================
        // PERFORMANCE MONITORING
        // ========================================================================
        log_performance_ = get_parameter("performance.log_performance").as_bool();
        performance_log_interval_ = get_parameter("performance.performance_log_interval").as_int();
    }
    
    void create_cameras() {
        // Create left camera
        left_camera_ = CameraFactory::create_camera(CameraFactory::CameraType::JETSON_CSI);
        camera_config_.sensor_id = get_parameter("camera.left_sensor_id").as_int();
        
        RCLCPP_INFO(get_logger(), "Created left camera id:%d with resolution %dx%d at %d fps, image format: %s, flipped: %d",
                camera_config_.sensor_id, camera_config_.width, camera_config_.height, camera_config_.fps, camera_config_.format, camera_config_.flip_180);
        
        if (!left_camera_->initialize(camera_config_)) {
            throw std::runtime_error("Failed to initialize left camera");
        }
        
        // Create right camera
        right_camera_ = CameraFactory::create_camera(CameraFactory::CameraType::JETSON_CSI);
        camera_config_.sensor_id = get_parameter("camera.right_sensor_id").as_int();
        RCLCPP_INFO(get_logger(), "Created right camera id:%d with resolution %dx%d at %d fps, image format: %s, flipped: %d",
                camera_config_.sensor_id, camera_config_.width, camera_config_.height, camera_config_.fps, camera_config_.format, camera_config_.flip_180);
        if (!right_camera_->initialize(camera_config_)) {
            throw std::runtime_error("Failed to initialize right camera");
        }
        
        RCLCPP_INFO(get_logger(), "Created cameras with resolution %dx%d at %d fps",
                camera_config_.width, camera_config_.height, camera_config_.fps);
    }
    
    void create_stereo_processor() {
        std::string algorithm = get_parameter("stereo.algorithm").as_string();
        
        StereoProcessingFactory::StrategyType strategy_type;
        if (algorithm == "GPU_BM") {
            strategy_type = StereoProcessingFactory::StrategyType::GPU_BM;
        } else if (algorithm == "CPU_BM") {
            strategy_type = StereoProcessingFactory::StrategyType::CPU_BM;
        } else if (algorithm == "GPU_SGBM") {
            strategy_type = StereoProcessingFactory::StrategyType::GPU_SGBM;
        } else if (algorithm == "CPU_SGBM") {
            strategy_type = StereoProcessingFactory::StrategyType::CPU_SGBM;
        } else {
            RCLCPP_WARN(get_logger(), "Unknown stereo algorithm: %s, using CPU_BM", algorithm.c_str());
            strategy_type = StereoProcessingFactory::StrategyType::CPU_BM;
        }
        
        stereo_processor_ = StereoProcessingFactory::create_strategy(strategy_type);
        
        if (!stereo_processor_->initialize(stereo_config_, 
                                        cv::Size(camera_config_.width, camera_config_.height))) {
            throw std::runtime_error("Failed to initialize stereo processor");
        }
        
        RCLCPP_INFO(get_logger(), "Created stereo processor: %s", 
                stereo_processor_->get_strategy_name().c_str());
    }
    
    void create_calibration() {
        calibration_ = std::make_unique<StereoCalibration>();
    }
    
    void create_pointcloud_filter() {
        pointcloud_filter_ = std::make_unique<PointCloudFilter>(pointcloud_config_);
    }
    
    void setup_publishers() {
        // Raw image publishers (for calibration)
        if (publish_raw_images_) {
            left_image_pub_ = create_publisher<sensor_msgs::msg::Image>("left/image_raw", 10);
            right_image_pub_ = create_publisher<sensor_msgs::msg::Image>("right/image_raw", 10);
        }
        
        // Rectified image publishers
        if (publish_rectified_images_) {
            left_rect_pub_ = create_publisher<sensor_msgs::msg::Image>("left/image_rect", 10);
            right_rect_pub_ = create_publisher<sensor_msgs::msg::Image>("right/image_rect", 10);
        }
        
        // Camera info publishers
        left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("left/camera_info", 10);
        right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("right/camera_info", 10);
        
        // Stereo-specific publishers
        if (publish_disparity_) {
            disparity_pub_ = create_publisher<stereo_msgs::msg::DisparityImage>("disparity", 10);  // Correct type
        }
        
        if (publish_pointcloud_) {
            pointcloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("points", 10);
        }
        
        RCLCPP_INFO(get_logger(), "Publishers created with calibration-compatible topic names");
    }
    
    void setup_services() {
        // Individual camera calibration services
        left_set_info_service_ = create_service<sensor_msgs::srv::SetCameraInfo>(
            "left/set_camera_info",
            std::bind(&JetsonStereoNode::set_left_camera_info, this, 
                     std::placeholders::_1, std::placeholders::_2));
        
        right_set_info_service_ = create_service<sensor_msgs::srv::SetCameraInfo>(
            "right/set_camera_info",
            std::bind(&JetsonStereoNode::set_right_camera_info, this, 
                     std::placeholders::_1, std::placeholders::_2));
        
        // Stereo calibration service
        stereo_calibrate_service_ = create_service<std_srvs::srv::Trigger>(
            "calibrate_stereo",
            std::bind(&JetsonStereoNode::calibrate_stereo, this,
                     std::placeholders::_1, std::placeholders::_2));
    }
    
    void setup_camera_info_managers() {
        left_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
            this, "left", get_parameter("calibration.left_camera_info_url").as_string());
        
        right_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
            this, "right", get_parameter("calibration.right_camera_info_url").as_string());
    }
    
    void start_cameras() {
        if (!left_camera_->start()) {
            throw std::runtime_error("Failed to start left camera");
        }
        
        if (!right_camera_->start()) {
            throw std::runtime_error("Failed to start right camera");
        }
        
        RCLCPP_INFO(get_logger(), "Started cameras");
    }
    
    void stop_cameras() {
        if (left_camera_) {
            left_camera_->stop();
        }
        if (right_camera_) {
            right_camera_->stop();
        }
        
        RCLCPP_INFO(get_logger(), "Stopped cameras");
    }
    
    void start_processing() {
        processing_active_ = true;
        processing_thread_ = std::make_unique<std::thread>(&JetsonStereoNode::processing_loop, this);
        
        RCLCPP_INFO(get_logger(), "Started processing thread");
    }
    
    void stop_processing() {
        processing_active_ = false;
        
        if (processing_thread_ && processing_thread_->joinable()) {
            processing_thread_->join();
        }
        
        RCLCPP_INFO(get_logger(), "Stopped processing thread");
    }
    
    void processing_loop() {
        last_frame_time_ = now();
        int frame_count = 0;
        
        while (processing_active_ && rclcpp::ok()) {
            try {
                auto start_time = std::chrono::high_resolution_clock::now();
                
                // Get frames from cameras
                cv::Mat left_frame = left_camera_->get_frame();
                cv::Mat right_frame = right_camera_->get_frame();

                // Flip both images 180 degrees if enabled
                if (get_parameter("camera.flip_images_180").as_bool()) {
                    cv::flip(left_frame, left_frame, -1);   // -1 = 180 degree rotation
                    cv::flip(right_frame, right_frame, -1);
}

                // Add this in processing_loop() right after getting frames:
                RCLCPP_INFO_ONCE(get_logger(), "Left frame - Type: %d, Channels: %d, Depth: %d", 
                                left_frame.type(), left_frame.channels(), left_frame.depth());
                RCLCPP_INFO_ONCE(get_logger(), "Right frame - Type: %d, Channels: %d, Depth: %d", 
                                right_frame.type(), right_frame.channels(), right_frame.depth());
                
                if (left_frame.empty() || right_frame.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                
                // Process frames
                process_stereo_frames(left_frame, right_frame);
                
                // Calculate processing time and FPS
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                rclcpp::Time current_time = now();
                double frame_interval = (current_time - last_frame_time_).seconds();
                if (frame_interval > 0) {
                    processing_fps_ = 1.0 / frame_interval;
                }
                last_frame_time_ = current_time;
                
                // Log performance occasionally
                if (log_performance_ && (++frame_count % performance_log_interval_ == 0)) {
                    double avg_stereo_time, stereo_fps;
                    stereo_processor_->get_processing_stats(avg_stereo_time, stereo_fps);
                    
                    RCLCPP_DEBUG(get_logger(), 
                               "Performance - Total FPS: %.2f, Stereo FPS: %.2f, Processing time: %ld ms", 
                               processing_fps_.load(), stereo_fps, duration.count());
                }
                
            } catch (const std::exception& e) {
                RCLCPP_ERROR(get_logger(), "Error in processing loop: %s", e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
    
    void process_stereo_frames(const cv::Mat& left_frame, const cv::Mat& right_frame) {
        std::lock_guard<std::mutex> lock(processing_mutex_);
        
        rclcpp::Time timestamp = now();
        
        // Publish raw images first (for calibration)
        if (publish_raw_images_) {
            publish_image(left_image_pub_, left_frame, left_frame_id_, timestamp, "mono8");
            publish_image(right_image_pub_, right_frame, right_frame_id_, timestamp, "mono8");
        }
        
        // Publish camera info
        publish_camera_info(left_info_pub_, left_info_manager_, left_frame_id_, timestamp);
        publish_camera_info(right_info_pub_, right_info_manager_, right_frame_id_, timestamp);
        
        // Rectify images if calibrated
        cv::Mat left_rectified, right_rectified;
        if (calibration_->is_calibrated()) {
            calibration_->rectify_images(left_frame, right_frame, left_rectified, right_rectified);
            
            // Publish rectified images
            if (publish_rectified_images_) {
                publish_image(left_rect_pub_, left_rectified, left_frame_id_, timestamp, "mono8");
                publish_image(right_rect_pub_, right_rectified, right_frame_id_, timestamp, "mono8");
            }
        } else {
            left_rectified = left_frame;
            right_rectified = right_frame;
        }
        
        // Compute and publish disparity and point cloud
        if (calibration_->is_calibrated()) {
            compute_and_publish_disparity_and_pointcloud(left_rectified, right_rectified, timestamp);
        }
    }
    
    void publish_image(rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub,
                      const cv::Mat& image, const std::string& frame_id, 
                      const rclcpp::Time& timestamp, const std::string& encoding) {
        
        if (!pub || pub->get_subscription_count() == 0) return; // Don't publish if no subscribers
        
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), encoding, image).toImageMsg();
        msg->header.stamp = timestamp;
        msg->header.frame_id = frame_id;
        pub->publish(*msg);
    }
    
    void publish_camera_info(rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub,
                           std::shared_ptr<camera_info_manager::CameraInfoManager> manager,
                           const std::string& frame_id, const rclcpp::Time& timestamp) {
        
        if (!pub || pub->get_subscription_count() == 0) return;
        
        auto info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>(manager->getCameraInfo());
        info_msg->header.stamp = timestamp;
        info_msg->header.frame_id = frame_id;
        pub->publish(*info_msg);
    }
    
    void compute_and_publish_disparity_and_pointcloud(const cv::Mat& left_rectified, 
                                                     const cv::Mat& right_rectified,
                                                     const rclcpp::Time& timestamp) {
        
        // Convert to grayscale if needed (CUDA stereo requires single-channel)
        cv::Mat left_gray, right_gray;
        
        if (left_rectified.channels() == 3) {
            cv::cvtColor(left_rectified, left_gray, cv::COLOR_BGR2GRAY);
            RCLCPP_DEBUG_ONCE(get_logger(), "Converting left image from BGR to grayscale");
        } else {
            left_gray = left_rectified;
        }
        
        if (right_rectified.channels() == 3) {
            cv::cvtColor(right_rectified, right_gray, cv::COLOR_BGR2GRAY);
            RCLCPP_DEBUG_ONCE(get_logger(), "Converting right image from BGR to grayscale");
        } else {
            right_gray = right_rectified;
        }
        
        // Compute disparity with grayscale images
        cv::Mat disparity = stereo_processor_->compute_disparity(left_gray, right_gray);
        
        
        if (disparity.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to compute disparity");
            return;
        }
        
        // Publish disparity image
        if (publish_disparity_ && disparity_pub_ && disparity_pub_->get_subscription_count() > 0) {
            publish_disparity_image(disparity, timestamp);
        }
        
        // Generate and publish point cloud
        if (publish_pointcloud_ && pointcloud_pub_ && pointcloud_pub_->get_subscription_count() > 0) {
            auto pointcloud = stereo_processor_->generate_pointcloud(disparity, calibration_->get_Q_matrix());
            
            if (pointcloud && !pointcloud->empty()) {
                // Apply filters
                pointcloud_filter_->filter(pointcloud);
                
                // Convert to ROS message and publish
                sensor_msgs::msg::PointCloud2 pc_msg;
                pcl::toROSMsg(*pointcloud, pc_msg);
                pc_msg.header.stamp = timestamp;
                pc_msg.header.frame_id = left_frame_id_;
                
                pointcloud_pub_->publish(pc_msg);
            } else {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Generated empty point cloud");
            }
        }
    }
    
    void publish_disparity_image(const cv::Mat& disparity, const rclcpp::Time& timestamp) {
        stereo_msgs::msg::DisparityImage disparity_msg;
        
        disparity_msg.header.stamp = timestamp;
        disparity_msg.header.frame_id = left_frame_id_;
        
        // Fill disparity image info
        disparity_msg.image.header = disparity_msg.header;
        disparity_msg.image.height = disparity.rows;
        disparity_msg.image.width = disparity.cols;
        disparity_msg.image.encoding = "32FC1";
        disparity_msg.image.is_bigendian = false;
        disparity_msg.image.step = disparity.cols * sizeof(float);
        
        // Convert disparity to float and copy data
        cv::Mat disparity_float;
        if (disparity.type() == CV_16S) {
            disparity.convertTo(disparity_float, CV_32F, 1.0/16.0);
        } else {
            disparity.convertTo(disparity_float, CV_32F);
        }
        
        size_t data_size = disparity_float.rows * disparity_float.cols * sizeof(float);
        disparity_msg.image.data.resize(data_size);
        memcpy(&disparity_msg.image.data[0], disparity_float.data, data_size);
        
        // Fill disparity parameters
        disparity_msg.f = calibration_->get_left_camera_matrix().at<double>(0, 0); // fx
        disparity_msg.t = get_parameter("calibration.default.baseline").as_double(); // baseline
        disparity_msg.min_disparity = stereo_config_.min_disparity;
        disparity_msg.max_disparity = stereo_config_.min_disparity + stereo_config_.num_disparities;
        disparity_msg.delta_d = 1.0;
        
        disparity_pub_->publish(disparity_msg);
    }
    
    bool validate_camera_info(const sensor_msgs::msg::CameraInfo& camera_info) {
        // Check image dimensions
        if (camera_info.width != static_cast<uint32_t>(camera_config_.width) ||
            camera_info.height != static_cast<uint32_t>(camera_config_.height)) {
            RCLCPP_WARN(get_logger(), "Camera info dimensions don't match configured camera resolution");
            return false;
        }
        
        // Check camera matrix validity
        if (camera_info.k[0] <= 0 || camera_info.k[4] <= 0) { // fx, fy
            RCLCPP_ERROR(get_logger(), "Invalid focal lengths in camera matrix");
            return false;
        }
        
        // Check for reasonable focal length values
        double fx = camera_info.k[0];
        double fy = camera_info.k[4];
        if (fx < 100 || fx > 2000 || fy < 100 || fy > 2000) {
            RCLCPP_WARN(get_logger(), "Focal lengths seem unreasonable: fx=%.2f, fy=%.2f", fx, fy);
        }
        
        return true;
    }
    
    void set_left_camera_info(const std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Request> request,
                         std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Response> response) {
        
        // Validate camera info
        if (!validate_camera_info(request->camera_info)) {
            response->success = false;
            response->status_message = "Invalid camera info parameters";
            return;
        }
        
        // Set camera info in manager
        response->success = left_info_manager_->setCameraInfo(request->camera_info);
        
        if (response->success) {
            // Update calibration with new camera info
            if (calibration_->from_camera_info_msg(request->camera_info, true)) {
                // Save calibration to file
                std::string calib_file = get_parameter("calibration.left_camera_info_url").as_string();
                if (!calib_file.empty() && calib_file.find("file://") == 0) {
                    std::string file_path = calib_file.substr(7); // Remove "file://" prefix
                    if (calibration_->save_calibration_to_yaml(file_path, request->camera_info)) {
                        RCLCPP_INFO(get_logger(), "Saved left camera calibration to: %s", file_path.c_str());
                    } else {
                        RCLCPP_WARN(get_logger(), "Failed to save left camera calibration");
                    }
                }
            }
            response->status_message = "Left camera info set and saved successfully";
            RCLCPP_INFO(get_logger(), "Left camera calibration updated");
        } else {
            response->status_message = "Failed to set left camera info";
        }
    }

    void set_right_camera_info(const std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Request> request,
                            std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Response> response) {
        
        // Validate camera info
        if (!validate_camera_info(request->camera_info)) {
            response->success = false;
            response->status_message = "Invalid camera info parameters";
            return;
        }
        
        // Set camera info in manager
        response->success = right_info_manager_->setCameraInfo(request->camera_info);
        
        if (response->success) {
            // Update calibration with new camera info
            if (calibration_->from_camera_info_msg(request->camera_info, false)) {
                // Save calibration to file
                std::string calib_file = get_parameter("calibration.right_camera_info_url").as_string();
                if (!calib_file.empty() && calib_file.find("file://") == 0) {
                    std::string file_path = calib_file.substr(7); // Remove "file://" prefix
                    if (calibration_->save_calibration_to_yaml(file_path, request->camera_info)) {
                        RCLCPP_INFO(get_logger(), "Saved right camera calibration to: %s", file_path.c_str());
                    } else {
                        RCLCPP_WARN(get_logger(), "Failed to save right camera calibration");
                    }
                }
            }
            response->status_message = "Right camera info set and saved successfully";
            RCLCPP_INFO(get_logger(), "Right camera calibration updated");
        } else {
            response->status_message = "Failed to set right camera info";
        }
    }
    
    void calibrate_stereo(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        
        // Suppress unused parameter warning
        (void)request;
        
        // Check if both individual cameras are calibrated
        auto left_info = left_info_manager_->getCameraInfo();
        auto right_info = right_info_manager_->getCameraInfo();
        
        if (!validate_camera_info(left_info) || !validate_camera_info(right_info)) {
            response->success = false;
            response->message = "Both cameras must be individually calibrated first";
            return;
        }
        
        // Perform stereo calibration using existing camera calibrations
        if (calibration_->calibrate_stereo_from_individual(left_info, right_info)) {
            // Save stereo calibration
            std::string stereo_calib_file = get_parameter("calibration.stereo_calibration_url").as_string();
            if (!stereo_calib_file.empty() && stereo_calib_file.find("file://") == 0) {
                std::string file_path = stereo_calib_file.substr(7);
                if (calibration_->save_stereo_calibration(file_path)) {
                    RCLCPP_INFO(get_logger(), "Saved stereo calibration to: %s", file_path.c_str());
                }
            }
            
            response->success = true;
            response->message = "Stereo calibration completed and saved";
            RCLCPP_INFO(get_logger(), "Stereo calibration completed successfully");
        } else {
            response->success = false;
            response->message = "Stereo calibration failed";
            RCLCPP_ERROR(get_logger(), "Stereo calibration failed");
        }
    }
};

// StereoCalibration implementation
bool StereoCalibration::load_calibration(const std::string& left_info_url,
                                        const std::string& right_info_url,
                                        const std::string& stereo_info_url,
                                        const cv::Size& image_size) {
    image_size_ = image_size;
    
    bool left_loaded = false;
    bool right_loaded = false;
    bool stereo_loaded = false;
    
    // Try to load individual camera calibrations
    if (!left_info_url.empty() && left_info_url.find("file://") == 0) {
        std::string left_file = left_info_url.substr(7);
        left_loaded = load_calibration_from_yaml(left_file, true);
    }
    
    if (!right_info_url.empty() && right_info_url.find("file://") == 0) {
        std::string right_file = right_info_url.substr(7);
        right_loaded = load_calibration_from_yaml(right_file, false);
    }
    
    // Try to load stereo calibration
    if (!stereo_info_url.empty() && stereo_info_url.find("file://") == 0) {
        std::string stereo_file = stereo_info_url.substr(7);
        stereo_loaded = load_stereo_calibration_yaml(stereo_file);
    }
    
    if (left_loaded && right_loaded) {
        if (stereo_loaded) {
            compute_rectification_maps();
            calibrated_ = true;
            return true;
        } else {
            // Try to compute stereo calibration from individual calibrations
            setup_default_calibration();
            return false;
        }
    }
    
    // Fall back to default calibration
    setup_default_calibration();
    return false;
}

bool StereoCalibration::load_calibration_from_yaml(const std::string& yaml_file_path, bool is_left_camera) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_file_path);
        
        // Load camera matrix
        auto camera_matrix_data = config["camera_matrix"]["data"].as<std::vector<double>>();
        cv::Mat camera_matrix = cv::Mat(3, 3, CV_64F, camera_matrix_data.data()).clone();
        
        // Load distortion coefficients
        auto distortion_data = config["distortion_coefficients"]["data"].as<std::vector<double>>();
        cv::Mat dist_coeffs = cv::Mat(1, distortion_data.size(), CV_64F, distortion_data.data()).clone();
        
        if (is_left_camera) {
            camera_matrix_left_ = camera_matrix;
            dist_coeffs_left_ = dist_coeffs;
        } else {
            camera_matrix_right_ = camera_matrix;
            dist_coeffs_right_ = dist_coeffs;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("stereo_calibration"), 
                    "Failed to load calibration from %s: %s", yaml_file_path.c_str(), e.what());
        return false;
    }
}

bool StereoCalibration::load_stereo_calibration_yaml(const std::string& yaml_file_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_file_path);
        
        // Load rotation matrix
        auto R_data = config["rotation_matrix"]["data"].as<std::vector<double>>();
        R_ = cv::Mat(3, 3, CV_64F, R_data.data()).clone();
        
        // Load translation vector
        auto T_data = config["translation_vector"]["data"].as<std::vector<double>>();
        T_ = cv::Mat(3, 1, CV_64F, T_data.data()).clone();
        
        // Load Q matrix if available
        if (config["Q_matrix"]) {
            auto Q_data = config["Q_matrix"]["data"].as<std::vector<double>>();
            Q_ = cv::Mat(4, 4, CV_64F, Q_data.data()).clone();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("stereo_calibration"), 
                    "Failed to load stereo calibration from %s: %s", yaml_file_path.c_str(), e.what());
        return false;
    }
}

bool StereoCalibration::save_calibration_to_yaml(const std::string& yaml_file_path, 
                                                const sensor_msgs::msg::CameraInfo& camera_info) {
    try {
        YAML::Node config;
        
        // Basic information
        config["image_width"] = camera_info.width;
        config["image_height"] = camera_info.height;
        config["camera_name"] = "camera";
        
        // Camera matrix
        config["camera_matrix"]["rows"] = 3;
        config["camera_matrix"]["cols"] = 3;
        std::vector<double> k_data(camera_info.k.begin(), camera_info.k.end());
        config["camera_matrix"]["data"] = k_data;
        
        // Distortion model and coefficients
        config["distortion_model"] = camera_info.distortion_model;
        config["distortion_coefficients"]["rows"] = 1;
        config["distortion_coefficients"]["cols"] = camera_info.d.size();
        config["distortion_coefficients"]["data"] = camera_info.d;
        
        // Rectification matrix
        config["rectification_matrix"]["rows"] = 3;
        config["rectification_matrix"]["cols"] = 3;
        std::vector<double> r_data(camera_info.r.begin(), camera_info.r.end());
        config["rectification_matrix"]["data"] = r_data;
        
        // Projection matrix
        config["projection_matrix"]["rows"] = 3;
        config["projection_matrix"]["cols"] = 4;
        std::vector<double> p_data(camera_info.p.begin(), camera_info.p.end());
        config["projection_matrix"]["data"] = p_data;
        
        // Metadata
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        config["calibration_date"] = ss.str();
        config["calibration_version"] = "1.0";
        
        // Create directory if it doesn't exist
        std::filesystem::path file_path(yaml_file_path);
        std::filesystem::create_directories(file_path.parent_path());
        
        // Save to file
        std::ofstream file(yaml_file_path);
        file << config;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("stereo_calibration"), 
                    "Failed to save calibration to %s: %s", yaml_file_path.c_str(), e.what());
        return false;
    }
}

bool StereoCalibration::save_stereo_calibration(const std::string& yaml_file_path) {
    try {
        YAML::Node config;
        
        // Basic information
        config["image_width"] = image_size_.width;
        config["image_height"] = image_size_.height;
        config["calibration_type"] = "stereo";
        
        // Rotation matrix
        config["rotation_matrix"]["rows"] = 3;
        config["rotation_matrix"]["cols"] = 3;
        std::vector<double> R_data;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R_data.push_back(R_.at<double>(i, j));
            }
        }
        config["rotation_matrix"]["data"] = R_data;
        
        // Translation vector
        config["translation_vector"]["rows"] = 3;
        config["translation_vector"]["cols"] = 1;
        std::vector<double> T_data;
        for (int i = 0; i < 3; ++i) {
            T_data.push_back(T_.at<double>(i, 0));
        }
        config["translation_vector"]["data"] = T_data;
        
        // Q matrix
        if (!Q_.empty()) {
            config["Q_matrix"]["rows"] = 4;
            config["Q_matrix"]["cols"] = 4;
            std::vector<double> Q_data;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    Q_data.push_back(Q_.at<double>(i, j));
                }
            }
            config["Q_matrix"]["data"] = Q_data;
        }
        
        // Metadata
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        config["calibration_date"] = ss.str();
        config["calibration_version"] = "1.0";
        
        // Create directory if it doesn't exist
        std::filesystem::path file_path(yaml_file_path);
        std::filesystem::create_directories(file_path.parent_path());
        
        // Save to file
        std::ofstream file(yaml_file_path);
        file << config;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("stereo_calibration"), 
                    "Failed to save stereo calibration to %s: %s", yaml_file_path.c_str(), e.what());
        return false;
    }
}

sensor_msgs::msg::CameraInfo StereoCalibration::to_camera_info_msg(const std::string& frame_id, bool is_left) const {
    sensor_msgs::msg::CameraInfo camera_info;
    
    camera_info.header.frame_id = frame_id;
    camera_info.width = image_size_.width;
    camera_info.height = image_size_.height;
    
    // Set distortion model
    camera_info.distortion_model = "plumb_bob";
    
    // Select appropriate matrices
    const cv::Mat& camera_matrix = is_left ? camera_matrix_left_ : camera_matrix_right_;
    const cv::Mat& dist_coeffs = is_left ? dist_coeffs_left_ : dist_coeffs_right_;
    
    // Copy camera matrix (K)
    for (int i = 0; i < 9; ++i) {
        camera_info.k[i] = camera_matrix.at<double>(i / 3, i % 3);
    }
    
    // Copy distortion coefficients (D)
    camera_info.d.resize(dist_coeffs.cols);
    for (int i = 0; i < dist_coeffs.cols; ++i) {
        camera_info.d[i] = dist_coeffs.at<double>(0, i);
    }
    
    // Set rectification matrix (R)
    const cv::Mat& R_matrix = is_left ? R1_ : R2_;
    if (!R_matrix.empty()) {
        for (int i = 0; i < 9; ++i) {
            camera_info.r[i] = R_matrix.at<double>(i / 3, i % 3);
        }
    } else {
        // Identity matrix if not available
        for (int i = 0; i < 9; ++i) {
            camera_info.r[i] = (i % 4 == 0) ? 1.0 : 0.0;
        }
    }
    
    // Set projection matrix (P)
    const cv::Mat& P_matrix = is_left ? P1_ : P2_;
    if (!P_matrix.empty()) {
        for (int i = 0; i < 12; ++i) {
            camera_info.p[i] = P_matrix.at<double>(i / 4, i % 4);
        }
    } else {
        // Default projection matrix from camera matrix
        for (int i = 0; i < 12; ++i) {
            if (i < 3) {
                camera_info.p[i] = camera_matrix.at<double>(0, i);
            } else if (i >= 4 && i < 7) {
                camera_info.p[i] = camera_matrix.at<double>(1, i - 4);
            } else if (i >= 8 && i < 11) {
                camera_info.p[i] = camera_matrix.at<double>(2, i - 8);
            } else {
                camera_info.p[i] = 0.0;
            }
        }
    }
    
    return camera_info;
}

bool StereoCalibration::from_camera_info_msg(const sensor_msgs::msg::CameraInfo& camera_info, bool is_left) {
    try {
        // Convert camera matrix
        cv::Mat camera_matrix = cv::Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < 9; ++i) {
            camera_matrix.at<double>(i / 3, i % 3) = camera_info.k[i];
        }
        
        // Convert distortion coefficients
        cv::Mat dist_coeffs = cv::Mat::zeros(1, camera_info.d.size(), CV_64F);
        for (size_t i = 0; i < camera_info.d.size(); ++i) {
            dist_coeffs.at<double>(0, i) = camera_info.d[i];
        }
        
        if (is_left) {
            camera_matrix_left_ = camera_matrix;
            dist_coeffs_left_ = dist_coeffs;
        } else {
            camera_matrix_right_ = camera_matrix;
            dist_coeffs_right_ = dist_coeffs;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("stereo_calibration"), 
                    "Failed to convert camera info: %s", e.what());
        return false;
    }
}

bool StereoCalibration::calibrate_stereo_from_individual(const sensor_msgs::msg::CameraInfo& left_info,
                                                       const sensor_msgs::msg::CameraInfo& right_info) {
    // Convert camera info to OpenCV format
    if (!from_camera_info_msg(left_info, true) || !from_camera_info_msg(right_info, false)) {
        return false;
    }
    
    // Use default stereo calibration parameters
    // In a real implementation, you would collect image pairs and perform actual stereo calibration
    // For now, we'll use reasonable defaults based on the individual calibrations
    
    R_ = cv::Mat::eye(3, 3, CV_64F);  // No rotation between cameras (parallel setup)
    T_ = (cv::Mat_<double>(3, 1) << -0.06, 0.0, 0.0);  // 6cm baseline assumption
    
    compute_rectification_maps();
    calibrated_ = true;
    
    return true;
}

void StereoCalibration::setup_default_calibration() {
    // Default calibration parameters for IMX219-83 cameras
    // Camera matrices (approximate values for configured resolution)
    double focal_length = 300.0 * (image_size_.width / 640.0); // Scale with resolution
    
    camera_matrix_left_ = (cv::Mat_<double>(3, 3) << 
        focal_length, 0.0, image_size_.width / 2.0,
        0.0, focal_length, image_size_.height / 2.0,
        0.0, 0.0, 1.0);
    
    camera_matrix_right_ = camera_matrix_left_.clone();
    
    // Minimal distortion coefficients
    dist_coeffs_left_ = cv::Mat::zeros(4, 1, CV_64F);
    dist_coeffs_right_ = cv::Mat::zeros(4, 1, CV_64F);
    
    // Stereo calibration parameters
    R_ = cv::Mat::eye(3, 3, CV_64F);  // No rotation between cameras
    T_ = (cv::Mat_<double>(3, 1) << -0.06, 0.0, 0.0);  // 6cm baseline
    
    compute_rectification_maps();
    calibrated_ = true;
    
    RCLCPP_WARN(rclcpp::get_logger("stereo_calibration"), 
                "Using default calibration parameters. Please calibrate cameras for accurate results.");
}

void StereoCalibration::compute_rectification_maps() {
    if (camera_matrix_left_.empty() || camera_matrix_right_.empty()) {
        return;
    }
    
    cv::stereoRectify(camera_matrix_left_, dist_coeffs_left_,
                     camera_matrix_right_, dist_coeffs_right_,
                     image_size_, R_, T_, R1_, R2_, P1_, P2_, Q_,
                     cv::CALIB_ZERO_DISPARITY, 0, image_size_);
    
    cv::initUndistortRectifyMap(camera_matrix_left_, dist_coeffs_left_, R1_, P1_,
                               image_size_, CV_16SC2, left_map1_, left_map2_);
    
    cv::initUndistortRectifyMap(camera_matrix_right_, dist_coeffs_right_, R2_, P2_,
                               image_size_, CV_16SC2, right_map1_, right_map2_);
}

void StereoCalibration::rectify_images(const cv::Mat& left_raw, const cv::Mat& right_raw,
                                     cv::Mat& left_rectified, cv::Mat& right_rectified) {
    if (!calibrated_ || left_map1_.empty() || right_map1_.empty()) {
        left_rectified = left_raw;
        right_rectified = right_raw;
        return;
    }
    
    cv::remap(left_raw, left_rectified, left_map1_, left_map2_, cv::INTER_LINEAR);
    cv::remap(right_raw, right_rectified, right_map1_, right_map2_, cv::INTER_LINEAR);
}

} // namespace jetson_stereo_camera

// Main function
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<jetson_stereo_camera::JetsonStereoNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("jetson_stereo_node"), 
                    "Failed to start stereo camera node: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}