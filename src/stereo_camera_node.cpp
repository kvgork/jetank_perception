#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/srv/set_camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <camera_info_manager/camera_info_manager.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "jetank_perception/camera_interface.hpp"
#include "jetank_perception/jetson_csi_camera.hpp"
#include "jetank_perception/stereo_processing_strategy.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

namespace jetson_stereo_camera {

class StereoCalibration {
public:
    bool load_calibration(const std::string& left_info_url, 
                         const std::string& right_info_url,
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
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    
    // Camera info managers
    std::shared_ptr<camera_info_manager::CameraInfoManager> left_info_manager_;
    std::shared_ptr<camera_info_manager::CameraInfoManager> right_info_manager_;
    
    // Services
    rclcpp::Service<sensor_msgs::srv::SetCameraInfo>::SharedPtr left_set_info_service_;
    rclcpp::Service<sensor_msgs::srv::SetCameraInfo>::SharedPtr right_set_info_service_;
    
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
    
    // Performance monitoring
    std::atomic<double> processing_fps_;
    rclcpp::Time last_frame_time_;
    
public:
    JetsonStereoNode() : Node("jetson_stereo_node"), processing_active_(false), processing_fps_(0.0) {
        // Initialize components
        initialize_parameters();
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
            get_parameter("left_camera_info_url").as_string(),
            get_parameter("right_camera_info_url").as_string(),
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
        // Declare parameters
        declare_parameter("camera_width", 640);
        declare_parameter("camera_height", 480);
        declare_parameter("camera_fps", 20);
        declare_parameter("left_frame_id", "camera_left_link");
        declare_parameter("right_frame_id", "camera_right_link");
        declare_parameter("left_camera_info_url", "");
        declare_parameter("right_camera_info_url", "");
        declare_parameter("use_gpu", true);
        declare_parameter("stereo_algorithm", "GPU_BM");
        declare_parameter("num_disparities", 64);
        declare_parameter("block_size", 15);
        declare_parameter("min_disparity", 0);
        declare_parameter("max_disparity", 64);
        declare_parameter("voxel_leaf_size", 0.01);
        declare_parameter("statistical_filter_k", 50);
        declare_parameter("statistical_filter_stddev", 1.0);
        declare_parameter("max_processing_threads", 4);
        
        // Get parameters
        camera_config_.width = get_parameter("camera_width").as_int();
        camera_config_.height = get_parameter("camera_height").as_int();
        camera_config_.fps = get_parameter("camera_fps").as_int();
        camera_config_.format = "GRAY8";
        camera_config_.use_hardware_acceleration = get_parameter("use_gpu").as_bool();
        
        left_frame_id_ = get_parameter("left_frame_id").as_string();
        right_frame_id_ = get_parameter("right_frame_id").as_string();
        
        // Stereo configuration
        stereo_config_.num_disparities = get_parameter("num_disparities").as_int();
        stereo_config_.block_size = get_parameter("block_size").as_int();
        stereo_config_.min_disparity = get_parameter("min_disparity").as_int();
        stereo_config_.max_disparity = get_parameter("max_disparity").as_int();
        stereo_config_.use_gpu = get_parameter("use_gpu").as_bool();
        
        // Point cloud configuration
        pointcloud_config_.voxel_leaf_size = get_parameter("voxel_leaf_size").as_double();
        pointcloud_config_.statistical_filter_k = get_parameter("statistical_filter_k").as_int();
        pointcloud_config_.statistical_filter_stddev = get_parameter("statistical_filter_stddev").as_double();
    }
    
    void create_cameras() {
        // Create left camera (sensor_id = 0)
        left_camera_ = CameraFactory::create_camera(CameraFactory::CameraType::JETSON_CSI);
        camera_config_.sensor_id = 0;
        if (!left_camera_->initialize(camera_config_)) {
            throw std::runtime_error("Failed to initialize left camera");
        }
        
        // Create right camera (sensor_id = 1)
        right_camera_ = CameraFactory::create_camera(CameraFactory::CameraType::JETSON_CSI);
        camera_config_.sensor_id = 1;
        if (!right_camera_->initialize(camera_config_)) {
            throw std::runtime_error("Failed to initialize right camera");
        }
        
        RCLCPP_INFO(get_logger(), "Created cameras with resolution %dx%d at %d fps",
                   camera_config_.width, camera_config_.height, camera_config_.fps);
    }
    
    void create_stereo_processor() {
        std::string algorithm = get_parameter("stereo_algorithm").as_string();
        
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
            RCLCPP_WARN(get_logger(), "Unknown stereo algorithm: %s, using GPU_BM", algorithm.c_str());
            strategy_type = StereoProcessingFactory::StrategyType::GPU_BM;
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
        left_image_pub_ = create_publisher<sensor_msgs::msg::Image>("left/image_raw", 10);
        right_image_pub_ = create_publisher<sensor_msgs::msg::Image>("right/image_raw", 10);
        left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("left/camera_info", 10);
        right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("right/camera_info", 10);
        pointcloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("points", 10);
    }
    
    void setup_services() {
        left_set_info_service_ = create_service<sensor_msgs::srv::SetCameraInfo>(
            "left/set_camera_info",
            std::bind(&JetsonStereoNode::set_left_camera_info, this, 
                     std::placeholders::_1, std::placeholders::_2));
        
        right_set_info_service_ = create_service<sensor_msgs::srv::SetCameraInfo>(
            "right/set_camera_info",
            std::bind(&JetsonStereoNode::set_right_camera_info, this, 
                     std::placeholders::_1, std::placeholders::_2));
    }
    
    void setup_camera_info_managers() {
        left_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
            this, "left", get_parameter("left_camera_info_url").as_string());
        
        right_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
            this, "right", get_parameter("right_camera_info_url").as_string());
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
        
        while (processing_active_ && rclcpp::ok()) {
            try {
                auto start_time = std::chrono::high_resolution_clock::now();
                
                // Get frames from cameras
                cv::Mat left_frame = left_camera_->get_frame();
                cv::Mat right_frame = right_camera_->get_frame();
                
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
                static int frame_count = 0;
                if (++frame_count % 100 == 0) {
                    RCLCPP_DEBUG(get_logger(), "Processing FPS: %.2f, Time: %ld ms", 
                               processing_fps_.load(), duration.count());
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
        
        // Rectify images if calibrated
        cv::Mat left_rectified, right_rectified;
        if (calibration_->is_calibrated()) {
            calibration_->rectify_images(left_frame, right_frame, left_rectified, right_rectified);
        } else {
            left_rectified = left_frame;
            right_rectified = right_frame;
        }
        
        // Publish rectified images
        publish_image(left_image_pub_, left_rectified, left_frame_id_, timestamp);
        publish_image(right_image_pub_, right_rectified, right_frame_id_, timestamp);
        
        // Publish camera info
        publish_camera_info(left_info_pub_, left_info_manager_, left_frame_id_, timestamp);
        publish_camera_info(right_info_pub_, right_info_manager_, right_frame_id_, timestamp);
        
        // Compute and publish point cloud
        if (calibration_->is_calibrated() && pointcloud_pub_->get_subscription_count() > 0) {
            compute_and_publish_pointcloud(left_rectified, right_rectified, timestamp);
        }
    }
    
    void publish_image(rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub,
                      const cv::Mat& image, const std::string& frame_id, 
                      const rclcpp::Time& timestamp) {
        
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", image).toImageMsg();
        msg->header.stamp = timestamp;
        msg->header.frame_id = frame_id;
        pub->publish(*msg);
    }
    
    void publish_camera_info(rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub,
                           std::shared_ptr<camera_info_manager::CameraInfoManager> manager,
                           const std::string& frame_id, const rclcpp::Time& timestamp) {
        
        auto info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>(manager->getCameraInfo());
        info_msg->header.stamp = timestamp;
        info_msg->header.frame_id = frame_id;
        pub->publish(*info_msg);
    }
    
    void compute_and_publish_pointcloud(const cv::Mat& left_rectified, 
                                      const cv::Mat& right_rectified,
                                      const rclcpp::Time& timestamp) {
        
        // Compute disparity
        cv::Mat disparity = stereo_processor_->compute_disparity(left_rectified, right_rectified);
        
        if (disparity.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to compute disparity");
            return;
        }
        
        // Generate point cloud
        auto pointcloud = stereo_processor_->generate_pointcloud(disparity, calibration_->get_Q_matrix());
        
        if (!pointcloud || pointcloud->empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Generated empty point cloud");
            return;
        }
        
        // Apply filters
        pointcloud_filter_->filter(pointcloud);
        
        // Convert to ROS message and publish
        sensor_msgs::msg::PointCloud2 pc_msg;
        pcl::toROSMsg(*pointcloud, pc_msg);
        pc_msg.header.stamp = timestamp;
        pc_msg.header.frame_id = left_frame_id_;
        
        pointcloud_pub_->publish(pc_msg);
    }
    
    void set_left_camera_info(const std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Request> request,
                             std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Response> response) {
        response->success = left_info_manager_->setCameraInfo(request->camera_info);
        response->status_message = response->success ? "Camera info set successfully" : "Failed to set camera info";
    }
    
    void set_right_camera_info(const std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Request> request,
                              std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Response> response) {
        response->success = right_info_manager_->setCameraInfo(request->camera_info);
        response->status_message = response->success ? "Camera info set successfully" : "Failed to set camera info";
    }
};

// StereoCalibration implementation
bool StereoCalibration::load_calibration(const std::string& left_info_url,
                                        const std::string& right_info_url,
                                        const cv::Size& image_size) {
    image_size_ = image_size;
    
    // Try to load calibration from files
    // This is a simplified implementation - in practice, you'd load from YAML files
    if (left_info_url.empty() || right_info_url.empty()) {
        setup_default_calibration();
        return false;
    }
    
    // TODO: Implement proper calibration loading from files
    // For now, use default calibration
    setup_default_calibration();
    return true;
}

void StereoCalibration::setup_default_calibration() {
    // Default calibration parameters for IMX219-83 cameras
    // These should be replaced with actual calibration data
    
    // Camera matrices (approximate values for 640x480)
    camera_matrix_left_ = (cv::Mat_<double>(3, 3) << 
        300.0, 0.0, 320.0,
        0.0, 300.0, 240.0,
        0.0, 0.0, 1.0);
    
    camera_matrix_right_ = (cv::Mat_<double>(3, 3) << 
        300.0, 0.0, 320.0,
        0.0, 300.0, 240.0,
        0.0, 0.0, 1.0);
    
    // Distortion coefficients (assuming minimal distortion)
    dist_coeffs_left_ = cv::Mat::zeros(4, 1, CV_64F);
    dist_coeffs_right_ = cv::Mat::zeros(4, 1, CV_64F);
    
    // Stereo calibration parameters (approximate)
    R_ = cv::Mat::eye(3, 3, CV_64F);  // No rotation between cameras
    T_ = (cv::Mat_<double>(3, 1) << -0.06, 0.0, 0.0);  // 6cm baseline
    
    compute_rectification_maps();
    calibrated_ = true;
}

void StereoCalibration::compute_rectification_maps() {
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
    if (!calibrated_) {
        left_rectified = left_raw;
        right_rectified = right_raw;
        return;
    }
    
    cv::remap(left_raw, left_rectified, left_map1_, left_map2_, cv::INTER_LINEAR);
    cv::remap(right_raw, right_rectified, right_map1_, right_map2_, cv::INTER_LINEAR);
}

}

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