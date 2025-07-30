#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include "jetank_perception/camera_interface.hpp"
#include <chrono>

class CameraNode : public rclcpp::Node
{
public:
    CameraNode() : Node("camera_node")
    {
        // Declare ROS2 parameters with default values
        this->declare_parameter("camera_width", 640);
        this->declare_parameter("camera_height", 480);
        this->declare_parameter("camera_fps", 30);
        this->declare_parameter("camera_format", "NV12");
        this->declare_parameter("sensor_id", 0);
        this->declare_parameter("use_hardware_acceleration", true);
        this->declare_parameter("publish_rate_hz", 30.0);
        
        // Get parameter values
        int width = this->get_parameter("camera_width").as_int();
        int height = this->get_parameter("camera_height").as_int();
        int fps = this->get_parameter("camera_fps").as_int();
        std::string format = this->get_parameter("camera_format").as_string();
        int sensor_id = this->get_parameter("sensor_id").as_int();
        bool hw_accel = this->get_parameter("use_hardware_acceleration").as_bool();
        double publish_rate = this->get_parameter("publish_rate_hz").as_double();
        
        RCLCPP_INFO(this->get_logger(), "Initializing camera with: %dx%d @ %d fps", width, height, fps);
        
        // Create publisher
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", 10);
        
        // Configure camera
        jetson_stereo_camera::CameraConfig config;
        config.width = width;
        config.height = height;
        config.fps = fps;
        config.format = format;
        config.sensor_id = sensor_id;
        config.use_hardware_acceleration = hw_accel;
        
        // Create camera instance
        camera_ = jetson_stereo_camera::CameraFactory::create_camera(
            jetson_stereo_camera::CameraFactory::CameraType::JETSON_CSI);
        
        if (!camera_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create camera instance");
            return;
        }
        
        // Initialize camera
        if (!camera_->initialize(config)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize camera");
            return;
        }
        
        // Enable threading for async operation
        camera_->enable_threading(true);
        
        // Start camera
        if (!camera_->start()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to start camera");
            return;
        }
        
        // Set up rate limiting
        frame_interval_ = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / publish_rate));
        last_published_time_ = std::chrono::steady_clock::now();
        
        // Set up async callback
        camera_->get_frame_async([this](const cv::Mat& frame) {
            this->frame_callback(frame);
        });
        
        RCLCPP_INFO(this->get_logger(), "Camera node initialized successfully");
        RCLCPP_INFO(this->get_logger(), "Publishing at %.1f Hz on topic: camera/image_raw", publish_rate);
    }
    
    ~CameraNode()
    {
        if (camera_ && camera_->is_running()) {
            camera_->stop();
            RCLCPP_INFO(this->get_logger(), "Camera stopped");
        }
    }

private:
    void frame_callback(const cv::Mat& frame)
    {
        // Check if frame is valid
        if (frame.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                                 "Received empty frame from camera");
            return;
        }
        
        // Rate limiting - only publish if enough time has passed
        auto current_time = std::chrono::steady_clock::now();
        if (current_time - last_published_time_ < frame_interval_) {
            return;  // Skip this frame
        }
        
        try {
            // Convert OpenCV Mat to ROS2 Image message
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
            
            // Set timestamp
            msg->header.stamp = this->get_clock()->now();
            msg->header.frame_id = "camera_frame";
            
            // Publish the image
            image_publisher_->publish(*msg);
            
            // Update last published time
            last_published_time_ = current_time;
            
            // Log occasionally to show it's working
            frame_count_++;
            if (frame_count_ % 100 == 0) {
                RCLCPP_INFO(this->get_logger(), "Published %d frames", frame_count_);
            }
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    // ROS2 components
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    
    // Camera interface
    std::unique_ptr<jetson_stereo_camera::CameraInterface> camera_;
    
    // Rate limiting
    std::chrono::nanoseconds frame_interval_;
    std::chrono::steady_clock::time_point last_published_time_;
    
    // Statistics
    int frame_count_ = 0;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<CameraNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("camera_node"), "Exception in main: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}