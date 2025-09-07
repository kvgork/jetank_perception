#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <functional>
#include <vector>
#include <map>
#include <atomic>        // ← Added this missing include
#include <thread>
#include <mutex>
#include <chrono>
#include <filesystem>    // ← Added this missing include
#include <iostream>
#include <algorithm>

namespace jetson_stereo_camera {

// Configuration structures
struct CameraConfig {
    int width = 640;
    int height = 480;
    int fps = 30;
    int sensor_id = 0;
    std::string format = "BGR";
    bool use_hardware_acceleration = true;
    bool flip_180 = false;
};

// Pipeline template structure for robust pipeline building
struct PipelineTemplate {
    std::string name;
    std::string template_str;
    std::string description;
    int priority;
};

// Abstract camera interface
class CameraInterface {
public:
    virtual ~CameraInterface() = default;
    
    // Core interface methods
    virtual bool initialize(const CameraConfig& config) = 0;
    virtual bool start() = 0;
    virtual bool stop() = 0;
    virtual bool is_running() const = 0;
    virtual cv::Mat get_frame() = 0;
    
    // Extended interface methods
    virtual bool get_frame_async(std::function<void(const cv::Mat&)> callback) = 0;
    virtual std::string get_camera_type() const = 0;
    virtual bool supports_hardware_acceleration() const = 0;
    virtual CameraConfig get_config() const = 0;
    virtual void set_buffer_size(int size) = 0;
    virtual void enable_threading(bool enable) = 0;
    
    // Parameter interface (for camera controls)
    virtual bool set_parameter(const std::string& param_name, double value) = 0;
    virtual double get_parameter(const std::string& param_name) const = 0;

protected:
    CameraConfig config_;
    std::atomic<bool> running_{false};
    
    // Static pipeline cache for GStreamer pipelines
    static std::map<std::string, std::string> pipeline_cache_;
};

// Initialize the static pipeline cache
std::map<std::string, std::string> CameraInterface::pipeline_cache_;

// ============================================================================
// JETSON CSI CAMERA IMPLEMENTATION (Header-Only)
// ============================================================================
class JetsonCSICamera : public CameraInterface {
private:
    cv::VideoCapture cap_;
    std::thread capture_thread_;
    std::function<void(const cv::Mat&)> async_callback_;
    std::atomic<bool> async_mode_{false};
    int buffer_size_ = 1;
    bool threading_enabled_ = false;
    mutable std::mutex frame_mutex_;
    cv::Mat latest_frame_;

public:
    ~JetsonCSICamera() {
        stop();
    }

    bool initialize(const CameraConfig& config) override {
        config_ = config;
        
        // Use robust pipeline building with testing and caching
        std::string pipeline = build_gstreamer_pipeline(config);
        
        cap_.open(pipeline, cv::CAP_GSTREAMER);

        // Test script;
        if (cap_.isOpened()) {
            double actual_width = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
            double actual_height = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
            double format = cap_.get(cv::CAP_PROP_FORMAT);
            
            std::cout << "=== DETAILED CAMERA DEBUG ===" << std::endl;
            std::cout << "Requested: " << config.width << "x" << config.height << std::endl;
            std::cout << "Actual: " << actual_width << "x" << actual_height << std::endl;
            std::cout << "Format code: " << format << std::endl;
            std::cout << "Pipeline used: " << pipeline << std::endl;
            
            // Test frame to see actual dimensions
            cv::Mat test_frame;
            cap_ >> test_frame;
            if (!test_frame.empty()) {
                std::cout << "Actual frame size: " << test_frame.cols << "x" << test_frame.rows << std::endl;
                std::cout << "Frame channels: " << test_frame.channels() << std::endl;
                std::cout << "Frame type: " << test_frame.type() << std::endl;
            }
            std::cout << "=============================" << std::endl;
        }
        
        if (!cap_.isOpened()) {
            std::cerr << "Failed to open CSI camera with pipeline: " << pipeline << std::endl;
            return false;
        }

        // Set buffer size for performance
        cap_.set(cv::CAP_PROP_BUFFERSIZE, buffer_size_);
        
        std::cout << "Jetson CSI Camera initialized successfully" << std::endl;
        return true;
    }

    bool start() override {
        if (running_) {
            return true;
        }
        
        if (!cap_.isOpened()) {
            std::cerr << "Camera not initialized" << std::endl;
            return false;
        }

        running_ = true;
        
        if (threading_enabled_ && async_mode_) {
            capture_thread_ = std::thread(&JetsonCSICamera::capture_loop, this);
        }
        
        return true;
    }

    bool stop() override {
        if (!running_) {
            return true;
        }
        
        running_ = false;
        async_mode_ = false;
        
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }
        
        cap_.release();
        return true;
    }

    bool is_running() const override {
        return running_;
    }

    cv::Mat get_frame() override {
        if (!running_ || !cap_.isOpened()) {
            return cv::Mat();
        }

        cv::Mat frame;
        if (threading_enabled_) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            frame = latest_frame_.clone();
        } else {
            cap_ >> frame;
        }

        // Debug: Log actual frame dimensions
        if (!frame.empty()) {
            static bool logged = false;
            if (!logged) {
                std::cout << "Actual frame received: " << frame.cols << "x" << frame.rows 
                        << ", channels: " << frame.channels() << std::endl;
                logged = true;
            }
        }

        return frame;
    }

    bool get_frame_async(std::function<void(const cv::Mat&)> callback) override {
        if (!running_) {
            return false;
        }
        
        async_callback_ = callback;
        async_mode_ = true;
        
        if (threading_enabled_ && !capture_thread_.joinable()) {
            capture_thread_ = std::thread(&JetsonCSICamera::capture_loop, this);
        }
        
        return true;
    }

    std::string get_camera_type() const override {
        return "Jetson CSI Camera";
    }

    bool supports_hardware_acceleration() const override {
        return true;
    }

    CameraConfig get_config() const override {
        return config_;
    }

    void set_buffer_size(int size) override {
        buffer_size_ = std::max(1, size);
        if (cap_.isOpened()) {
            cap_.set(cv::CAP_PROP_BUFFERSIZE, buffer_size_);
        }
    }

    void enable_threading(bool enable) override {
        threading_enabled_ = enable;
    }

    bool set_parameter(const std::string& param_name, double value) override {
        // For CSI cameras, we can set some basic parameters
        if (param_name == "brightness" && cap_.isOpened()) {
            return cap_.set(cv::CAP_PROP_BRIGHTNESS, value);
        } else if (param_name == "contrast" && cap_.isOpened()) {
            return cap_.set(cv::CAP_PROP_CONTRAST, value);
        } else if (param_name == "saturation" && cap_.isOpened()) {
            return cap_.set(cv::CAP_PROP_SATURATION, value);
        }
        return false;
    }

    double get_parameter(const std::string& param_name) const override {
        if (param_name == "brightness" && cap_.isOpened()) {
            return cap_.get(cv::CAP_PROP_BRIGHTNESS);
        } else if (param_name == "contrast" && cap_.isOpened()) {
            return cap_.get(cv::CAP_PROP_CONTRAST);
        } else if (param_name == "saturation" && cap_.isOpened()) {
            return cap_.get(cv::CAP_PROP_SATURATION);
        }
        return -1.0;
    }

private:
    // Robust pipeline builder with testing and caching
    std::string build_gstreamer_pipeline(const CameraConfig& config) {
        // Create a cache key based on configuration
        std::string cache_key = create_cache_key(config);
        
        // Check if we have a cached working pipeline for this configuration
        auto cached_it = pipeline_cache_.find(cache_key);
        if (cached_it != pipeline_cache_.end()) {
            std::cout << "[PIPELINE] Using cached pipeline for " << cache_key << std::endl;
            return cached_it->second;
        }
        
        // Define pipeline templates in order of preference
        std::vector<PipelineTemplate> templates = get_pipeline_templates();
        
        std::cout << "[PIPELINE] Testing " << templates.size() << " pipeline configurations" << std::endl;
        
        // Test each pipeline template
        for (const auto& template_config : templates) {
            std::string pipeline = build_pipeline_from_template(template_config, config);
            
            std::cout << "[PIPELINE] Testing: " << template_config.name << std::endl;
            
            if (test_pipeline_compatibility(pipeline)) {
                std::cout << "[PIPELINE] ✓ SUCCESS: " << template_config.name << " works!" << std::endl;
                
                // Cache this successful configuration
                pipeline_cache_[cache_key] = pipeline;
                return pipeline;
            } else {
                std::cout << "[PIPELINE] ✗ FAILED: " << template_config.name << std::endl;
            }
        }
        
        // If all templates fail, return a minimal fallback
        std::string fallback = get_fallback_pipeline(config);
        std::cout << "[PIPELINE] ⚠ WARNING: Using fallback pipeline" << std::endl;
        return fallback;
    }
    
    // std::vector<PipelineTemplate> get_pipeline_templates() {
    //     std::vector<PipelineTemplate> templates;
        
    //     // High-performance, hardware-accelerated
    //     templates.push_back({
    //         "HW_Accelerated_BGRx",
    //         "nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM), width=%d, height=%d, format=%s, framerate=%d/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
    //         "Hardware accelerated with NVMM memory",
    //         1
    //     });
        
    //     // Simple hardware-accelerated
    //     templates.push_back({
    //         "HW_Accelerated_Auto",
    //         "nvarguscamerasrc sensor-id=%d ! nvvidconv ! video/x-raw, width=%d, height=%d ! videoconvert ! video/x-raw, format=BGR ! appsink",
    //         "Hardware accelerated with auto format",
    //         2
    //     });
        
    //     // Software fallback
    //     templates.push_back({
    //         "Software_Fallback",
    //         "nvarguscamerasrc sensor-id=%d ! videoconvert ! video/x-raw, width=%d, height=%d, format=BGR ! appsink",
    //         "Software-only conversion",
    //         3
    //     });
        
    //     return templates;
    // }
    std::vector<PipelineTemplate> get_pipeline_templates() {
        std::vector<PipelineTemplate> templates;
        
        // Force full sensor readout, no dimension specification to nvarguscamerasrc
        // templates.push_back({
        //     "Full_Sensor_No_Crop",
        //     "nvarguscamerasrc sensor-id=%d ! nvvidconv ! videoconvert ! video/x-raw, format=BGRx ! appsink",
        //     "Full sensor without cropping",
        //     1
        // });
        
        // // If that doesn't work, try explicitly requesting 1920x1080
        // templates.push_back({
        //     "Explicit_1080p",
        //     "nvarguscamerasrc sensor-id=%d ! video/x-raw, width=1920, height=1080 ! nvvidconv ! videoconvert ! video/x-raw, format=BGR ! appsink",
        //     "Explicit 1080p request",
        //     2
        // });
        
        // // Your working but cropping pipeline
        // templates.push_back({
        //     "Working_Cropped",
        //     "nvarguscamerasrc sensor-id=%d ! nvvidconv ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw, format=BGR ! appsink",
        //     "Working cropped pipeline",
        //     3
        // });

        // Test
        // templates.push_back({
        //     "Test",
        //     "nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert  ! appsink",
        //     "Test pipeline",
        //     4
        // });

        // templates.push_back({
        //     "Test2",
        //     "nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM),width=3280,height=2464,format=NV12,framerate=15/1 ! nvvidconv ! video/x-raw,width=640,height=360,format=BGRx ! videoconvert ! appsink",
        //     "Test2 pipeline",
        //     5
        // });

        templates.push_back({
            "Scaled_Output",
            "nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! nvvidconv ! video/x-raw,width=%d,height=%d,format=BGRx ! videoconvert ! appsink",
            "Sensor with hardware scaling",
            6
        });
        
        return templates;
    }
    
    // std::string build_pipeline_from_template(const PipelineTemplate& template_config, 
    //                                        const CameraConfig& config) {
    //     char pipeline_buffer[1024];
        
    //     if (template_config.name == "HW_Accelerated_BGRx") {
    //         snprintf(pipeline_buffer, sizeof(pipeline_buffer), 
    //                  template_config.template_str.c_str(),
    //                  config.sensor_id, config.width, config.height, 
    //                  config.format.c_str(), config.fps);
    //     } else {
    //         snprintf(pipeline_buffer, sizeof(pipeline_buffer), 
    //                  template_config.template_str.c_str(),
    //                  config.sensor_id, config.width, config.height);
    //     }
        
    //     return std::string(pipeline_buffer);
    // }

    std::string build_pipeline_from_template(const PipelineTemplate& template_config, 
                                        const CameraConfig& config) {
        char pipeline_buffer[1024];
        
        if (template_config.name == "Scaled_Output") {
            snprintf(pipeline_buffer, sizeof(pipeline_buffer), 
                    template_config.template_str.c_str(),
                    config.sensor_id, config.width, config.height);
        } else {
            // Other templates
            snprintf(pipeline_buffer, sizeof(pipeline_buffer), 
                    template_config.template_str.c_str(),
                    config.sensor_id);
        }
        
        return std::string(pipeline_buffer);
    }
    
    bool test_pipeline_compatibility(const std::string& pipeline) {
        cv::VideoCapture test_cap;
        
        try {
            test_cap.open(pipeline, cv::CAP_GSTREAMER);
            
            if (!test_cap.isOpened()) {
                return false;
            }
            
            cv::Mat test_frame;
            bool frame_read = test_cap.read(test_frame);
            test_cap.release();
            
            return frame_read && !test_frame.empty();
            
        } catch (const std::exception& e) {
            if (test_cap.isOpened()) {
                test_cap.release();
            }
            return false;
        }
    }
    
    std::string get_fallback_pipeline(const CameraConfig& config) {
        return "nvarguscamerasrc sensor-id=" + std::to_string(config.sensor_id) + 
               " ! videoconvert ! appsink";
    }
    
    std::string create_cache_key(const CameraConfig& config) {
        return std::to_string(config.sensor_id) + "_" +
               std::to_string(config.width) + "x" + std::to_string(config.height) + "_" +
               std::to_string(config.fps) + "fps_" + config.format;
    }

    void capture_loop() {
        while (running_) {
            cv::Mat frame;
            cap_ >> frame;
            
            if (!frame.empty()) {
                {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    latest_frame_ = frame.clone();
                }
                
                if (async_mode_ && async_callback_) {
                    async_callback_(frame);
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

// ============================================================================
// USB CAMERA IMPLEMENTATION (Header-Only)
// ============================================================================
class USBCamera : public CameraInterface {
private:
    cv::VideoCapture cap_;
    std::thread capture_thread_;
    std::function<void(const cv::Mat&)> async_callback_;
    std::atomic<bool> async_mode_{false};
    int buffer_size_ = 1;
    bool threading_enabled_ = false;
    mutable std::mutex frame_mutex_;
    cv::Mat latest_frame_;

public:
    ~USBCamera() {
        stop();
    }

    bool initialize(const CameraConfig& config) override {
        config_ = config;
        
        cap_.open(config.sensor_id);
        
        if (!cap_.isOpened()) {
            std::cerr << "Failed to open USB camera with ID: " << config.sensor_id << std::endl;
            return false;
        }

        // Set camera properties
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, config.width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config.height);
        cap_.set(cv::CAP_PROP_FPS, config.fps);
        cap_.set(cv::CAP_PROP_BUFFERSIZE, buffer_size_);
        
        std::cout << "USB Camera initialized successfully" << std::endl;
        return true;
    }

    bool start() override {
        if (running_) return true;
        if (!cap_.isOpened()) return false;

        running_ = true;
        
        if (threading_enabled_ && async_mode_) {
            capture_thread_ = std::thread(&USBCamera::capture_loop, this);
        }
        
        return true;
    }

    bool stop() override {
        if (!running_) return true;
        
        running_ = false;
        async_mode_ = false;
        
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }
        
        cap_.release();
        return true;
    }

    bool is_running() const override {
        return running_;
    }

    cv::Mat get_frame() override {
        if (!running_ || !cap_.isOpened()) {
            return cv::Mat();
        }

        cv::Mat frame;
        if (threading_enabled_) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            frame = latest_frame_.clone();
        } else {
            cap_ >> frame;
        }
        
        return frame;
    }

    bool get_frame_async(std::function<void(const cv::Mat&)> callback) override {
        if (!running_) return false;
        
        async_callback_ = callback;
        async_mode_ = true;
        
        if (threading_enabled_ && !capture_thread_.joinable()) {
            capture_thread_ = std::thread(&USBCamera::capture_loop, this);
        }
        
        return true;
    }

    std::string get_camera_type() const override {
        return "USB Camera";
    }

    bool supports_hardware_acceleration() const override {
        return false;
    }

    CameraConfig get_config() const override {
        return config_;
    }

    void set_buffer_size(int size) override {
        buffer_size_ = std::max(1, size);
        if (cap_.isOpened()) {
            cap_.set(cv::CAP_PROP_BUFFERSIZE, buffer_size_);
        }
    }

    void enable_threading(bool enable) override {
        threading_enabled_ = enable;
    }

    bool set_parameter(const std::string& param_name, double value) override {
        if (!cap_.isOpened()) return false;
        
        if (param_name == "brightness") {
            return cap_.set(cv::CAP_PROP_BRIGHTNESS, value);
        } else if (param_name == "contrast") {
            return cap_.set(cv::CAP_PROP_CONTRAST, value);
        } else if (param_name == "saturation") {
            return cap_.set(cv::CAP_PROP_SATURATION, value);
        } else if (param_name == "exposure") {
            return cap_.set(cv::CAP_PROP_EXPOSURE, value);
        }
        return false;
    }

    double get_parameter(const std::string& param_name) const override {
        if (!cap_.isOpened()) return -1.0;
        
        if (param_name == "brightness") {
            return cap_.get(cv::CAP_PROP_BRIGHTNESS);
        } else if (param_name == "contrast") {
            return cap_.get(cv::CAP_PROP_CONTRAST);
        } else if (param_name == "saturation") {
            return cap_.get(cv::CAP_PROP_SATURATION);
        } else if (param_name == "exposure") {
            return cap_.get(cv::CAP_PROP_EXPOSURE);
        }
        return -1.0;
    }

private:
    void capture_loop() {
        while (running_) {
            cv::Mat frame;
            cap_ >> frame;
            
            if (!frame.empty()) {
                {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    latest_frame_ = frame.clone();
                }
                
                if (async_mode_ && async_callback_) {
                    async_callback_(frame);
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

// ============================================================================
// VIRTUAL CAMERA IMPLEMENTATION (Header-Only)
// ============================================================================
class VirtualCamera : public CameraInterface {
private:
    std::thread capture_thread_;
    std::function<void(const cv::Mat&)> async_callback_;
    std::atomic<bool> async_mode_{false};
    bool threading_enabled_ = false;
    mutable std::mutex frame_mutex_;
    cv::Mat latest_frame_;
    int frame_counter_ = 0;

public:
    ~VirtualCamera() {
        stop();
    }

    bool initialize(const CameraConfig& config) override {
        config_ = config;
        generate_test_frame();
        std::cout << "Virtual Camera initialized successfully" << std::endl;
        return true;
    }

    bool start() override {
        if (running_) return true;
        running_ = true;
        
        if (threading_enabled_) {
            capture_thread_ = std::thread(&VirtualCamera::capture_loop, this);
        }
        
        return true;
    }

    bool stop() override {
        if (!running_) return true;
        
        running_ = false;
        async_mode_ = false;
        
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }
        
        return true;
    }

    bool is_running() const override {
        return running_;
    }

    cv::Mat get_frame() override {
        if (!running_) return cv::Mat();

        if (threading_enabled_) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            return latest_frame_.clone();
        } else {
            generate_test_frame();
            return latest_frame_.clone();
        }
    }

    bool get_frame_async(std::function<void(const cv::Mat&)> callback) override {
        if (!running_) return false;
        
        async_callback_ = callback;
        async_mode_ = true;
        
        if (threading_enabled_ && !capture_thread_.joinable()) {
            capture_thread_ = std::thread(&VirtualCamera::capture_loop, this);
        }
        
        return true;
    }

    std::string get_camera_type() const override {
        return "Virtual Camera";
    }

    bool supports_hardware_acceleration() const override {
        return false;
    }

    CameraConfig get_config() const override {
        return config_;
    }

    void set_buffer_size(int size) override {
        // Not applicable for virtual camera
        (void)size;
    }

    void enable_threading(bool enable) override {
        threading_enabled_ = enable;
    }

    bool set_parameter(const std::string& param_name, double value) override {
        // Virtual camera doesn't have real parameters, but we can simulate some
        (void)param_name;
        (void)value;
        return true;
    }

    double get_parameter(const std::string& param_name) const override {
        // Return some default values
        (void)param_name;
        return 0.5;
    }

private:
    void generate_test_frame() {
        cv::Mat frame(config_.height, config_.width, CV_8UC3);
        
        // Create a simple test pattern with moving elements
        frame.setTo(cv::Scalar(50, 50, 50));
        
        // Add some moving circles
        int center_x = (frame_counter_ * 2) % config_.width;
        int center_y = config_.height / 2;
        
        cv::circle(frame, cv::Point(center_x, center_y), 20, cv::Scalar(0, 255, 0), -1);
        cv::circle(frame, cv::Point(config_.width - center_x, center_y), 15, cv::Scalar(255, 0, 0), -1);
        
        // Add frame counter text
        std::string text = "Frame: " + std::to_string(frame_counter_);
        cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = frame.clone();
        }
        
        frame_counter_++;
    }

    void capture_loop() {
        auto frame_duration = std::chrono::milliseconds(1000 / config_.fps);
        
        while (running_) {
            auto start_time = std::chrono::steady_clock::now();
            
            generate_test_frame();
            
            if (async_mode_ && async_callback_) {
                async_callback_(latest_frame_);
            }
            
            auto end_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (elapsed < frame_duration) {
                std::this_thread::sleep_for(frame_duration - elapsed);
            }
        }
    }
};

// ============================================================================
// CAMERA FACTORY (Header-Only)
// ============================================================================
class CameraFactory {
public:
    enum class CameraType {
        JETSON_CSI,
        USB_CAMERA,
        VIRTUAL_CAMERA
    };

    static std::unique_ptr<CameraInterface> create_camera(CameraType type) {
        switch (type) {
            case CameraType::JETSON_CSI:
                return std::make_unique<JetsonCSICamera>();
            case CameraType::USB_CAMERA:
                return std::make_unique<USBCamera>();
            case CameraType::VIRTUAL_CAMERA:
                return std::make_unique<VirtualCamera>();
            default:
                std::cerr << "Unknown camera type" << std::endl;
                return nullptr;
        }
    }

    static std::unique_ptr<CameraInterface> create_auto_camera(int sensor_id = 0) {
        // Try Jetson CSI first
        if (std::filesystem::exists("/proc/device-tree/tegra-camera-platform")) {
            auto jetson_camera = std::make_unique<JetsonCSICamera>();
            CameraConfig config;
            config.sensor_id = sensor_id;
            if (jetson_camera->initialize(config)) {
                return jetson_camera;
            }
        }
        
        // Fall back to USB camera
        auto usb_camera = std::make_unique<USBCamera>();
        CameraConfig config;
        config.sensor_id = sensor_id;
        if (usb_camera->initialize(config)) {
            return usb_camera;
        }
        
        // Last resort: virtual camera
        return std::make_unique<VirtualCamera>();
    }

    static std::vector<CameraType> get_available_cameras() {
        std::vector<CameraType> available_cameras;
        
        // Check for Jetson CSI cameras
        if (std::filesystem::exists("/proc/device-tree/tegra-camera-platform")) {
            available_cameras.push_back(CameraType::JETSON_CSI);
        }
        
        // Check for USB cameras
        for (int i = 0; i < 4; ++i) {
            cv::VideoCapture temp_cap(i);
            if (temp_cap.isOpened()) {
                available_cameras.push_back(CameraType::USB_CAMERA);
                temp_cap.release();
                break; // Just add one USB camera type
            }
        }
        
        // Virtual camera is always available
        available_cameras.push_back(CameraType::VIRTUAL_CAMERA);
        
        return available_cameras;
    }
};

} // namespace jetson_stereo_camera