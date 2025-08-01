#include "jetank_perception/camera_interface.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <mutex>

namespace jetson_stereo_camera {

// Initialize the static pipeline cache
std::map<std::string, std::string> CameraInterface::pipeline_cache_;

// =============================================================================
// Jetson CSI Camera Implementation
// =============================================================================
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

private:
    // NEW: Robust pipeline builder with testing and caching
    std::string build_gstreamer_pipeline(const CameraConfig& config) {
        // Create a cache key based on configuration
        std::string cache_key = create_cache_key(config);
        
        // Check if we have a cached working pipeline for this configuration
        auto cached_it = pipeline_cache_.find(cache_key);
        if (cached_it != pipeline_cache_.end()) {
            std::cout << "[PIPELINE] Using cached pipeline for " << cache_key << std::endl;
            std::cout << "[PIPELINE] " << cached_it->second << std::endl;
            return cached_it->second;
        }
        
        // Define pipeline templates in order of preference
        std::vector<PipelineTemplate> templates = get_pipeline_templates(config);
        
        std::cout << "[PIPELINE] Testing " << templates.size() << " pipeline configurations for:" << std::endl;
        std::cout << "[PIPELINE]   Resolution: " << config.width << "x" << config.height << std::endl;
        std::cout << "[PIPELINE]   FPS: " << config.fps << ", Format: " << config.format << std::endl;
        std::cout << "[PIPELINE]   Sensor ID: " << config.sensor_id << std::endl;
        std::cout << "[PIPELINE]   HW Accel: " << (config.use_hardware_acceleration ? "Yes" : "No") << std::endl;
        
        // Test each pipeline template
        for (const auto& template_config : templates) {
            std::string pipeline = build_pipeline_from_template(template_config, config);
            
            std::cout << "[PIPELINE] Testing: " << template_config.name << std::endl;
            std::cout << "[PIPELINE]   Description: " << template_config.description << std::endl;
            std::cout << "[PIPELINE]   Pipeline: " << pipeline << std::endl;
            
            if (test_pipeline_compatibility(pipeline)) {
                std::cout << "[PIPELINE] ✓ SUCCESS: " << template_config.name << " works!" << std::endl;
                
                // Cache this successful configuration
                pipeline_cache_[cache_key] = pipeline;
                
                return pipeline;
            } else {
                std::cout << "[PIPELINE] ✗ FAILED: " << template_config.name << " - trying next option" << std::endl;
            }
        }
        
        // If all templates fail, return a minimal fallback
        std::string fallback = get_fallback_pipeline(config);
        std::cout << "[PIPELINE] ⚠ WARNING: All templates failed, using fallback:" << std::endl;
        std::cout << "[PIPELINE] " << fallback << std::endl;
        
        return fallback;
    }
    
    // NEW: Get pipeline templates in order of preference
    std::vector<PipelineTemplate> get_pipeline_templates(const CameraConfig& config) {
        // Note: config parameter reserved for future template customization
        (void)config; // Suppress unused parameter warning
        std::vector<PipelineTemplate> templates;
        
        // Template 1: High-performance, hardware-accelerated (best for stereo vision)
        templates.push_back({
            "HW_Accelerated_BGRx",
            "nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM), width=%d, height=%d, format=%s, framerate=%d/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
            "Hardware accelerated with NVMM memory, BGRx->BGR conversion",
            1
        });
        
        // Template 2: Hardware-accelerated with RGBA (good fallback)
        templates.push_back({
            "HW_Accelerated_RGBA",
            "nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM), width=%d, height=%d, format=%s, framerate=%d/1 ! nvvidconv ! video/x-raw, format=RGBA ! videoconvert ! video/x-raw, format=BGR ! appsink",
            "Hardware accelerated with NVMM memory, RGBA->BGR conversion",
            2
        });
        
        // Template 3: Simple hardware-accelerated (auto-negotiated format)
        templates.push_back({
            "HW_Accelerated_Auto",
            "nvarguscamerasrc sensor-id=%d ! nvvidconv ! video/x-raw, width=%d, height=%d ! videoconvert ! video/x-raw, format=BGR ! appsink",
            "Hardware accelerated with auto format negotiation",
            3
        });
        
        // Template 4: Software fallback (no hardware acceleration)
        templates.push_back({
            "Software_Fallback",
            "nvarguscamerasrc sensor-id=%d ! videoconvert ! video/x-raw, width=%d, height=%d, format=BGR ! appsink",
            "Software-only conversion (slower but compatible)",
            4
        });
        
        // Template 5: Minimal pipeline (last resort)
        templates.push_back({
            "Minimal",
            "nvarguscamerasrc sensor-id=%d ! videoconvert ! video/x-raw, format=BGR ! appsink",
            "Minimal pipeline with no size constraints",
            5
        });
        
        // Sort by priority (lower number = higher priority)
        std::sort(templates.begin(), templates.end(), 
                  [](const PipelineTemplate& a, const PipelineTemplate& b) {
                      return a.priority < b.priority;
                  });
        
        return templates;
    }
    
    // NEW: Build pipeline string from template
    std::string build_pipeline_from_template(const PipelineTemplate& template_config, 
                                           const CameraConfig& config) {
        char pipeline_buffer[1024];
        
        // Different templates need different parameters
        if (template_config.name == "Minimal") {
            snprintf(pipeline_buffer, sizeof(pipeline_buffer), 
                     template_config.template_str.c_str(),
                     config.sensor_id);
        } else if (template_config.name == "Software_Fallback") {
            snprintf(pipeline_buffer, sizeof(pipeline_buffer), 
                     template_config.template_str.c_str(),
                     config.sensor_id, config.width, config.height);
        } else if (template_config.name == "HW_Accelerated_Auto") {
            snprintf(pipeline_buffer, sizeof(pipeline_buffer), 
                     template_config.template_str.c_str(),
                     config.sensor_id, config.width, config.height);
        } else {
            // Full parameter templates
            snprintf(pipeline_buffer, sizeof(pipeline_buffer), 
                     template_config.template_str.c_str(),
                     config.sensor_id, config.width, config.height, 
                     config.format.c_str(), config.fps);
        }
        
        return std::string(pipeline_buffer);
    }
    
    // NEW: Test if a pipeline can be created and used
    bool test_pipeline_compatibility(const std::string& pipeline) {
        // Test if the pipeline can be created and initialized
        cv::VideoCapture test_cap;
        
        try {
            // Try to open the pipeline
            test_cap.open(pipeline, cv::CAP_GSTREAMER);
            
            if (!test_cap.isOpened()) {
                return false;
            }
            
            // Try to read one frame to ensure it actually works
            cv::Mat test_frame;
            bool frame_read = test_cap.read(test_frame);
            
            test_cap.release();
            
            return frame_read && !test_frame.empty();
            
        } catch (const std::exception& e) {
            std::cout << "[PIPELINE]   Exception during test: " << e.what() << std::endl;
            if (test_cap.isOpened()) {
                test_cap.release();
            }
            return false;
        }
    }
    
    // NEW: Get minimal fallback pipeline
    std::string get_fallback_pipeline(const CameraConfig& config) {
        // Most basic pipeline that should work on any system
        return "nvarguscamerasrc sensor-id=" + std::to_string(config.sensor_id) + 
               " ! videoconvert ! appsink";
    }
    
    // NEW: Create cache key for configuration
    std::string create_cache_key(const CameraConfig& config) {
        // Create a unique key for this configuration
        return std::to_string(config.sensor_id) + "_" +
               std::to_string(config.width) + "x" + std::to_string(config.height) + "_" +
               std::to_string(config.fps) + "fps_" + config.format + "_" +
               (config.use_hardware_acceleration ? "hw" : "sw");
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

// =============================================================================
// USB Camera Implementation (unchanged)
// =============================================================================
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
        if (running_) {
            return true;
        }
        
        if (!cap_.isOpened()) {
            std::cerr << "Camera not initialized" << std::endl;
            return false;
        }

        running_ = true;
        
        if (threading_enabled_ && async_mode_) {
            capture_thread_ = std::thread(&USBCamera::capture_loop, this);
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
        
        return frame;
    }

    bool get_frame_async(std::function<void(const cv::Mat&)> callback) override {
        if (!running_) {
            return false;
        }
        
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

// =============================================================================
// Virtual Camera Implementation (unchanged)
// =============================================================================
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
        
        // Create a test pattern
        generate_test_frame();
        
        std::cout << "Virtual Camera initialized successfully" << std::endl;
        return true;
    }

    bool start() override {
        if (running_) {
            return true;
        }

        running_ = true;
        
        if (threading_enabled_) {
            capture_thread_ = std::thread(&VirtualCamera::capture_loop, this);
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
        
        return true;
    }

    bool is_running() const override {
        return running_;
    }

    cv::Mat get_frame() override {
        if (!running_) {
            return cv::Mat();
        }

        if (threading_enabled_) {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            return latest_frame_.clone();
        } else {
            generate_test_frame();
            return latest_frame_.clone();
        }
    }

    bool get_frame_async(std::function<void(const cv::Mat&)> callback) override {
        if (!running_) {
            return false;
        }
        
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
        (void)size; // Suppress unused parameter warning
    }

    void enable_threading(bool enable) override {
        threading_enabled_ = enable;
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

// =============================================================================
// Camera Factory Implementation
// =============================================================================
std::unique_ptr<CameraInterface> CameraFactory::create_camera(CameraType type) {
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

std::vector<std::string> CameraFactory::get_available_cameras() {
    std::vector<std::string> available_cameras;
    
    // Check for Jetson CSI cameras
    if (std::filesystem::exists("/proc/device-tree/tegra-camera-platform")) {
        available_cameras.push_back("Jetson CSI Camera (ID: 0)");
        available_cameras.push_back("Jetson CSI Camera (ID: 1)");
    }
    
    // Check for USB cameras
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture temp_cap(i);
        if (temp_cap.isOpened()) {
            available_cameras.push_back("USB Camera (ID: " + std::to_string(i) + ")");
            temp_cap.release();
        }
    }
    
    // Virtual camera is always available
    available_cameras.push_back("Virtual Camera");
    
    return available_cameras;
}

} // namespace jetson_stereo_camera