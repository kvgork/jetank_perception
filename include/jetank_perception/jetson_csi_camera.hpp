// include/jetson_stereo_camera/jetson_csi_camera.hpp
#pragma once

#include "camera_interface.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace jetson_stereo_camera {

class JetsonCSICamera : public CameraInterface {
public:
    JetsonCSICamera();
    ~JetsonCSICamera() override;
    
    // CameraInterface implementation
    bool initialize(const CameraConfig& config) override;
    bool start() override;
    bool stop() override;
    bool is_running() const override;
    
    cv::Mat get_frame() override;
    bool get_frame_async(std::function<void(const cv::Mat&)> callback) override;
    
    std::string get_camera_type() const override { return "Jetson CSI"; }
    bool supports_hardware_acceleration() const override { return true; }
    CameraConfig get_config() const override { return config_; }
    
    void set_buffer_size(int size) override;
    void enable_threading(bool enable) override;

private:
    // GStreamer components
    GstElement* pipeline_;
    GstElement* sink_;
    
    // Threading and buffering
    std::unique_ptr<std::thread> capture_thread_;
    std::queue<cv::Mat> frame_buffer_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_cv_;
    int max_buffer_size_;
    bool threading_enabled_;
    
    // Async callback
    std::function<void(const cv::Mat&)> async_callback_;
    
    // Private methods
    void setup_pipeline();
    void capture_loop();
    cv::Mat extract_frame_from_sample(GstSample* sample);
    std::string build_pipeline_string() const;
    
    // Performance optimization
    void optimize_pipeline_for_jetson();
    void configure_nvargus_properties();
};

} // namespace jetson_stereo_camera

// Implementation file: src/jetson_csi_camera.cpp
#include "jetank_perception/jetson_csi_camera.hpp"
#include <sstream>
#include <iostream>

namespace jetson_stereo_camera {

JetsonCSICamera::JetsonCSICamera() 
    : pipeline_(nullptr), sink_(nullptr), max_buffer_size_(2), threading_enabled_(true) {
    // Initialize GStreamer if not already done
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
}

JetsonCSICamera::~JetsonCSICamera() {
    stop();
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
    }
}

bool JetsonCSICamera::initialize(const CameraConfig& config) {
    config_ = config;
    
    try {
        setup_pipeline();
        optimize_pipeline_for_jetson();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Jetson CSI camera: " << e.what() << std::endl;
        return false;
    }
}

bool JetsonCSICamera::start() {
    if (running_) return true;
    
    if (!pipeline_) {
        std::cerr << "Pipeline not initialized" << std::endl;
        return false;
    }
    
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Failed to start GStreamer pipeline" << std::endl;
        return false;
    }
    
    running_ = true;
    
    // Start capture thread if threading is enabled
    if (threading_enabled_) {
        capture_thread_ = std::make_unique<std::thread>(&JetsonCSICamera::capture_loop, this);
    }
    
    return true;
}

bool JetsonCSICamera::stop() {
    if (!running_) return true;
    
    running_ = false;
    
    // Stop capture thread
    if (capture_thread_ && capture_thread_->joinable()) {
        buffer_cv_.notify_all();
        capture_thread_->join();
        capture_thread_.reset();
    }
    
    // Stop pipeline
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
    }
    
    // Clear buffer
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        while (!frame_buffer_.empty()) {
            frame_buffer_.pop();
        }
    }
    
    return true;
}

bool JetsonCSICamera::is_running() const {
    return running_;
}

cv::Mat JetsonCSICamera::get_frame() {
    if (!running_) return cv::Mat();
    
    if (threading_enabled_) {
        // Get frame from buffer
        std::unique_lock<std::mutex> lock(buffer_mutex_);
        if (frame_buffer_.empty()) {
            buffer_cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
        
        if (!frame_buffer_.empty()) {
            cv::Mat frame = frame_buffer_.front();
            frame_buffer_.pop();
            return frame;
        }
    } else {
        // Direct capture
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
        if (sample) {
            cv::Mat frame = extract_frame_from_sample(sample);
            gst_sample_unref(sample);
            return frame;
        }
    }
    
    return cv::Mat();
}

bool JetsonCSICamera::get_frame_async(std::function<void(const cv::Mat&)> callback) {
    if (!running_) return false;
    
    async_callback_ = callback;
    return true;
}

void JetsonCSICamera::set_buffer_size(int size) {
    max_buffer_size_ = std::max(1, size);
}

void JetsonCSICamera::enable_threading(bool enable) {
    threading_enabled_ = enable;
}

void JetsonCSICamera::setup_pipeline() {
    std::string pipeline_str = build_pipeline_string();
    
    GError* error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    
    if (!pipeline_ || error) {
        std::string error_msg = error ? error->message : "Unknown error";
        if (error) g_error_free(error);
        throw std::runtime_error("Failed to create GStreamer pipeline: " + error_msg);
    }
    
    sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!sink_) {
        throw std::runtime_error("Failed to get sink element from pipeline");
    }
    
    // Configure sink properties for optimal performance
    g_object_set(sink_, "max-buffers", max_buffer_size_, nullptr);
    g_object_set(sink_, "drop", TRUE, nullptr);
    g_object_set(sink_, "sync", FALSE, nullptr);
}

std::string JetsonCSICamera::build_pipeline_string() const {
    std::stringstream ss;
    
    ss << "nvarguscamerasrc sensor-id=" << config_.sensor_id;
    
    // Add Jetson-specific optimizations
    ss << " bufapi-version=1";  // Use newer buffer API
    ss << " maxperf=1";         // Maximum performance mode
    ss << " wbmode=1";          // Auto white balance
    ss << " saturation=1.0";
    ss << " exposuretimerange=\"13000 683709000\"";  // Exposure range for IMX219
    ss << " ispdigitalgainrange=\"1 8\"";
    
    ss << " ! video/x-raw(memory:NVMM)";
    ss << ",width=" << config_.width;
    ss << ",height=" << config_.height;
    ss << ",format=NV12";
    ss << ",framerate=" << config_.fps << "/1";
    
    if (config_.use_hardware_acceleration) {
        ss << " ! nvvidconv flip-method=2";  // Rotate 180 degrees (equivalent to flip -1)
        ss << " ! video/x-raw,format=BGRx";
        ss << " ! videoconvert";
    }
    
    if (config_.format == "GRAY8") {
        ss << " ! video/x-raw,format=GRAY8";
    } else {
        ss << " ! video/x-raw,format=BGR";
    }
    
    ss << " ! appsink name=sink";
    ss << " max-buffers=" << max_buffer_size_;
    ss << " drop=true";
    ss << " sync=false";
    
    return ss.str();
}

void JetsonCSICamera::optimize_pipeline_for_jetson() {
    // Set additional properties for Jetson Orin Nano optimization
    if (GstElement* nvargus = gst_bin_get_by_name(GST_BIN(pipeline_), "nvarguscamerasrc0")) {
        // Enable sensor mode for best performance
        g_object_set(nvargus, "sensor-mode", 0, nullptr);  // Mode 0 for 640x480
        g_object_set(nvargus, "ee-mode", 0, nullptr);      // Disable edge enhancement
        g_object_set(nvargus, "ee-strength", 0.0, nullptr);
        g_object_set(nvargus, "tnr-mode", 0, nullptr);     // Disable temporal noise reduction
        g_object_set(nvargus, "tnr-strength", 0.0, nullptr);
        gst_object_unref(nvargus);
    }
}

void JetsonCSICamera::configure_nvargus_properties() {
    // Additional nvargus configuration for IMX219-83
    // This can be expanded based on specific camera requirements
}

void JetsonCSICamera::capture_loop() {
    while (running_) {
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
        if (!sample) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        cv::Mat frame = extract_frame_from_sample(sample);
        gst_sample_unref(sample);
        
        if (frame.empty()) continue;
        
        // Handle async callback
        if (async_callback_) {
            async_callback_(frame);
        }
        
        // Add to buffer
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            if (frame_buffer_.size() >= static_cast<size_t>(max_buffer_size_)) {
                frame_buffer_.pop();  // Remove oldest frame
            }
            frame_buffer_.push(frame.clone());
        }
        buffer_cv_.notify_one();
    }
}

cv::Mat JetsonCSICamera::extract_frame_from_sample(GstSample* sample) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) return cv::Mat();
    
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        return cv::Mat();
    }
    
    cv::Mat frame;
    if (config_.format == "GRAY8") {
        frame = cv::Mat(config_.height, config_.width, CV_8UC1, map.data).clone();
    } else {
        frame = cv::Mat(config_.height, config_.width, CV_8UC3, map.data).clone();
    }
    
    gst_buffer_unmap(buffer, &map);
    return frame;
}

}