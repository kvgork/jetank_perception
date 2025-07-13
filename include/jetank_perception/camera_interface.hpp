#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <atomic>
#include <functional>

namespace jetson_stereo_camera {

struct CameraConfig {
    int width;
    int height;
    int fps;
    int sensor_id;
    std::string format;
    bool use_hardware_acceleration;
    
    CameraConfig(int w = 640, int h = 480, int f = 20, int id = 0, 
                const std::string& fmt = "GRAY8", bool hw_accel = true)
        : width(w), height(h), fps(f), sensor_id(id), format(fmt), use_hardware_acceleration(hw_accel) {}
};

class CameraInterface {
public:
    virtual ~CameraInterface() = default;
    
    // Core camera operations
    virtual bool initialize(const CameraConfig& config) = 0;
    virtual bool start() = 0;
    virtual bool stop() = 0;
    virtual bool is_running() const = 0;
    
    // Frame acquisition
    virtual cv::Mat get_frame() = 0;
    virtual bool get_frame_async(std::function<void(const cv::Mat&)> callback) = 0;
    
    // Camera properties
    virtual std::string get_camera_type() const = 0;
    virtual bool supports_hardware_acceleration() const = 0;
    virtual CameraConfig get_config() const = 0;
    
    // Performance optimization
    virtual void set_buffer_size(int size) = 0;
    virtual void enable_threading(bool enable) = 0;
    
protected:
    std::atomic<bool> running_{false};
    CameraConfig config_;
};

// Factory for creating camera instances
class CameraFactory {
public:
    enum class CameraType {
        JETSON_CSI,
        USB_CAMERA,
        VIRTUAL_CAMERA
    };
    
    static std::unique_ptr<CameraInterface> create_camera(CameraType type);
    static std::vector<std::string> get_available_cameras();
};

}