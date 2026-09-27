#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>
#include <algorithm>

namespace jetson_stereo_camera
{

// Configuration structures
struct CameraConfig
{
  int width = 640;
  int height = 480;
  int fps = 30;
  int sensor_id = 0;
  std::string format = "BGR";
  bool use_hardware_acceleration = true;
  bool flip_180 = false;
};

// Abstract camera interface
class CameraInterface
{
public:
  virtual ~CameraInterface() = default;

  // Core interface methods
  virtual bool initialize(const CameraConfig & config) = 0;
  virtual bool start() = 0;
  virtual bool stop() = 0;
  virtual bool is_running() const = 0;
  virtual cv::Mat get_frame() = 0;

  // Extended interface methods
  virtual bool get_frame_async(std::function<void(const cv::Mat &)> callback) = 0;
  virtual std::string get_camera_type() const = 0;
  virtual bool supports_hardware_acceleration() const = 0;
  virtual CameraConfig get_config() const = 0;
  virtual void set_buffer_size(int size) = 0;
  virtual void enable_threading(bool enable) = 0;

  // Parameter interface (for camera controls)
  virtual bool set_parameter(const std::string & param_name, double value) = 0;
  virtual double get_parameter(const std::string & param_name) const = 0;

protected:
  CameraConfig config_;
  std::atomic<bool> running_{false};
};

// ============================================================================
// JETSON CSI CAMERA IMPLEMENTATION (Header-Only)
// ============================================================================
class JetsonCSICamera : public CameraInterface
{
private:
  cv::VideoCapture cap_;
  std::thread capture_thread_;
  std::function<void(const cv::Mat &)> async_callback_;
  std::atomic<bool> async_mode_{false};
  int buffer_size_ = 1;
  bool threading_enabled_ = false;
  mutable std::mutex frame_mutex_;
  cv::Mat latest_frame_;

public:
  ~JetsonCSICamera()
  {
    stop();
  }

  bool initialize(const CameraConfig & config) override
  {
    config_ = config;

    std::string pipeline = make_pipeline(config);

    cap_.open(pipeline, cv::CAP_GSTREAMER);

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
        std::cout << "Actual frame size: " << test_frame.cols << "x" << test_frame.rows <<
          std::endl;
        std::cout << "Frame channels: " << test_frame.channels() << std::endl;
        std::cout << "Frame type: " << test_frame.type() << std::endl;
      }
      std::cout << "=============================" << std::endl;
    }

    if (!cap_.isOpened()) {
      std::cerr << "Failed to open CSI camera with pipeline: " << pipeline << std::endl;
      return false;
    }

    std::cout << "Jetson CSI Camera initialized successfully" << std::endl;
    return true;
  }

  bool start() override
  {
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

  bool stop() override
  {
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

  bool is_running() const override
  {
    return running_;
  }

  cv::Mat get_frame() override
  {
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

  bool get_frame_async(std::function<void(const cv::Mat &)> callback) override
  {
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

  std::string get_camera_type() const override
  {
    return "Jetson CSI Camera";
  }

  bool supports_hardware_acceleration() const override
  {
    return true;
  }

  CameraConfig get_config() const override
  {
    return config_;
  }

  void set_buffer_size(int size) override
  {
    // CAP_PROP_BUFFERSIZE is a no-op for OpenCV's GStreamer backend; queue
    // depth is bounded in the pipeline itself via appsink's
    // max-buffers=1 drop=true (see make_pipeline()). Kept as a stored value
    // only so callers built against this interface still compile.
    buffer_size_ = std::max(1, size);
  }

  void enable_threading(bool enable) override
  {
    threading_enabled_ = enable;
  }

  bool set_parameter(const std::string & param_name, double value) override
  {
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

  double get_parameter(const std::string & param_name) const override
  {
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
  // Build the nvarguscamerasrc pipeline for this config. GRAY8 lets
  // nvvidconv output grayscale directly in hardware, removing the CPU
  // videoconvert stage and the consumer's BGR->GRAY cvtColor. BGR8 (the
  // shipped default, config/stereo_camera_config.yaml) is colour-preserving
  // but similarly skips the CPU videoconvert stage: nvvidconv outputs BGRx
  // (4-channel) directly, appsink hands that straight to OpenCV as a
  // CV_8UC4 Mat, and the consumer (stereo_camera_node.cpp) publishes it as
  // bgra8 and uses cv::COLOR_BGRA2GRAY where a single channel is needed.
  // This is what removes the CPU videoconvert + BGR2GRAY double-conversion
  // jetank_perception-04 measured, without dropping colour for the raw/web
  // viewer path. Any other requested format falls back to the original
  // BGRx + CPU videoconvert path, since it is not one this node's
  // consumers have been made 4-channel-aware for. max-buffers=1 drop=true
  // sync=false bounds appsink's queue depth itself, since
  // CAP_PROP_BUFFERSIZE is a no-op for OpenCV's GStreamer backend.
  //
  // NOTE: nvvidconv -> BGRx in system memory was already the shipped path; this
  // branch only drops the CPU videoconvert element. The real dependency is
  // OpenCV's GStreamer appsink caps fallback (>= 4.5.1), which accepts BGRx and
  // delivers CV_8UC4 frames; both OpenCV builds on the Jetson qualify. Runtime
  // check: the "Frame channels:" debug print below should report 4.
  //
  // This replaces a template/cache/fallback/compatibility-test system that
  // only ever had one real template: the "cache" was a single entry keyed
  // by config, the "fallback" (plain videoconvert, no nvvidconv, no size
  // caps) could not have negotiated NVMM output or honoured the requested
  // resolution, and the compatibility test opened a second full camera
  // capture (with a frame read) per sensor at startup purely to validate
  // the one template that always ran.
  std::string make_pipeline(const CameraConfig & config)
  {
    const char * caps_template;
    if (config.format == "GRAY8") {
      caps_template =
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=%d/1 ! "
        "nvvidconv ! video/x-raw,format=GRAY8,width=%d,height=%d ! "
        "appsink max-buffers=1 drop=true sync=false";
    } else if (config.format == "BGR8") {
      caps_template =
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=%d/1 ! "
        "nvvidconv ! video/x-raw,format=BGRx,width=%d,height=%d ! "
        "appsink max-buffers=1 drop=true sync=false";
    } else {
      caps_template =
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=%d/1 ! "
        "nvvidconv ! video/x-raw,width=%d,height=%d,format=BGRx ! videoconvert ! "
        "appsink max-buffers=1 drop=true sync=false";
    }

    char pipeline_buffer[512];
    snprintf(
      pipeline_buffer, sizeof(pipeline_buffer),
      caps_template, config.sensor_id, config.fps, config.width, config.height);

    return std::string(pipeline_buffer);
  }

  void capture_loop()
  {
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
// CAMERA FACTORY (Header-Only)
// ============================================================================
class CameraFactory
{
public:
  enum class CameraType
  {
    JETSON_CSI
  };

  static std::unique_ptr<CameraInterface> create_camera(CameraType type)
  {
    switch (type) {
      case CameraType::JETSON_CSI:
        return std::make_unique<JetsonCSICamera>();
      default:
        std::cerr << "Unknown camera type" << std::endl;
        return nullptr;
    }
  }
};

} // namespace jetson_stereo_camera
