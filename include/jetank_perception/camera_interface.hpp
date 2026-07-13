#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <functional>
#include <vector>
#include <map>
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

// Pipeline template structure for robust pipeline building
struct PipelineTemplate
{
  std::string name;
  std::string template_str;
  std::string description;
  int priority;
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

  // Static pipeline cache for GStreamer pipelines
  static std::map<std::string, std::string> pipeline_cache_;
};

// Initialize the static pipeline cache
std::map<std::string, std::string> CameraInterface::pipeline_cache_;

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

    // Use robust pipeline building with testing and caching
    std::string pipeline = build_gstreamer_pipeline(config);

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

    // Set buffer size for performance
    cap_.set(cv::CAP_PROP_BUFFERSIZE, buffer_size_);

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
    buffer_size_ = std::max(1, size);
    if (cap_.isOpened()) {
      cap_.set(cv::CAP_PROP_BUFFERSIZE, buffer_size_);
    }
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
  // Robust pipeline builder with testing and caching
  std::string build_gstreamer_pipeline(const CameraConfig & config)
  {
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

    std::cout << "[PIPELINE] Testing " << templates.size() << " pipeline configurations" <<
      std::endl;

    // Test each pipeline template
    for (const auto & template_config : templates) {
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

  std::vector<PipelineTemplate> get_pipeline_templates()
  {
    std::vector<PipelineTemplate> templates;

    templates.push_back(
      {
        "Scaled_Output",
        "nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! nvvidconv ! video/x-raw,width=%d,height=%d,format=BGRx ! videoconvert ! appsink",
        "Sensor with hardware scaling",
        6
      });

    return templates;
  }

  std::string build_pipeline_from_template(
    const PipelineTemplate & template_config,
    const CameraConfig & config)
  {
    char pipeline_buffer[1024];

    if (template_config.name == "Scaled_Output") {
      snprintf(
        pipeline_buffer, sizeof(pipeline_buffer),
        template_config.template_str.c_str(),
        config.sensor_id, config.width, config.height);
    } else {
      // Other templates
      snprintf(
        pipeline_buffer, sizeof(pipeline_buffer),
        template_config.template_str.c_str(),
        config.sensor_id);
    }

    return std::string(pipeline_buffer);
  }

  bool test_pipeline_compatibility(const std::string & pipeline)
  {
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

    } catch (const std::exception & e) {
      if (test_cap.isOpened()) {
        test_cap.release();
      }
      return false;
    }
  }

  std::string get_fallback_pipeline(const CameraConfig & config)
  {
    return "nvarguscamerasrc sensor-id=" + std::to_string(config.sensor_id) +
           " ! videoconvert ! appsink";
  }

  std::string create_cache_key(const CameraConfig & config)
  {
    return std::to_string(config.sensor_id) + "_" +
           std::to_string(config.width) + "x" + std::to_string(config.height) + "_" +
           std::to_string(config.fps) + "fps_" + config.format;
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
