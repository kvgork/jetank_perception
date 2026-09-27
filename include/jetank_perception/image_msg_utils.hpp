#pragma once

#include <sensor_msgs/msg/image.hpp>
#include <opencv2/core.hpp>
#include <cstring>
#include <memory>
#include <string>

namespace jetson_stereo_camera
{

// Minimal cv::Mat -> sensor_msgs::msg::Image encoder, replacing
// cv_bridge::CvImage(...).toImageMsg() for the call sites in this package
// that only need to wrap an already-correctly-encoded cv::Mat (no pixel
// format conversion). cv_bridge links against a system OpenCV build that
// differs from this workspace's OpenCV, so every cv_bridge call site is a
// second OpenCV runtime loaded into the process (see jetank_perception-14);
// removing these trivial wrap sites is what lets the package drop the
// cv_bridge dependency entirely.
/// Encoding label for a raw capture frame by channel count: 1 -> mono8,
/// 4 -> bgra8 (BGR8-config BGRx capture without CPU videoconvert),
/// otherwise bgr8 (CPU-videoconvert fallback path).
inline const char * raw_frame_encoding(const cv::Mat & frame)
{
  switch (frame.channels()) {
    case 1: return "mono8";
    case 4: return "bgra8";
    default: return "bgr8";
  }
}

inline std::shared_ptr<sensor_msgs::msg::Image> mat_to_image_msg(
  const std::string & encoding, const cv::Mat & image)
{
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = static_cast<uint32_t>(image.rows);
  msg->width = static_cast<uint32_t>(image.cols);
  msg->encoding = encoding;
  msg->is_bigendian = 0;
  msg->step = static_cast<uint32_t>(image.step);

  const size_t row_bytes = static_cast<size_t>(image.step);
  const size_t size = row_bytes * static_cast<size_t>(image.rows);
  msg->data.resize(size);
  if (image.isContinuous()) {
    std::memcpy(msg->data.data(), image.data, size);
  } else {
    for (int r = 0; r < image.rows; ++r) {
      std::memcpy(msg->data.data() + static_cast<size_t>(r) * row_bytes, image.ptr(r), row_bytes);
    }
  }
  return msg;
}

}  // namespace jetson_stereo_camera
