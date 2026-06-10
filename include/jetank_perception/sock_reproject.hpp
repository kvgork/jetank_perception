// sock_reproject.hpp
//
// Pure, ROS-node-free reprojection of a disparity ROI into a 3D point cloud in
// the camera optical frame. Factored out of sock_segmentation_server.cpp so the
// math can be unit-tested without a ROS graph, camera, or GPU.
//
// Pinhole model (camera optical frame: +x right, +y down, +z forward):
//   for a pixel (u,v) with disparity d > 0:
//     Z = f*t/d,  X = (u-cx)*Z/f,  Y = (v-cy)*Z/f
// Equivalent to cv::reprojectImageTo3D with Q built from f,t,cx,cy.

#ifndef JETANK_PERCEPTION__SOCK_REPROJECT_HPP_
#define JETANK_PERCEPTION__SOCK_REPROJECT_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

#include <sensor_msgs/msg/camera_info.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace jetank_perception
{

// Reproject the disparity pixels inside the ROI [u0,u1) x [v0,v1) into a
// pcl::PointCloud<pcl::PointXYZ> in the camera optical frame.
//
//   disparity   : DisparityImage carrying a 32FC1 disparity image, f, t and
//                 (optionally) min/max_disparity gates.
//   camera_info : provides cx=K[2], cy=K[5] (and fx=K[0] if disparity.f == 0).
//   bbox        : {u0,v0,u1,v1} pixel ROI (clamped internally to the image).
//   max_range   : z clip in metres (<=0 disables).
//
// Returns a (possibly empty) cloud; never null.
inline pcl::PointCloud<pcl::PointXYZ>::Ptr reproject_roi(
  const stereo_msgs::msg::DisparityImage & disparity,
  const sensor_msgs::msg::CameraInfo & camera_info,
  int u0, int v0, int u1, int v1,
  double max_range)
{
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  const sensor_msgs::msg::Image & img = disparity.image;
  if (img.width == 0 || img.height == 0) {
    return cloud;
  }

  // Focal length: prefer the value carried by the DisparityImage; fall back to
  // fx from CameraInfo K[0] when disparity.f is unset.
  double f = disparity.f;
  if (f <= 0.0) {
    f = camera_info.k[0];
  }
  const double t = disparity.t;          // baseline (m)
  const double cx = camera_info.k[2];
  const double cy = camera_info.k[5];
  if (f <= 0.0 || t <= 0.0) {
    return cloud;                        // cannot reproject without geometry
  }

  // Optional disparity gates from the message (gate only when max > min).
  const float min_disp = disparity.min_disparity;
  const float max_disp = disparity.max_disparity;
  const bool has_disp_gate = (max_disp > min_disp);

  // Clamp the ROI to the image bounds.
  const int W = static_cast<int>(img.width);
  const int H = static_cast<int>(img.height);
  u0 = std::max(0, std::min(u0, W));
  u1 = std::max(0, std::min(u1, W));
  v0 = std::max(0, std::min(v0, H));
  v1 = std::max(0, std::min(v1, H));
  if (u1 <= u0 || v1 <= v0) {
    return cloud;
  }

  const uint8_t * base = img.data.data();
  const size_t step = img.step;          // bytes per row

  cloud->reserve(static_cast<size_t>(u1 - u0) * static_cast<size_t>(v1 - v0));

  for (int v = v0; v < v1; ++v) {
    const float * row =
      reinterpret_cast<const float *>(base + static_cast<size_t>(v) * step);
    for (int u = u0; u < u1; ++u) {
      const float d = row[u];
      if (!std::isfinite(d) || d <= 0.0f) {
        continue;
      }
      if (has_disp_gate && (d < min_disp || d > max_disp)) {
        continue;
      }
      const double Z = (f * t) / static_cast<double>(d);
      if (!std::isfinite(Z) || Z <= 0.0) {
        continue;
      }
      if (max_range > 0.0 && Z > max_range) {
        continue;
      }
      const double X = (static_cast<double>(u) - cx) * Z / f;
      const double Y = (static_cast<double>(v) - cy) * Z / f;
      cloud->emplace_back(
        static_cast<float>(X), static_cast<float>(Y), static_cast<float>(Z));
    }
  }
  cloud->width = static_cast<uint32_t>(cloud->size());
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

}  // namespace jetank_perception

#endif  // JETANK_PERCEPTION__SOCK_REPROJECT_HPP_
