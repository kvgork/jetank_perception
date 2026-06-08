// Unit tests for the pure disparity-ROI reprojection math
// (jetank_perception::reproject_roi in sock_reproject.hpp).
//
// No ROS graph, camera, GPU or TF needed: we hand-build a synthetic
// DisparityImage + CameraInfo with known f, t, cx, cy and assert that the
// reprojected XYZ matches the pinhole model Z = f*t/d, X = (u-cx)*Z/f, etc.

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "jetank_perception/sock_reproject.hpp"

namespace
{

using jetank_perception::reproject_roi;
using DisparityImage = stereo_msgs::msg::DisparityImage;
using CameraInfo = sensor_msgs::msg::CameraInfo;

// Build a width x height 32FC1 disparity image with all pixels = `disp`.
DisparityImage make_disparity(
  int width, int height, float disp, double f, double t,
  float min_disparity = 0.0f, float max_disparity = 0.0f)
{
  DisparityImage msg;
  msg.f = static_cast<float>(f);
  msg.t = static_cast<float>(t);
  msg.min_disparity = min_disparity;
  msg.max_disparity = max_disparity;

  auto & img = msg.image;
  img.width = static_cast<uint32_t>(width);
  img.height = static_cast<uint32_t>(height);
  img.encoding = "32FC1";
  img.is_bigendian = 0;
  img.step = static_cast<uint32_t>(width * sizeof(float));
  img.data.resize(static_cast<size_t>(img.step) * height);

  for (int v = 0; v < height; ++v) {
    auto * row = reinterpret_cast<float *>(img.data.data() + static_cast<size_t>(v) * img.step);
    for (int u = 0; u < width; ++u) {
      row[u] = disp;
    }
  }
  return msg;
}

CameraInfo make_camera_info(int width, int height, double fx, double cx, double cy)
{
  CameraInfo ci;
  ci.width = static_cast<uint32_t>(width);
  ci.height = static_cast<uint32_t>(height);
  ci.k[0] = fx; ci.k[2] = cx;
  ci.k[4] = fx; ci.k[5] = cy;
  ci.k[8] = 1.0;
  return ci;
}

constexpr float kEps = 1e-3f;

}  // namespace

// A single pixel at the principal point reprojects to (0, 0, Z) with Z=f*t/d.
TEST(Reproject, SinglePixelAtPrincipalPoint)
{
  const double f = 500.0, t = 0.1, cx = 320.0, cy = 240.0;
  const float disp = 50.0f;   // Z = 500*0.1/50 = 1.0 m
  auto disparity = make_disparity(640, 480, disp, f, t);
  auto ci = make_camera_info(640, 480, f, cx, cy);

  // 1x1 ROI exactly at the principal point pixel (320,240).
  auto cloud = reproject_roi(disparity, ci, 320, 240, 321, 241, 0.0);

  ASSERT_EQ(cloud->size(), 1u);
  const auto & p = (*cloud)[0];
  EXPECT_NEAR(p.x, 0.0f, kEps);
  EXPECT_NEAR(p.y, 0.0f, kEps);
  EXPECT_NEAR(p.z, 1.0f, kEps);   // f*t/d
}

// An off-axis pixel reprojects with the expected X,Y offsets.
TEST(Reproject, OffAxisPixelMatchesPinhole)
{
  const double f = 500.0, t = 0.1, cx = 320.0, cy = 240.0;
  const float disp = 25.0f;   // Z = 500*0.1/25 = 2.0 m
  auto disparity = make_disparity(640, 480, disp, f, t);
  auto ci = make_camera_info(640, 480, f, cx, cy);

  // Pixel (420, 340): du=100, dv=100.
  auto cloud = reproject_roi(disparity, ci, 420, 340, 421, 341, 0.0);

  ASSERT_EQ(cloud->size(), 1u);
  const auto & p = (*cloud)[0];
  const double Z = f * t / disp;            // 2.0
  EXPECT_NEAR(p.z, static_cast<float>(Z), kEps);
  EXPECT_NEAR(p.x, static_cast<float>((420 - cx) * Z / f), kEps);  // 0.4
  EXPECT_NEAR(p.y, static_cast<float>((340 - cy) * Z / f), kEps);  // 0.4
}

// ROI of N pixels with uniform disparity yields N points all at the same depth.
TEST(Reproject, UniformRoiCountAndDepth)
{
  const double f = 600.0, t = 0.12, cx = 320.0, cy = 240.0;
  const float disp = 40.0f;   // Z = 600*0.12/40 = 1.8 m
  auto disparity = make_disparity(640, 480, disp, f, t);
  auto ci = make_camera_info(640, 480, f, cx, cy);

  // 10x8 = 80-pixel ROI.
  auto cloud = reproject_roi(disparity, ci, 100, 100, 110, 108, 0.0);
  EXPECT_EQ(cloud->size(), 80u);
  const float expectedZ = static_cast<float>(f * t / disp);
  for (const auto & p : *cloud) {
    EXPECT_NEAR(p.z, expectedZ, kEps);
  }
}

// Invalid disparities (<=0) are dropped.
TEST(Reproject, DropsNonPositiveDisparity)
{
  const double f = 500.0, t = 0.1, cx = 320.0, cy = 240.0;
  auto disparity = make_disparity(64, 48, 0.0f, f, t);  // all zero -> invalid
  auto ci = make_camera_info(64, 48, f, cx, cy);

  auto cloud = reproject_roi(disparity, ci, 0, 0, 64, 48, 0.0);
  EXPECT_EQ(cloud->size(), 0u);
}

// max_range clips far points (large Z) away.
TEST(Reproject, MaxRangeClips)
{
  const double f = 500.0, t = 0.1, cx = 320.0, cy = 240.0;
  const float disp = 10.0f;   // Z = 500*0.1/10 = 5.0 m
  auto disparity = make_disparity(64, 48, disp, f, t);
  auto ci = make_camera_info(64, 48, f, cx, cy);

  // Z = 5.0 m; clip at 2.0 m -> everything dropped.
  auto clipped = reproject_roi(disparity, ci, 0, 0, 64, 48, 2.0);
  EXPECT_EQ(clipped->size(), 0u);

  // Clip at 10.0 m -> all kept.
  auto kept = reproject_roi(disparity, ci, 0, 0, 64, 48, 10.0);
  EXPECT_EQ(kept->size(), 64u * 48u);
}

// min/max disparity gate (when set) excludes out-of-range disparities.
TEST(Reproject, DisparityGate)
{
  const double f = 500.0, t = 0.1, cx = 320.0, cy = 240.0;
  const float disp = 5.0f;
  // Gate [10, 100]; disp=5 is below min -> excluded.
  auto disparity = make_disparity(32, 32, disp, f, t, 10.0f, 100.0f);
  auto ci = make_camera_info(32, 32, f, cx, cy);

  auto cloud = reproject_roi(disparity, ci, 0, 0, 32, 32, 0.0);
  EXPECT_EQ(cloud->size(), 0u);
}

// ROI is clamped to the image; an out-of-bounds ROI is safe.
TEST(Reproject, RoiClampedToImage)
{
  const double f = 500.0, t = 0.1, cx = 16.0, cy = 16.0;
  const float disp = 50.0f;
  auto disparity = make_disparity(32, 32, disp, f, t);
  auto ci = make_camera_info(32, 32, f, cx, cy);

  // ROI extends well past the 32x32 image; only the in-image overlap counts.
  auto cloud = reproject_roi(disparity, ci, 16, 16, 1000, 1000, 0.0);
  EXPECT_EQ(cloud->size(), 16u * 16u);  // [16,32) x [16,32)
}

// fx from CameraInfo is used when DisparityImage.f is unset (<=0).
TEST(Reproject, FallsBackToCameraInfoFocal)
{
  const double f = 700.0, t = 0.1, cx = 320.0, cy = 240.0;
  const float disp = 70.0f;   // Z = 700*0.1/70 = 1.0 m
  // DisparityImage.f = 0 -> should fall back to CameraInfo K[0] = 700.
  auto disparity = make_disparity(640, 480, disp, 0.0, t);
  auto ci = make_camera_info(640, 480, f, cx, cy);

  auto cloud = reproject_roi(disparity, ci, 320, 240, 321, 241, 0.0);
  ASSERT_EQ(cloud->size(), 1u);
  EXPECT_NEAR((*cloud)[0].z, 1.0f, kEps);
}
