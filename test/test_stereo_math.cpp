// Unit tests for pure-logic helpers in jetank_perception.
//
// These tests import the package's real headers and exercise behaviour that
// needs no camera, GPU, or ROS graph:
//   * QualityMonitoringConfig gating/validation logic (quality_monitoring.hpp)
//   * StereoProcessingStrategy FPS math and factory/strategy naming
//     (stereo_processing_strategy.hpp)

#include <gtest/gtest.h>

#include "jetank_perception/quality_monitoring.hpp"
#include "jetank_perception/stereo_processing_strategy.hpp"

using namespace jetson_stereo_camera;

// ---------------------------------------------------------------------------
// QualityMonitoringConfig
// ---------------------------------------------------------------------------

TEST(QualityMonitoringConfig, DefaultsAreDisabledAndValid) {
  QualityMonitoringConfig cfg;
  // Defaults: master switch off, so no metric/visualization work runs.
  EXPECT_FALSE(cfg.should_compute_any_metrics());
  EXPECT_FALSE(cfg.should_visualize());
  // Expensive metrics default off.
  EXPECT_FALSE(cfg.any_expensive_metrics_enabled());
  // Default thresholds are all in range -> validate() passes.
  EXPECT_TRUE(cfg.validate());
}

TEST(QualityMonitoringConfig, ComputeMetricsRequiresBothSwitches) {
  QualityMonitoringConfig cfg;
  // Master enabled but compute_metrics still off -> no metrics.
  cfg.enable = true;
  EXPECT_FALSE(cfg.should_compute_any_metrics());
  // Both on -> metrics compute.
  cfg.compute_metrics.enable = true;
  EXPECT_TRUE(cfg.should_compute_any_metrics());
  // compute_metrics on but master off -> still no metrics.
  cfg.enable = false;
  EXPECT_FALSE(cfg.should_compute_any_metrics());
}

TEST(QualityMonitoringConfig, VisualizeRequiresBothSwitches) {
  QualityMonitoringConfig cfg;
  cfg.visualization.enable = true;
  EXPECT_FALSE(cfg.should_visualize());    // master off
  cfg.enable = true;
  EXPECT_TRUE(cfg.should_visualize());
}

TEST(QualityMonitoringConfig, ExpensiveMetricsFlag) {
  QualityMonitoringConfig cfg;
  cfg.metrics.temporal_stability = true;
  EXPECT_TRUE(cfg.any_expensive_metrics_enabled());
  cfg.metrics.temporal_stability = false;
  cfg.metrics.reprojection_error = true;
  EXPECT_TRUE(cfg.any_expensive_metrics_enabled());
}

TEST(QualityMonitoringConfig, ValidateRejectsBadValues) {
  // Non-positive log interval (with metrics enabled) is invalid.
  QualityMonitoringConfig cfg;
  cfg.enable = true;
  cfg.compute_metrics.enable = true;
  cfg.compute_metrics.log_interval = 0;
  EXPECT_FALSE(cfg.validate());

  // Thresholds outside [0, 1] are invalid.
  QualityMonitoringConfig density_bad;
  density_bad.thresholds.min_point_density = 1.5;
  EXPECT_FALSE(density_bad.validate());

  QualityMonitoringConfig noise_bad;
  noise_bad.thresholds.max_noise_ratio = -0.1;
  EXPECT_FALSE(noise_bad.validate());

  QualityMonitoringConfig coverage_bad;
  coverage_bad.thresholds.min_disparity_coverage = 2.0;
  EXPECT_FALSE(coverage_bad.validate());
}

// ---------------------------------------------------------------------------
// StereoProcessingStrategy: FPS math (no GPU / camera needed)
// ---------------------------------------------------------------------------

// Test shim that exposes the protected processing-time field so we can drive
// the real get_processing_stats() formula in the base class with known inputs.
class FpsProbeStrategy : public StereoProcessingStrategy
{
public:
  bool initialize(const StereoConfig &, const cv::Size &) override {return true;}
  cv::Mat compute_disparity(const cv::Mat &, const cv::Mat &) override {return cv::Mat();}
  pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(
    const cv::Mat &, const cv::Mat &) override
  {
    return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  }
  void update_config(const StereoConfig &) override {}
  std::string get_strategy_name() const override {return "Probe";}

  void set_time(double ms) {last_processing_time_ = ms;}
};

TEST(StereoProcessingStats, ZeroTimeGivesZeroFps) {
  FpsProbeStrategy s;    // last_processing_time_ defaults to 0.0
  double avg = -1.0, fps = -1.0;
  s.get_processing_stats(avg, fps);
  EXPECT_DOUBLE_EQ(avg, 0.0);
  EXPECT_DOUBLE_EQ(fps, 0.0);
}

TEST(StereoProcessingStats, FpsIsReciprocalOfMilliseconds) {
  FpsProbeStrategy s;
  s.set_time(20.0);    // 20 ms per frame -> 50 fps
  double avg = 0.0, fps = 0.0;
  s.get_processing_stats(avg, fps);
  EXPECT_DOUBLE_EQ(avg, 20.0);
  EXPECT_DOUBLE_EQ(fps, 50.0);

  s.set_time(40.0);    // 40 ms per frame -> 25 fps
  s.get_processing_stats(avg, fps);
  EXPECT_DOUBLE_EQ(fps, 25.0);
}

// ---------------------------------------------------------------------------
// StereoProcessingFactory + strategy naming
// ---------------------------------------------------------------------------

TEST(StereoProcessingFactory, CreatesNonNullForEveryStrategyType) {
  using F = StereoProcessingFactory;
  EXPECT_NE(F::create_strategy(F::StrategyType::GPU_BM), nullptr);
  EXPECT_NE(F::create_strategy(F::StrategyType::CPU_BM), nullptr);
  EXPECT_NE(F::create_strategy(F::StrategyType::GPU_SGBM), nullptr);
  EXPECT_NE(F::create_strategy(F::StrategyType::CPU_SGBM), nullptr);
}

TEST(StereoStrategyNaming, SgbmNameReflectsGpuFlag) {
  SGBMStereoStrategy cpu(false);
  SGBMStereoStrategy gpu(true);
  EXPECT_EQ(cpu.get_strategy_name(), "CPU Semi-Global Block Matching");
  EXPECT_EQ(gpu.get_strategy_name(), "GPU Semi-Global Block Matching");
}

TEST(StereoStrategyNaming, ConcreteSubclassNames) {
  CPUBlockMatchingStereo bm;
  GPUSGBMStereo gpu_sgbm;
  CPUSGBMStereo cpu_sgbm;
  EXPECT_EQ(bm.get_strategy_name(), "CPU Block Matching Stereo");
  EXPECT_EQ(gpu_sgbm.get_strategy_name(), "GPU SGBM Stereo");
  EXPECT_EQ(cpu_sgbm.get_strategy_name(), "CPU SGBM Stereo");
}

TEST(StereoConfigDefaults, MatchHardwareExpectations) {
  StereoConfig cfg;
  // Guard the documented defaults so an accidental edit is caught.
  EXPECT_EQ(cfg.num_disparities, 64);
  EXPECT_EQ(cfg.block_size, 15);
  EXPECT_EQ(cfg.min_disparity, 0);
  EXPECT_TRUE(cfg.use_gpu);
}
