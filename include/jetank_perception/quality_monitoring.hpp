#pragma once

#include <string>

namespace jetson_stereo_camera {

struct ComputeMetricsConfig {
    bool enable = false;
    int log_interval = 100;
    bool publish_to_topic = false;
};

struct QualityMetricsConfig {
    bool density = true;
    bool completeness = true;
    bool noise_level = true;
    bool temporal_stability = false;
    bool reprojection_error = false;
};

struct VisualizationConfig {
    bool enable = false;
    bool publish_disparity_colored = false;
    bool publish_depth_uncertainty = false;
    bool publish_density_map = false;
    bool publish_epipolar_overlay = false;
};

struct CalibrationValidationConfig {
    bool enable = false;
    bool validate_on_startup = true;
    bool log_rectification_quality = false;
};

struct ThresholdsConfig {
    double min_point_density = 0.3;
    double max_noise_ratio = 0.2;
    double min_disparity_coverage = 0.5;
};

struct QualityMonitoringConfig {
    bool enable = false;

    ComputeMetricsConfig compute_metrics;
    QualityMetricsConfig metrics;
    VisualizationConfig visualization;
    CalibrationValidationConfig calibration_validation;
    ThresholdsConfig thresholds;

    // Convenience methods for common checks
    bool should_compute_any_metrics() const {
        return enable && compute_metrics.enable;
    }

    bool should_visualize() const {
        return enable && visualization.enable;
    }

    bool any_expensive_metrics_enabled() const {
        return metrics.temporal_stability || metrics.reprojection_error;
    }

    // Validation method
    bool validate() const {
        if (enable && compute_metrics.enable && compute_metrics.log_interval <= 0) {
            return false;
        }
        if (thresholds.min_point_density < 0.0 || thresholds.min_point_density > 1.0) {
            return false;
        }
        if (thresholds.max_noise_ratio < 0.0 || thresholds.max_noise_ratio > 1.0) {
            return false;
        }
        if (thresholds.min_disparity_coverage < 0.0 || thresholds.min_disparity_coverage > 1.0) {
            return false;
        }
        return true;
    }
};

} // namespace jetson_stereo_camera