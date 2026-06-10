# jetank_perception

Stereo and single-camera perception nodes for the JeTank's CSI rig. Provides
rectification, disparity (GPU / SGBM strategy pattern) and point-cloud
generation on the Jetson Orin Nano.

## ROS 2 API

This package (`jetank_perception`, build type `ament_cmake`) provides C++ camera perception nodes for the JeTank's CSI stereo rig. It contains three executables and three launch files. It defines no custom `msg/srv/action` types of its own; the `sock_segmentation_server` implements the `SegmentSocks` action defined in `jetank_detection`. All other interfaces use standard ROS message/service types.

### Nodes

| Node name | Executable | Role |
|---|---|---|
| `stereo_camera_node` | `stereo_camera_node` | Dual-CSI stereo capture (or ROS-topic input in sim), image rectification, disparity + point-cloud generation (GPU/CPU BM/SGBM strategies), optional JPEG/PNG compression, optional quality monitoring. |
| `camera_node` | `camera_node` | Single CSI camera capture; publishes a rate-limited BGR8 image stream. |
| `sock_segmentation_server` | `sock_segmentation_server` | `SegmentSocks` action server: reprojects detector bbox ROIs from the cached disparity image into a 3D sock blob in a requested frame. |

> **Topic namespacing:** `stereo_camera.launch.py` launches `stereo_camera_node` under the namespace `stereo_camera` (default), so the topics/services below appear as `/stereo_camera/<name>` at runtime. `single_camera.launch.py` / `simple_camera.launch.py` launch `camera_node` with an empty namespace, so `camera/image_raw` resolves to `/camera/image_raw`. No launch file applies topic remappings (the stereo launch file has only commented-out examples).

> **Input source:** `stereo_camera_node` selects its capture path via the `input_source` parameter. `csi` (default) opens the dual CSI cameras directly. `ros_topics` (simulation) instead time-syncs an incoming stereo image stream — e.g. Gazebo's `left/image_raw` + `right/image_raw` — and feeds those frames through the same rectify/disparity/point-cloud pipeline. The subscribed topics are configurable via `ros_input.left_image_topic` / `right_image_topic` / `left_info_topic` / `right_info_topic`.

### Published topics

#### `stereo_camera_node`

| Topic (relative) | Type | Notes |
|---|---|---|
| `left/image_raw` | `sensor_msgs/msg/Image` | Published when `publishing.publish_raw_images` is true. |
| `right/image_raw` | `sensor_msgs/msg/Image` | Published when `publishing.publish_raw_images` is true. |
| `left/image_rect` | `sensor_msgs/msg/Image` | Rectified; only when calibrated and `publish_rectified_images` true. |
| `right/image_rect` | `sensor_msgs/msg/Image` | Rectified; only when calibrated and `publish_rectified_images` true. |
| `left/camera_info` | `sensor_msgs/msg/CameraInfo` | |
| `right/camera_info` | `sensor_msgs/msg/CameraInfo` | |
| `left/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Only when raw-image compression is enabled. |
| `right/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Only when raw-image compression is enabled. |
| `left/image_rect/compressed` | `sensor_msgs/msg/CompressedImage` | Only when rectified-image compression is enabled. |
| `right/image_rect/compressed` | `sensor_msgs/msg/CompressedImage` | Only when rectified-image compression is enabled. |
| `disparity` | `stereo_msgs/msg/DisparityImage` | Only when calibrated and `publish_disparity` true (queue depth 1). |
| `points` | `sensor_msgs/msg/PointCloud2` | Only when calibrated and `publish_pointcloud` true (queue depth 1). |
| `diagnostics/disparity_colored` | `sensor_msgs/msg/Image` | Only when quality-monitoring visualization is enabled. |
| `diagnostics/depth_uncertainty` | `sensor_msgs/msg/Image` | Only when quality-monitoring visualization is enabled. |

#### `camera_node`

| Topic (relative) | Type |
|---|---|
| `camera/image_raw` | `sensor_msgs/msg/Image` |

### Subscribed topics

Neither node subscribes to any topics.

### Services (`stereo_camera_node`)

| Service (relative) | Type | Role |
|---|---|---|
| `left/set_camera_info` | `sensor_msgs/srv/SetCameraInfo` | Set and persist left camera intrinsics. |
| `right/set_camera_info` | `sensor_msgs/srv/SetCameraInfo` | Set and persist right camera intrinsics. |
| `calibrate_stereo` | `std_srvs/srv/Trigger` | Compute stereo extrinsics from both individually-calibrated cameras. |

### Key parameters

#### `stereo_camera_node` (see `config/stereo_camera_config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `camera.width` / `camera.height` | `640` / `480` | Capture resolution. |
| `camera.fps` | `20` | Capture frame rate. |
| `camera.format` | `GRAY8` | Capture pixel format. |
| `camera.left_sensor_id` / `camera.right_sensor_id` | `0` / `1` | CSI sensor IDs. |
| `camera.flip_images_180` | `false` | Rotate both frames 180 degrees. |
| `stereo.algorithm` | `GPU_BM` | One of `GPU_BM`, `CPU_BM`, `GPU_SGBM`, `CPU_SGBM`. |
| `stereo.num_disparities` | `64` | Disparity search range. |
| `stereo.block_size` | `15` | Matching block size. |
| `calibration.left_camera_info_url` / `right_camera_info_url` / `stereo_calibration_url` | `""` | Calibration file URLs (`file://...`); the stereo launch file resolves these to files under `config/calibration/`. |
| `calibration.default.baseline` | `0.06` | Fallback baseline (m) used for the disparity message. |
| `frames.left_frame_id` / `right_frame_id` / `base_frame_id` | `camera_left_link` / `camera_right_link` / `base_link` | TF frame IDs. |
| `publishing.publish_raw_images` / `publish_rectified_images` / `publish_disparity` / `publish_pointcloud` | `true` | Output enable flags. |
| `publishing.raw_images.compression.enabled` (+ `.mode`, `.format`, `.jpeg_quality`, `.png_level`) | `false` | Raw-image compression config (analogous `rectified_images.*` set exists). |
| `quality_monitoring.enable` | `false` | Master switch for the quality-monitoring/diagnostics pipeline (many `quality_monitoring.*` sub-parameters). |

(Many additional `pointcloud.*`, `performance.*`, `logging.*`, and `development.*` parameters are declared — see `initialize_parameters()` in `src/stereo_camera_node.cpp`.)

#### `camera_node`

| Parameter | Default | Description |
|---|---|---|
| `camera_width` / `camera_height` | `640` / `480` | Capture resolution. |
| `camera_fps` | `30` | Capture frame rate. |
| `camera_format` | `NV12` | Capture pixel format. |
| `sensor_id` | `0` | CSI sensor ID. |
| `use_hardware_acceleration` | `true` | Enable Jetson HW-accelerated pipeline. |
| `publish_rate_hz` | `30.0` | Image publish rate (rate-limited). |

### Launch files

| Launch file | Brings up |
|---|---|
| `stereo_camera.launch.py` | `stereo_camera_node` (namespace `stereo_camera`) + optional static TF publishers for `camera_left_link` / `camera_right_link`. |
| `single_camera.launch.py` | `camera_node` (configurable name/namespace) + optional RViz2 and image_view. |
| `simple_camera.launch.py` | `camera_node` with fixed defaults for quick testing. |

## Sock segmentation (`sock_segmentation_server`)

`sock_segmentation_server` turns the 2D sock detector's bounding boxes into a 3D
point-cloud blob. It serves the `SegmentSocks` action (defined in
`jetank_detection`) on `/segment_socks` and consumes the latest cached
`DisparityImage` + `CameraInfo` + `Detection2DArray` (no custom interfaces of
its own). Returns the single sock whose centroid is nearest the robot.

### Interfaces

| Kind | Name | Type | Notes |
|---|---|---|---|
| Action | `segment_socks` | `jetank_detection/action/SegmentSocks` | Goal: `target_frame` (empty → `base_link`), `min_score`, `max_range` (z clip, m), `publish_debug`. Result: `found`, `sock` (`SockCloud`: cloud, centroid, AABB `dimensions`, `label`, `score`, all in `target_frame`). Feedback: `processed` / `total`. |
| Sub | `/stereo_camera/disparity` | `stereo_msgs/msg/DisparityImage` | Latest-message cache; RELIABLE `KeepLast(5)`. |
| Sub | `/stereo_camera/left/camera_info` | `sensor_msgs/msg/CameraInfo` | Pinhole intrinsics for reprojection. |
| Sub | `/detections/socks` | `vision_msgs/msg/Detection2DArray` | Detector bboxes + scores. |
| Pub | `/socks/points` | `sensor_msgs/msg/PointCloud2` | Chosen blob, latched (transient-local); published only when `publish_debug`. |

### Pipeline (per goal)

1. **Snapshot** the cached disparity / camera_info / detections. Reject (→ `found=false`, succeed) if any is missing, stale (older than `max_age`), or out of sync (disparity vs detections stamp differ by more than `max_sync_dt`).
2. **Reproject ROI** — for each detection scoring ≥ `min_score`, reproject the clamped bbox pixels into 3D via the pinhole model (`Z=f·t/d`, `X=(u-cx)Z/f`, `Y=(v-cy)Z/f`) in the disparity optical frame, dropping NaN/zero disparities and points beyond `max_range`.
3. **Ground removal** (when `remove_ground`) via `ground_filter` mode (below).
4. **Cluster** — Euclidean clustering at `cluster_tolerance`; keep the largest cluster as the sock blob. Stages dropping below `min_points` discard the detection.
5. **Transform** the chosen blob's cloud + centroid into `target_frame` via tf2 (with stamp→latest fallback), recomputing the AABB `dimensions` in that frame.
6. **Select** the blob whose centroid is nearest the `base_frame` origin and fill the result; optionally publish it on `/socks/points`. Cancellation is honoured between detections.

### Ground removal

Both modes RANSAC-fit the dominant flat floor plane (`ground_distance_threshold`) in the optical frame, then differ in what they keep:

- **`height`** (default) — keep points lying more than `ground_margin` (default 0.012 m) **above** the plane on the camera side. The sock sits proud of the floor, so it survives even when sock and floor are near-coplanar.
- **`ransac`** — legacy binary inlier removal: discard every point within `ground_distance_threshold` of the plane. This can delete a whole low-lying sock that falls inside the plane band (observed 2507 → 0 points where the height gate gives 2507 → 176).

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `ground_filter` | `height` | Ground-removal mode: `height` (gate) or `ransac` (inlier removal). |
| `ground_margin` | `0.012` | (height mode) keep points more than this far above the floor plane (m). |
| `ground_distance_threshold` | `0.02` | RANSAC plane inlier distance (m); also the `ransac`-mode removal band. |
| `remove_ground` | `true` | Master switch for ground removal. |
| `min_points` | `30` | Minimum points required at the reproject / ground / cluster stages. |
| `cluster_tolerance` | `0.05` | Euclidean cluster distance (m). |
| `max_sync_dt` | `0.5` | Max disparity↔detections stamp skew (s). |
| `max_age` | `1.0` | Max input age before a goal is rejected (s). |
| `default_target_frame` / `base_frame` | `base_link` | Output frame when goal leaves it empty / frame for nearest-sock selection. |

`max_range` (z clip) is supplied per goal, not as a parameter. The pure reprojection math lives in `include/jetank_perception/sock_reproject.hpp` (`reproject_roi`) and is unit-tested separately (`test/test_reproject.cpp`).

## Tests

GTest unit tests for the pure stereo/quality math in the headers — no camera, GPU, or ROS graph required.

| Test file | Imports | Asserts |
|---|---|---|
| `test/test_stereo_math.cpp` | `quality_monitoring.hpp`, `stereo_processing_strategy.hpp` | `QualityMonitoringConfig` gating (`should_compute_any_metrics`/`should_visualize` need both master + sub switch) and `validate()` rejecting out-of-range thresholds and a non-positive log interval. `StereoProcessingStrategy::get_processing_stats()` FPS math (0 ms → 0 fps, 20 ms → 50 fps). `StereoProcessingFactory` returns non-null for every strategy type, and the strategy classes report their expected `get_strategy_name()`. `StereoConfig` defaults match the documented hardware values (`num_disparities=64`, `block_size=15`, `min_disparity=0`, `use_gpu=true`). |

Build and run via colcon:

```bash
colcon test --packages-select jetank_perception
colcon test-result --verbose
```
