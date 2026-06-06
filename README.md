# jetank_perception

Stereo and single-camera perception nodes for the JeTank's CSI rig. Provides
rectification, disparity (GPU / SGBM strategy pattern) and point-cloud
generation on the Jetson Orin Nano.

## ROS 2 API

This package (`jetank_perception`, build type `ament_cmake`) provides C++ camera perception nodes for the JeTank's CSI stereo rig. It contains two executables and three launch files. There are no actions and no custom `msg/srv/action` definitions in this package; all interfaces use standard ROS message/service types.

### Nodes

| Node name | Executable | Role |
|---|---|---|
| `stereo_camera_node` | `stereo_camera_node` | Dual-CSI stereo capture, image rectification, disparity + point-cloud generation (GPU/CPU BM/SGBM strategies), optional JPEG/PNG compression, optional quality monitoring. |
| `camera_node` | `camera_node` | Single CSI camera capture; publishes a rate-limited BGR8 image stream. |

> **Topic namespacing:** `stereo_camera.launch.py` launches `stereo_camera_node` under the namespace `stereo_camera` (default), so the topics/services below appear as `/stereo_camera/<name>` at runtime. `single_camera.launch.py` / `simple_camera.launch.py` launch `camera_node` with an empty namespace, so `camera/image_raw` resolves to `/camera/image_raw`. No launch file applies topic remappings (the stereo launch file has only commented-out examples).

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
