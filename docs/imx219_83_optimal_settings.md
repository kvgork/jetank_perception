# IMX219-83 Stereo Camera - Optimal Settings

## Camera Hardware Specifications

**Waveshare IMX219-83 Stereo Camera**
- **Sensor**: IMX219 (2× 8MP cameras)
- **Resolution**: 3280 × 2464 native (using 640×360 for performance)
- **Baseline**: 60mm (0.06m)
- **Focal Length**: 2.6mm physical
- **FOV**: 83° diagonal, 73° horizontal, 50° vertical
- **Aperture**: F2.4

## Depth Range Calculations

For 60mm baseline at 640×360 resolution with focal length ≈ 300 pixels:

### Distance vs Disparity Table

| Distance | Disparity | Notes |
|----------|-----------|-------|
| 0.3m | ~60 px | Closest reliable distance |
| 0.5m | ~36 px | Good quality |
| 1.0m | ~18 px | Good quality |
| 1.5m | ~12 px | Fair quality |
| 2.0m | ~9 px | Poor quality, noisy |
| 3.0m+ | <6 px | Very poor/unreliable |

### Recommended Working Range
**0.3m to 1.5m** for best results

## Optimal Configuration Settings

### Stereo Processing Parameters

```yaml
stereo.algorithm: "GPU_BM"              # Use GPU acceleration
stereo.use_gpu: true
stereo.num_disparities: 64              # Search range 0-64 pixels
stereo.block_size: 15                   # Match window size (odd number)
stereo.min_disparity: 0                 # Start search from 0
stereo.uniqueness_ratio: 15             # Match quality threshold
stereo.speckle_window_size: 150         # Noise removal
stereo.speckle_range: 16                # Speckle filtering strength
stereo.disp12_max_diff: 1               # Left-right consistency
stereo.pre_filter_cap: 63               # Texture normalization
stereo.pre_filter_size: 9               # Pre-filter window
stereo.texture_threshold: 5             # Minimum texture required
```

**Why these values:**
- `num_disparities: 64` covers 0.28m to infinity (but quality drops after 2m)
- `block_size: 15` balances detail vs noise at close range
- `uniqueness_ratio: 15` filters poor matches (higher = stricter)
- `texture_threshold: 5` allows matching on moderate texture scenes

### Point Cloud Filters

```yaml
# Range filter - hard limits on depth
pointcloud.range_filter.enable: true
pointcloud.range_filter.min_range: 0.25     # 25cm minimum
pointcloud.range_filter.max_range: 2.0      # 2m maximum

# Voxel filter - downsampling for performance
pointcloud.voxel_filter.enable: true
pointcloud.voxel_filter.leaf_size: 0.005    # 5mm voxels

# Statistical outlier removal
pointcloud.statistical_filter.enable: true
pointcloud.statistical_filter.k_neighbors: 30
pointcloud.statistical_filter.stddev_threshold: 2.0
```

**Why these values:**
- Min/max range filters out unreliable near/far points
- 5mm voxels preserve detail at close range
- Lenient outlier filter (2.0 std) keeps more valid points

## Tuning for Different Scenarios

### Scenario 1: Indoor Robot Navigation (0.3-1.2m)
**Current settings are optimal** ✓

### Scenario 2: Longer Range (1.0-2.5m)
```yaml
stereo.num_disparities: 48          # Reduce search range
stereo.block_size: 21               # Larger blocks for stability
pointcloud.range_filter.max_range: 3.0
```
Note: Quality will be poor beyond 2m with 60mm baseline.

### Scenario 3: Very Close Objects (0.2-0.8m)
```yaml
stereo.num_disparities: 96          # Increase for high disparity
stereo.block_size: 11               # Smaller for more detail
stereo.texture_threshold: 10        # Stricter matching
pointcloud.range_filter.min_range: 0.15
pointcloud.range_filter.max_range: 1.0
```

### Scenario 4: Low Texture Environment (white walls, etc.)
```yaml
stereo.block_size: 21               # Larger windows
stereo.texture_threshold: 3         # Very lenient
stereo.pre_filter_cap: 63           # Max texture enhancement
stereo.uniqueness_ratio: 10         # Less strict matching
```

## Performance vs Quality Trade-offs

### For Better Quality (Slower)
```yaml
stereo.block_size: 21               # Larger matching window
stereo.algorithm: "GPU_SGBM"        # Better algorithm (slower)
stereo.uniqueness_ratio: 20         # Very strict matching
pointcloud.statistical_filter.k_neighbors: 50
```

### For Better Performance (Faster, Noisier)
```yaml
stereo.block_size: 9                # Smaller window
stereo.uniqueness_ratio: 10         # Less strict
pointcloud.statistical_filter.enable: false
pointcloud.voxel_filter.leaf_size: 0.01  # Coarser voxels
```

## Algorithm Choices

### GPU_BM (Block Matching) - Current Choice
- **Speed**: Very fast on Jetson
- **Quality**: Good for textured scenes
- **Best for**: Real-time robotics, good lighting

### GPU_SGBM (Semi-Global Block Matching)
- **Speed**: ~2-3x slower than BM
- **Quality**: Better, especially on low-texture scenes
- **Best for**: When quality > speed, challenging scenes

To switch:
```yaml
stereo.algorithm: "GPU_SGBM"
```

## Calibration Requirements

For best results, you MUST calibrate your stereo camera:

1. **Capture 30+ image pairs** of a checkerboard pattern
2. **Use ROS2 camera_calibration** tool
3. **Verify rectification** - epipolar error should be < 1 pixel
4. **Update calibration files** in `config/calibration/`

Without proper calibration:
- Point clouds will be distorted
- Depth measurements will be inaccurate
- Noise will be much higher

## Expected Results with Optimal Settings

At **0.5m distance** in **good lighting** with **proper calibration**:

| Metric | Expected Value |
|--------|----------------|
| Disparity Valid % | 60-80% |
| Point Density | 70-90% |
| Outlier Ratio | < 5% |
| Depth Accuracy | ±1-2cm |
| Update Rate | 20-30 FPS |

## Troubleshooting by Symptom

### "Disparity mostly black, valid < 30%"
**Cause**: Stereo matching failure
**Fix**:
1. Increase lighting
2. Add texture to scene
3. Lower `texture_threshold: 3`
4. Increase `pre_filter_cap: 63`

### "Speckled/noisy disparity"
**Cause**: Poor match quality
**Fix**:
1. Increase `block_size: 21`
2. Increase `uniqueness_ratio: 20`
3. Increase `speckle_window_size: 200`

### "Point cloud has few points after filtering"
**Cause**: Filters too aggressive
**Fix**:
1. Increase `stddev_threshold: 3.0`
2. Decrease `k_neighbors: 20`
3. Disable filters one-by-one to identify culprit

### "Objects appear at wrong distance"
**Cause**: Bad calibration or baseline
**Fix**:
1. Recalibrate cameras
2. Verify `baseline: 0.06` in config
3. Check focal length in calibration files

### "Everything too far away (> 2m)"
**Cause**: Not using optimal range
**Fix**:
1. Move closer to objects (0.3-1.5m)
2. This camera is designed for close range
3. 60mm baseline cannot reliably measure > 2m

## Resolution Trade-offs

Current: **640×360**

### Higher Resolution (1280×720)
**Pros:**
- More detail in point cloud
- Better texture for matching
- Higher disparity values

**Cons:**
- 4× more pixels to process
- Slower frame rate (15-20 FPS)
- More memory usage

### Lower Resolution (320×180)
**Pros:**
- Much faster (40+ FPS)
- Less memory

**Cons:**
- Less detail
- Lower disparity precision
- Noisier results

**Recommendation**: Stick with 640×360 for balanced performance on Jetson Orin Nano.

## Hardware Optimization Tips

1. **Good Lighting**: 80% of quality comes from good input images
2. **Clean Lenses**: Dust/smudges severely impact matching
3. **Stable Mount**: Any vibration causes motion blur
4. **GPU Thermal**: Keep Jetson cool for consistent GPU performance
5. **Power Mode**: Use MAX-N mode on Jetson for best performance

## References

- Camera: https://www.waveshare.com/imx219-83-stereo-camera.htm
- Stereo Vision Theory: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- ROS2 Camera Calibration: https://navigation.ros.org/tutorials/docs/camera_calibration.html
