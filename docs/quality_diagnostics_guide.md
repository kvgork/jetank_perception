# Point Cloud Quality Diagnostics Guide

## Overview

This guide helps you diagnose point cloud noise issues by analyzing quality metrics and visualizations at each stage of the stereo vision pipeline.

---

## Pipeline Stages & What to Check

### Stage 1: Raw Image Quality

**Console Logs:**
```
Left Camera - Blur: 450.23, Brightness: 128.5±45.2, Contrast: 198.4
Right Camera - Blur: 445.67, Brightness: 130.2±44.8, Contrast: 195.3
```

**What to Look For:**

| Metric | Good Values | Problem Indicators | Likely Cause |
|--------|-------------|-------------------|--------------|
| **Blur** | > 300 | < 100 | Camera out of focus, motion blur, dirty lens |
| **Brightness** | 80-180 | < 50 or > 220 | Poor lighting, exposure issues |
| **Contrast** | > 150 | < 80 | Low scene texture, overexposed, foggy lens |

**Action if Problems Found:**
- Low blur → Clean lenses, check focus, reduce motion
- Poor brightness → Adjust lighting, camera exposure settings
- Low contrast → Improve scene lighting, adjust camera settings

---

### Stage 2: Rectification Quality

**Console Logs:**
```
Rectification - Mean epipolar error: 0.45 px, Max: 0.98 px, Quality: GOOD
```

**What to Look For:**

| Metric | Good Values | Problem Indicators | Likely Cause |
|--------|-------------|-------------------|--------------|
| **Mean Error** | < 1.0 px | > 2.0 px | Poor calibration |
| **Max Error** | < 2.0 px | > 5.0 px | Calibration severely wrong |
| **Quality** | GOOD | POOR | Need to recalibrate cameras |

**Visualizations to Check:**
- Rectified images should have horizontal epipolar lines
- Corresponding features should be on the same row

**Action if Problems Found:**
- Recalibrate stereo camera using calibration tools
- Check if cameras moved after calibration
- Verify calibration files are loaded correctly

---

### Stage 3: Disparity Quality

**Console Logs:**
```
Disparity - Valid: 45.2% (104832/230400), Range: [2.0, 63.5], Mean: 18.3±12.4
WARNING: Low disparity coverage (45.2% < 50.0% threshold)
```

**What to Look For:**

| Metric | Good Values | Problem Indicators | Root Cause |
|--------|-------------|-------------------|------------|
| **Valid Ratio** | > 60% | < 30% | Poor stereo matching |
| **Mean Disparity** | 10-50 | < 5 or > 60 | Wrong distance range or parameters |
| **Std Deviation** | 5-20 | > 30 | Very noisy matching |

**Visualizations:**

**Disparity Colored Image** (`/diagnostics/disparity_colored`):
- **Look for:**
  - Large black areas = No stereo match found
  - Smooth color gradients = Good depth estimation
  - Speckled/noisy colors = Poor matching

**Common Patterns:**

| What You See | Problem | Solution |
|--------------|---------|----------|
| Mostly black image | Almost no matches | Check: lighting, texture, calibration |
| Horizontal stripes | Incorrect rectification | Recalibrate cameras |
| Random speckles | Noise/low texture | Increase `block_size`, reduce `num_disparities` |
| Smooth but wrong | Wrong distance range | Adjust `num_disparities`, `min_disparity` |

---

### Stage 4 & 5: Point Cloud Quality (Before/After Filtering)

**Console Logs:**
```
BEFORE Filtering Point Cloud - Total: 104832, Finite: 98234 (93.7%), Outliers: 8234 (8.4%)
  Spatial - Z: [0.245, 8.923], Mean: (0.023, -0.015, 1.234), Std: (0.456, 0.389, 0.823)

AFTER Filtering Point Cloud - Total: 89456, Finite: 89456 (100%), Outliers: 1234 (1.4%)
  Spatial - Z: [0.350, 4.500], Mean: (0.018, -0.012, 1.156), Std: (0.234, 0.198, 0.456)
WARNING: High noise ratio (15.2% > 20.0% threshold)
```

**What to Look For:**

| Metric | Good Values | Problem Indicators | Meaning |
|--------|-------------|-------------------|---------|
| **Density (Finite %)** | > 90% | < 60% | Many invalid depth values |
| **Outlier Ratio** | < 5% | > 20% | Lots of noise points |
| **Z Range** | Scene-specific | Very wide (0.1-100m) | Invalid depth estimates |
| **Std Deviation** | Low relative to range | Very high | Scattered/noisy cloud |

**Depth Uncertainty Map** (`/diagnostics/depth_uncertainty`):
- **Cooler colors (blue/green)** = Reliable depth
- **Hotter colors (red/yellow)** = Uncertain depth
- Most uncertainty at far distances is normal

**Action if Problems Found:**

| Problem | Likely Stage | Fix |
|---------|-------------|-----|
| Low density before filtering | Disparity (Stage 3) | Fix stereo matching parameters |
| High outliers before filtering | Disparity or reprojection | Check Q matrix, disparity quality |
| Filters removing too much | Filtering too aggressive | Relax filter parameters |
| Points all over the place | Multiple stages | Start from Stage 1 |

---

## Diagnostic Workflow

### Step 1: Check Console Logs

Start the node and look for the quality monitoring output every 10 frames:

```bash
ros2 launch jetank_perception stereo_camera.launch.py
```

Watch for WARNING messages - they indicate threshold violations.

### Step 2: Analyze Each Stage

Go through stages 1-5 in order:

1. **Stage 1 bad?** → Fix camera/lighting issues first
2. **Stage 2 bad?** → Recalibrate cameras
3. **Stage 3 bad?** → Tune stereo matching parameters
4. **Stage 4/5 bad but 1-3 good?** → Adjust filtering

### Step 3: Visual Inspection

```bash
# Terminal 1: Launch node
ros2 launch jetank_perception stereo_camera.launch.py

# Terminal 2: View diagnostics
rviz2
# Add Image displays for:
#  - /stereo_camera/diagnostics/disparity_colored
#  - /stereo_camera/diagnostics/depth_uncertainty
#  - /stereo_camera/left/image_rect
#  - /stereo_camera/right/image_rect
# Add PointCloud2 for:
#  - /stereo_camera/points
```

### Step 4: Parameter Tuning

Based on findings, edit `config/stereo_camera_config.yaml`:

**For Low Disparity Coverage:**
```yaml
stereo.num_disparities: 128  # Increase from 64
stereo.block_size: 21        # Increase from 15 (smoother but less detail)
```

**For Noisy Disparity:**
```yaml
stereo.uniqueness_ratio: 15   # Increase from 10 (stricter matching)
stereo.speckle_window_size: 200  # Increase from 100
```

**For Over-Filtering:**
```yaml
pointcloud.statistical_filter.stddev_threshold: 2.0  # Increase from 1.0
pointcloud.voxel_filter.leaf_size: 0.02  # Increase from 0.01
```

---

## Common Issues & Solutions

### Issue 1: "Mostly Black Disparity Image"

**Symptoms:**
- Valid ratio < 20%
- Disparity image is mostly black
- Very few points in cloud

**Diagnosis:**
- Stage 3 problem (stereo matching failure)

**Solutions:**
1. Check Stage 1: Are images clear and well-lit?
2. Check Stage 2: Is rectification working? (Quality: GOOD?)
3. Increase texture: Add lighting, point at textured surfaces
4. Try different stereo algorithm: Switch from GPU_BM to GPU_SGBM in config
5. Adjust parameters:
   ```yaml
   stereo.pre_filter_cap: 63        # From 31
   stereo.texture_threshold: 5      # From 10
   ```

### Issue 2: "Speckled/Noisy Disparity"

**Symptoms:**
- Valid ratio ~50-70% but very noisy
- Colored disparity has random speckles
- High outlier ratio in point cloud

**Diagnosis:**
- Stage 3 problem (poor matching quality)

**Solutions:**
1. Increase block size: `stereo.block_size: 21` or `25`
2. Increase uniqueness: `stereo.uniqueness_ratio: 15`
3. Improve speckle filtering:
   ```yaml
   stereo.speckle_window_size: 200
   stereo.speckle_range: 16
   ```

### Issue 3: "Good Disparity but Noisy Point Cloud"

**Symptoms:**
- Valid ratio > 60% in disparity
- But high outliers in point cloud (> 20%)
- Points scattered unrealistically

**Diagnosis:**
- Stage 4 problem (reprojection or Q matrix issue)

**Solutions:**
1. Verify calibration loaded: Check startup logs
2. Check Q matrix values are reasonable
3. Might need to recalibrate with better samples
4. Adjust point cloud filters:
   ```yaml
   pointcloud.range_filter.min_range: 0.3  # Increase minimum
   pointcloud.range_filter.max_range: 5.0  # Decrease maximum
   ```

### Issue 4: "Filters Removing Everything"

**Symptoms:**
- Good metrics before filtering
- Very few points after filtering
- High outlier detection rate

**Diagnosis:**
- Stage 5 problem (overly aggressive filtering)

**Solutions:**
1. Relax statistical filter:
   ```yaml
   pointcloud.statistical_filter.stddev_threshold: 2.5  # Increase
   pointcloud.statistical_filter.k_neighbors: 30        # Decrease
   ```
2. Disable filters one by one to find culprit:
   ```yaml
   pointcloud.voxel_filter.enable: false
   pointcloud.statistical_filter.enable: false
   pointcloud.range_filter.enable: false
   ```
3. Check if your scene actually has that much noise (view before filtering)

---

## Parameter Reference

### Stereo Matching Parameters

| Parameter | Effect on Noise | Recommended Range |
|-----------|----------------|-------------------|
| `num_disparities` | More = wider depth range but slower | 64, 128, 192 (must be multiple of 16) |
| `block_size` | Larger = smoother but less detail | 5-25 (odd numbers only) |
| `uniqueness_ratio` | Higher = stricter, fewer but better matches | 5-20 |
| `speckle_window_size` | Larger = removes more noise | 50-200 |
| `speckle_range` | Larger = more aggressive despeckling | 4-32 |
| `texture_threshold` | Higher = requires more texture | 5-30 |

### Point Cloud Filter Parameters

| Parameter | Effect | Recommended Values |
|-----------|--------|-------------------|
| `voxel_filter.leaf_size` | Larger = more downsampling | 0.005-0.05 m |
| `statistical_filter.k_neighbors` | More neighbors = slower but better | 20-100 |
| `statistical_filter.stddev_threshold` | Higher = keeps more points | 1.0-3.0 |
| `range_filter.min_range` | Filter close noise | 0.1-0.5 m |
| `range_filter.max_range` | Filter far noise | 3.0-10.0 m |

---

## Tips for Best Results

1. **Start with good images**: 80% of point cloud quality comes from good input images
2. **Calibrate carefully**: Poor calibration = noisy point clouds no matter what
3. **Tune incrementally**: Change one parameter at a time
4. **Use quality monitoring**: The logs tell you exactly where things go wrong
5. **Scene matters**: Textureless surfaces (white walls) will always be noisy
6. **Test in good conditions first**: Good lighting, textured scene, then tackle harder cases

---

## Quick Diagnostic Checklist

- [ ] Images sharp and clear (blur > 300)?
- [ ] Images well-lit (brightness 80-180)?
- [ ] Sufficient contrast (> 150)?
- [ ] Rectification quality GOOD (error < 1.0 px)?
- [ ] Disparity coverage > 50%?
- [ ] Point density before filtering > 80%?
- [ ] Outlier ratio after filtering < 10%?
- [ ] Visual inspection: Disparity looks smooth?
- [ ] Visual inspection: Point cloud represents scene?

If all checks pass, your setup is good! If not, use this guide to fix the failing stage.
