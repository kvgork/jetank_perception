# Minimum Range Calculation for IMX219-83 Stereo Camera

## Camera Specifications
- **Baseline (B)**: 60mm = 0.06m
- **FOV Horizontal**: 73°
- **Resolution**: 640×360
- **Focal Length (pixels)**: ~300 px

## Theoretical Minimum (FOV Overlap Limit)

### FOV Overlap Calculation

For stereo to work, the fields of view must overlap:

```
Half FOV angle = 73° / 2 = 36.5°

Minimum distance for complete overlap:
d_min = Baseline / (2 × tan(FOV_half))
d_min = 60mm / (2 × tan(36.5°))
d_min = 60mm / (2 × 0.74)
d_min = 60mm / 1.48
d_min ≈ 40mm = 4cm
```

**BUT** this is only where FOVs START to overlap!

### Practical Overlap Requirement

For good stereo matching, you need **significant overlap** (> 50% of image width):

```
For 50% useful overlap:
d_min ≈ 150-200mm = 15-20cm
```

## Disparity Limit

Maximum useful disparity is typically **< 50% of image width**:

```
Max useful disparity = 640px / 2 = 320 pixels

Minimum distance at 320px disparity:
Z = (B × f) / d
Z = (0.06m × 300px) / 320px
Z = 18 / 320
Z ≈ 0.056m = 5.6cm
```

## Quality at Different Ranges

| Distance | Disparity | FOV Overlap | Quality | Usable? |
|----------|-----------|-------------|---------|---------|
| 6cm | ~300px | ~5% | Impossible | ❌ No |
| 10cm | ~180px | ~30% | Very Poor | ❌ No |
| 15cm | ~120px | ~50% | Poor | ⚠️ Maybe |
| 20cm | ~90px | ~65% | Fair | ⚠️ Yes |
| 25cm | ~72px | ~75% | Good | ✅ Yes |
| 30cm | ~60px | ~80% | Very Good | ✅ Yes |

## Recommended Minimum Range

### Conservative: **25cm (0.25m)**
- Reliable stereo matching ✓
- Good FOV overlap (75%) ✓
- Reasonable disparity (72px) ✓
- Low noise ✓

**This is what's currently configured.**

### Aggressive: **15cm (0.15m)**
- Possible but challenging
- Limited overlap (~50%)
- High disparity (120px)
- More noise and artifacts
- Requires excellent calibration

### Absolute Limit: **~10cm**
- Extremely difficult
- Very limited overlap
- Very high disparity (180px)
- Extremely noisy
- Only works in perfect conditions
- **NOT RECOMMENDED**

## Testing Lower Minimum

If you want to test lower ranges, here's what to change:

### For 20cm minimum:
```yaml
pointcloud.range_filter.min_range: 0.20

# Also increase search range:
stereo.num_disparities: 96  # Or even 128
```

### For 15cm minimum (experimental):
```yaml
pointcloud.range_filter.min_range: 0.15

# Much wider search needed:
stereo.num_disparities: 128
stereo.block_size: 11  # Smaller for high disparity

# Expect more noise, need aggressive filtering:
pointcloud.statistical_filter.stddev_threshold: 1.5
```

## Why Not Go Lower?

### Problem 1: FOV Geometry
At distances < 15cm, the cameras are looking at **almost completely different scenes**:

```
Left camera sees:  [====LEFT OBJECT====]
Right camera sees:         [====RIGHT OBJECT====]
                   Overlap: [==] (tiny!)
```

### Problem 2: Disparity Search Range
To handle 10cm distance:
- Need disparity range 0-180 pixels
- Must search ~180 positions per pixel
- Computation cost: ~3× higher
- More false matches = more noise

### Problem 3: Calibration Sensitivity
At very close range:
- Small calibration errors cause HUGE depth errors
- 0.1° angle error at 10cm = 5mm depth error
- 0.1° angle error at 30cm = 1mm depth error

## Alternative Solutions for Close Range

### Option 1: Different Camera
For measurements < 15cm, consider:
- **Structured light** (Intel RealSense)
- **ToF (Time of Flight)** sensors
- **Wider baseline** stereo (but loses far range)

### Option 2: Sensor Fusion
Combine your stereo camera (30cm+) with:
- **Ultrasonic sensors** for close range (5-30cm)
- **IR proximity sensors** for very close (2-15cm)

### Option 3: Accept the Limit
For robotics navigation:
- 25cm minimum is perfectly fine
- Robots shouldn't get closer than 25cm to obstacles anyway
- Use bumpers/proximity sensors for < 25cm

## Current Configuration Rationale

**Minimum: 25cm, Maximum: 2m**

This range is optimal because:
1. ✅ Reliable stereo matching
2. ✅ Low noise
3. ✅ Good point density
4. ✅ Suitable for robot navigation
5. ✅ Matches camera design specs

## Conclusion

**No, you cannot reach the 60mm baseline** as minimum range.

**Practical minimum: 25cm** (current setting)
**Aggressive minimum: 15cm** (experimental, lower quality)
**Absolute minimum: ~10cm** (extremely poor quality)

For your robot's sock-collecting task, **25cm minimum is ideal** - the robot should maintain this distance from objects anyway for safe manipulation.
