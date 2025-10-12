#!/bin/bash
# Quick quality check script

echo "=== Stereo Camera Quality Check ==="
echo ""
echo "Checking if node is running..."
ros2 node list | grep stereo_camera_node

if [ $? -ne 0 ]; then
    echo "ERROR: Node not running!"
    echo "Start with: ros2 launch jetank_perception stereo_camera.launch.py"
    exit 1
fi

echo ""
echo "Available topics:"
ros2 topic list | grep -E "(stereo|diagnostics|disparity|points)"

echo ""
echo "=== Checking disparity quality (one sample) ==="
timeout 3 ros2 topic echo /stereo_camera/disparity --once 2>&1 | grep -A 5 "min_disparity\|max_disparity\|delta_d" || echo "No disparity data"

echo ""
echo "=== Diagnostic topic info ==="
ros2 topic info /stereo_camera/diagnostics/disparity_colored 2>&1 || echo "Disparity colored topic not available"
ros2 topic info /stereo_camera/diagnostics/depth_uncertainty 2>&1 || echo "Depth uncertainty topic not available"

echo ""
echo "=== Quick point cloud check ==="
echo "Sampling point cloud..."
timeout 3 ros2 topic echo /stereo_camera/points --once 2>&1 | head -30 | grep -E "width:|height:" || echo "No point cloud data"

echo ""
echo "Done! If diagnostics topics are missing, quality monitoring may not be enabled."
