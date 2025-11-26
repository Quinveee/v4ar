#!/bin/bash

echo "=========================================="
echo "OAK-D Topic Information"
echo "=========================================="

echo ""
echo "1. /stereo/depth"
echo "   Type:"
ros2 topic info /stereo/depth
echo "   Sample (first message):"
timeout 2s ros2 topic echo /stereo/depth --once | head -20

echo ""
echo "=========================================="
echo "2. /stereo/converted_depth"
echo "   Type:"
ros2 topic info /stereo/converted_depth

echo ""
echo "=========================================="
echo "3. /stereo/camera_info"
echo "   Type:"
ros2 topic info /stereo/camera_info
echo "   Content:"
ros2 topic echo /stereo/camera_info --once

echo ""
echo "=========================================="
echo "4. /stereo/points"
echo "   Type:"
ros2 topic info /stereo/points
echo "   Sample:"
timeout 2s ros2 topic echo /stereo/points --once | head -30

echo ""
echo "=========================================="
echo "Topic Rates:"
echo "=========================================="
for topic in /stereo/depth /stereo/converted_depth /stereo/camera_info /stereo/points; do
    echo -n "$topic: "
    timeout 3s ros2 topic hz $topic 2>&1 | grep "average rate" || echo "N/A"
done

echo ""
echo "=========================================="