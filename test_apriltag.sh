#!/bin/bash

# Test AprilTag Detection with ROS Bag
cd /v4ar
source install/setup.bash

echo "Starting AprilTag Visualization Test..."
echo "1. Open another terminal and run: ros2 bag play src/apriltag_localization/apriltag_localization_cleaned/"
echo "2. Press 's' to save screenshots, 'q' to quit"
echo ""

# Run the AprilTag visualization node
# The bag data publishes to /image_raw, so we need to remap if needed
python3 src/perception/marker_detector/pupil_apriltag_vis_node.py --tag_family tagStandard41h12 --output_dir apriltag_test_results

echo "AprilTag test completed. Check apriltag_test_results/ for screenshots."