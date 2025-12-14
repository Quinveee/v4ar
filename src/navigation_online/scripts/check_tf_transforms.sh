#!/bin/bash
# Script to check TF transforms for Nav2 planner_server

echo "=== Checking TF Transforms for Nav2 Planner ==="
echo ""

# 0. Check current ROS time vs system time
echo "0. Checking time synchronization:"
echo "   Current ROS time:"
ros2 topic echo /clock --once 2>&1 | grep -E "sec:|nanosec:" | head -2 || echo "   (No /clock topic - using system time)"
echo "   System time: $(date +%s)"
echo ""

# 1. Check if map -> base_link transform exists
echo "1. Checking map -> base_link transform:"
TF_OUTPUT=$(timeout 2 ros2 run tf2_ros tf2_echo map base_link 2>&1)
if echo "$TF_OUTPUT" | grep -q "Translation\|At time"; then
    echo "   ✓ map -> base_link transform EXISTS"
    echo "$TF_OUTPUT" | head -10
    # Check timestamp
    if echo "$TF_OUTPUT" | grep -q "At time"; then
        TF_TIME=$(echo "$TF_OUTPUT" | grep "At time" | head -1 | grep -oE "[0-9]+\.[0-9]+")
        echo "   Transform timestamp: $TF_TIME"
    fi
else
    echo "   ✗ map -> base_link transform MISSING or STALE"
    echo "$TF_OUTPUT" | grep -E "ERROR|Invalid|frame does not exist" | head -3
    echo "   → This is what planner_server costmap is waiting for!"
fi
echo ""

# 2. Check if map -> base_footprint transform exists
echo "2. Checking map -> base_footprint transform:"
if timeout 2 ros2 run tf2_ros tf2_echo map base_footprint 2>&1 | grep -q "Translation\|At time"; then
    echo "   ✓ map -> base_footprint transform EXISTS"
    timeout 2 ros2 run tf2_ros tf2_echo map base_footprint 2>&1 | head -10
else
    echo "   ✗ map -> base_footprint transform MISSING"
fi
echo ""

# 3. Check if map -> odom transform exists
echo "3. Checking map -> odom transform:"
if timeout 2 ros2 run tf2_ros tf2_echo map odom 2>&1 | grep -q "Translation\|At time"; then
    echo "   ✓ map -> odom transform EXISTS"
    timeout 2 ros2 run tf2_ros tf2_echo map odom 2>&1 | head -10
else
    echo "   ✗ map -> odom transform MISSING"
    echo "   → RTAB-Map should publish this!"
fi
echo ""

# 4. Check if odom -> base_footprint transform exists
echo "4. Checking odom -> base_footprint transform:"
if timeout 2 ros2 run tf2_ros tf2_echo odom base_footprint 2>&1 | grep -q "Translation\|At time"; then
    echo "   ✓ odom -> base_footprint transform EXISTS"
    timeout 2 ros2 run tf2_ros tf2_echo odom base_footprint 2>&1 | head -10
else
    echo "   ✗ odom -> base_footprint transform MISSING"
    echo "   → odom_to_tf node should publish this!"
fi
echo ""

# 5. Check if base_footprint -> base_link transform exists
echo "5. Checking base_footprint -> base_link transform:"
if timeout 2 ros2 run tf2_ros tf2_echo base_footprint base_link 2>&1 | grep -q "Translation\|At time"; then
    echo "   ✓ base_footprint -> base_link transform EXISTS"
    timeout 2 ros2 run tf2_ros tf2_echo base_footprint base_link 2>&1 | head -10
else
    echo "   ✗ base_footprint -> base_link transform MISSING"
    echo "   → This might be needed if Nav2 expects base_link"
fi
echo ""

# 6. Generate TF tree visualization
echo "6. Generating TF tree (saved to frames.pdf):"
ros2 run tf2_tools view_frames 2>&1 | tail -5
if [ -f frames.pdf ]; then
    echo "   ✓ TF tree saved to frames.pdf"
    echo "   → Open with: evince frames.pdf"
else
    echo "   ✗ Failed to generate TF tree"
fi
echo ""

# 7. List all available transforms
echo "7. All TF frames currently available:"
timeout 2 ros2 run tf2_ros tf2_monitor 2>&1 | grep -E "Frame|Frames" | head -20
echo ""

# 8. Check TF buffer status
echo "8. TF buffer status:"
timeout 2 ros2 topic echo /tf --once 2>&1 | head -5
echo ""

# 9. Check transform timestamps vs current time
echo "9. Checking transform timestamps:"
echo "   (If transforms are from the past, planner_server won't accept them)"
CURRENT_TIME=$(date +%s)
echo "   Current system time: $CURRENT_TIME"
echo "   Check view_frames output above for transform timestamps"
echo "   If timestamps are much older, transforms are STALE"
echo ""

# 10. Diagnosis
echo "=== Diagnosis ==="
echo ""
echo "Required TF chain for Nav2 planner_server:"
echo "  map -> odom -> base_footprint -> base_link"
echo ""
echo "From view_frames output above:"
echo "  - If all transforms are listed: ✓ Chain structure exists"
echo "  - Check transform timestamps: Should be recent (within last few seconds)"
echo ""
echo "Common issues:"
echo "  1. STALE TRANSFORMS: Timestamps are from the past"
echo "     → Restart RTAB-Map and odom_to_tf nodes"
echo "     → Check if /clock topic is being published (sim_time issue)"
echo ""
echo "  2. TIMING MISMATCH: Transforms exist but timestamps don't match"
echo "     → Increase transform_tolerance in nav2_params.yaml (>= 2.0)"
echo "     → Check if use_sim_time is consistent across all nodes"
echo ""
echo "  3. FRAME NAMES: Planner expects base_link but config says base_footprint"
echo "     → Check nav2_params.yaml robot_base_frame setting"
echo "     → Nav2 might internally need base_link even if config says base_footprint"

