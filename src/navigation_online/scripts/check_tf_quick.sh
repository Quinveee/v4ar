#!/bin/bash
# Quick TF and planner_server diagnostic commands

echo "=== Quick TF & Planner Server Diagnostics ==="
echo ""

echo "1. Check if map -> base_link transform is QUERYABLE (most critical):"
echo "   Command: ros2 run tf2_ros tf2_echo map base_link"
echo "   Expected: Should show translation/rotation values, NOT 'Invalid frame ID'"
echo "   Running now..."
timeout 3 ros2 run tf2_ros tf2_echo map base_link 2>&1 | head -15
echo ""

echo "2. Check planner_server lifecycle state (must be 'active'):"
echo "   Command: ros2 service call /planner_server/get_state lifecycle_msgs/srv/GetState"
echo "   Expected: current_state.id should be 3 (active), NOT 1 (unconfigured) or 2 (inactive)"
echo "   Running now..."
ros2 service call /planner_server/get_state lifecycle_msgs/srv/GetState 2>&1 | grep -A 5 "current_state\|id\|label" || echo "   ✗ Service call failed - planner_server may not be running"
echo ""

echo "3. Check if action server EXISTS (if exists, planner is activated):"
echo "   Command: ros2 action list | grep compute_path_to_pose"
echo "   Expected: Should show /planner_server/compute_path_to_pose"
echo "   Running now..."
if ros2 action list 2>&1 | grep -q "compute_path_to_pose"; then
    echo "   ✓ Action server EXISTS:"
    ros2 action list | grep compute_path_to_pose
    echo "   → Planner is activated, but may not be discoverable by client"
else
    echo "   ✗ Action server NOT FOUND"
    echo "   → Planner is NOT activated or not running"
fi
echo ""

echo "4. Check TF transform timestamps vs current ROS time:"
echo "   Command: Check if transform timestamps match current ROS time"
echo "   Expected: Transform timestamps should be close to current ROS time (within tolerance)"
echo "   Running now..."
CURRENT_ROS_TIME=$(ros2 topic echo /clock --once 2>&1 | grep -E "sec:" | head -1 | grep -oE "[0-9]+" | head -1)
if [ -n "$CURRENT_ROS_TIME" ]; then
    echo "   Current ROS time (from /clock): $CURRENT_ROS_TIME"
    echo "   Checking transform timestamps..."
    # Try to get a transform and check its timestamp
    TF_OUTPUT=$(timeout 2 ros2 run tf2_ros tf2_echo map base_link 2>&1)
    if echo "$TF_OUTPUT" | grep -q "At time"; then
        TF_TIME=$(echo "$TF_OUTPUT" | grep "At time" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        if [ -n "$TF_TIME" ]; then
            TF_TIME_SEC=$(echo "$TF_TIME" | cut -d. -f1)
            DIFF=$((CURRENT_ROS_TIME - TF_TIME_SEC))
            echo "   Transform timestamp: $TF_TIME_SEC"
            echo "   Time difference: $DIFF seconds"
            if [ $DIFF -gt 10 ] || [ $DIFF -lt -10 ]; then
                echo "   ⚠ WARNING: Transform timestamp is $DIFF seconds from ROS time!"
                echo "   → This may cause planner_server to reject the transform"
            else
                echo "   ✓ Timestamps are close (within 10 seconds)"
            fi
        fi
    else
        echo "   ✗ Could not get transform timestamp"
    fi
else
    echo "   ✗ Could not get ROS time from /clock topic"
fi
echo ""

echo "5. Check if planner_server node is running and what it's doing:"
echo "   Command: ros2 node list | grep planner"
echo "   Expected: Should show planner_server node"
echo "   Running now..."
if ros2 node list 2>&1 | grep -q "planner_server"; then
    echo "   ✓ planner_server node is running:"
    ros2 node list | grep planner_server
    echo ""
    echo "   Now check planner_server logs in your launch terminal for:"
    echo "   - 'Waiting for transform' errors"
    echo "   - 'Costmap not initialized' warnings"
    echo "   - Lifecycle state transitions"
else
    echo "   ✗ planner_server node is NOT running!"
    echo "   → Check launch file - planner_server should be started"
fi
echo ""

echo "6. Check if costmap is publishing (planner needs this):"
echo "   Command: ros2 topic list | grep costmap"
echo "   Expected: Should show /planner_server/global_costmap/costmap"
echo "   Running now..."
if ros2 topic list 2>&1 | grep -q "costmap"; then
    echo "   ✓ Costmap topics found:"
    ros2 topic list | grep costmap
    echo ""
    echo "   Checking if costmap is actually publishing data..."
    COSTMAP_TOPIC=$(ros2 topic list | grep "planner_server.*costmap" | head -1)
    if [ -n "$COSTMAP_TOPIC" ]; then
        echo "   Checking topic: $COSTMAP_TOPIC"
        timeout 2 ros2 topic echo "$COSTMAP_TOPIC" --once 2>&1 | head -5 | grep -E "header|info|data" || echo "   ⚠ Costmap topic exists but no data received"
    fi
else
    echo "   ✗ No costmap topics found"
    echo "   → Costmap may not be initialized (waiting for TF transforms)"
fi
echo ""

echo "7. Check ROS_DOMAIN_ID (DDS discovery issue):"
echo "   Command: echo \$ROS_DOMAIN_ID"
echo "   Expected: Should be same for all nodes (usually 0 or unset)"
echo "   Running now..."
if [ -n "$ROS_DOMAIN_ID" ]; then
    echo "   ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
    echo "   → Make sure ALL nodes use the same domain ID"
else
    echo "   ROS_DOMAIN_ID: not set (defaults to 0)"
    echo "   → This is fine, but verify all nodes use default"
fi
echo ""

echo "=== Summary ==="
echo ""
echo "KEY FINDINGS:"
if ros2 action list 2>&1 | grep -q "compute_path_to_pose"; then
    echo "  ✓ Action server EXISTS - planner is activated"
    echo "  → Issue is DDS discovery: client can't find server"
    echo ""
    echo "SOLUTION: DDS Discovery Issue"
    echo "  1. Wait 60+ seconds for DDS discovery (very slow in Docker)"
    echo "  2. Check ROS_DOMAIN_ID matches on all nodes"
    echo "  3. Restart all nodes to reset DDS discovery"
    echo "  4. Check network connectivity between nodes"
else
    echo "  ✗ Action server NOT FOUND - planner not activated"
    echo "  → Check lifecycle_manager logs for activation errors"
fi
echo ""
if timeout 2 ros2 run tf2_ros tf2_echo map base_link 2>&1 | grep -q "Invalid frame ID"; then
    echo "  ⚠ Transforms not queryable by tf2_echo (command-line tool)"
    echo "  → This is OK if planner_server can query them (it uses ROS node, not command-line)"
    echo "  → Check planner_server logs to see if IT can query transforms"
fi
echo ""
echo "If action server exists but client can't discover it:"
echo "  1. This is a ROS2 DDS discovery issue (common in Docker)"
echo "  2. Wait longer (60+ seconds) - DDS discovery is very slow"
echo "  3. Verify ROS_DOMAIN_ID matches"
echo "  4. Check planner_server logs for TF errors (costmap may be blocked)"

