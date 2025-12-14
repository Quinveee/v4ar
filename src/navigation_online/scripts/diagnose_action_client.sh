#!/bin/bash
# Script to diagnose why action client isn't showing up in ros2 action info

echo "=== Diagnosing Action Client Connection ==="
echo ""

echo "1. Check if online_navigator node is running:"
ros2 node list | grep online_navigator || echo "   ✗ online_navigator node NOT FOUND"
echo ""

echo "2. Check what action topic the client should be using:"
echo "   Expected: /planner_server/compute_path_to_pose"
echo "   Checking both possible topics..."
echo ""

echo "3. Check /planner_server/compute_path_to_pose (correct topic):"
ros2 action info /planner_server/compute_path_to_pose 2>&1
echo ""

echo "4. Check /compute_path_to_pose (wrong topic if planner_server is empty):"
ros2 action info /compute_path_to_pose 2>&1
echo ""

echo "5. List all compute_path_to_pose action servers:"
ros2 action list | grep compute_path_to_pose
echo ""

echo "6. Check online_navigator node info (what topics it's using):"
if ros2 node list | grep -q online_navigator; then
    ros2 node info /online_navigator 2>&1 | grep -A 5 -B 5 "action\|Action" || echo "   (No action-related info found)"
else
    echo "   ✗ online_navigator node not running"
fi
echo ""

echo "7. Check online_navigator logs for action client creation:"
echo "   Look in your launch terminal for:"
echo "   - '[INIT] Created action client for ...'"
echo "   - '[PLANNER] Action server ... not yet discovered'"
echo ""

echo "8. Check if planner_server parameter is set correctly:"
echo "   Run this in Python to check the parameter:"
echo "   ros2 param get /online_navigator planner_server"
echo ""

echo "=== Diagnosis ==="
echo ""
echo "If 'Action clients: 0' for /planner_server/compute_path_to_pose:"
echo "  → The client exists but hasn't completed DDS discovery yet"
echo "  → Wait 60+ seconds and check again"
echo ""
echo "If 'Action clients: 0' for /compute_path_to_pose:"
echo "  → Client is using wrong topic (planner_server parameter might be empty)"
echo "  → Check: ros2 param get /online_navigator planner_server"
echo ""
echo "If online_navigator node not found:"
echo "  → Node not running - check launch file"
echo ""

