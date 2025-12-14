#!/bin/bash
# Script to check planner_server node status and lifecycle state

echo "=== Checking Planner Server Status ==="
echo ""

# 1. Check if planner_server node is running
echo "1. Checking if planner_server node exists:"
if ros2 node list | grep -q "planner_server"; then
    echo "   ✓ planner_server node is running"
    ros2 node list | grep planner_server
else
    echo "   ✗ planner_server node is NOT running"
    echo "   → Check launch file - planner_server should be started"
    echo "   → This is the main problem!"
fi
echo ""

# 2. Check lifecycle_manager node
echo "2. Checking if lifecycle_manager node exists:"
if ros2 node list | grep -q "lifecycle_manager"; then
    echo "   ✓ lifecycle_manager node is running"
    ros2 node list | grep lifecycle_manager
else
    echo "   ✗ lifecycle_manager node is NOT running"
fi
echo ""

# 3. Check for planner_server lifecycle services
echo "3. Checking planner_server lifecycle services:"
if ros2 service list | grep -q "planner_server.*lifecycle\|planner_server.*get_state"; then
    echo "   ✓ planner_server lifecycle services found:"
    ros2 service list | grep planner_server | grep -E "(lifecycle|get_state|change_state)"
else
    echo "   ✗ planner_server lifecycle services NOT found"
    echo "   → This means planner_server node is not running or not configured as lifecycle node"
fi
echo ""

# 4. Check if action server is available
echo "4. Checking Nav2 planner action server:"
if ros2 action list | grep -q "compute_path_to_pose"; then
    echo "   ✓ Planner action server is available:"
    ros2 action list | grep compute_path_to_pose
    echo ""
    echo "   This means planner_server is fully activated!"
else
    echo "   ✗ Planner action server is NOT available"
    echo "   → planner_server is not running or not activated"
fi
echo ""

# 5. Check all nodes
echo "5. All running nodes:"
ros2 node list
echo ""

# 6. Check if nav2_planner package is installed
echo "6. Checking if nav2_planner package is available:"
if ros2 pkg list | grep -q "nav2_planner"; then
    echo "   ✓ nav2_planner package is installed"
else
    echo "   ✗ nav2_planner package is NOT installed"
    echo "   → Install with: sudo apt install ros-humble-nav2-planner"
fi
echo ""

# 7. Diagnosis
echo "=== Diagnosis ==="
if ! ros2 node list | grep -q "planner_server"; then
    echo "PROBLEM: planner_server node is not running!"
    echo ""
    echo "Possible causes:"
    echo "  1. Launch file didn't start planner_server"
    echo "  2. planner_server crashed on startup (check launch terminal for errors)"
    echo "  3. nav2_planner package not installed"
    echo "  4. Launch file has an error"
    echo ""
    echo "Check your launch terminal for:"
    echo "  - [planner_server-*] logs"
    echo "  - Any errors about 'planner_server' or 'nav2_planner'"
    echo "  - Check if process started: [INFO] [planner_server-10]: process started"
fi

