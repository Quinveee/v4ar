#!/bin/bash
# Quick debugging script for mapping issues

echo "=== Step 1: Check RTAB-Map Node ==="
ros2 node list | grep rtabmap && echo "✓ RTAB-Map node found" || echo "✗ RTAB-Map node NOT found"

echo ""
echo "=== Step 2: Check RTAB-Map Topics ==="
echo "RTAB-Map topics:"
ros2 topic list | grep rtabmap

echo ""
echo "=== Step 3: Check if grid_map exists ==="
if ros2 topic list | grep -q "/rtabmap/grid_map"; then
    echo "✓ /rtabmap/grid_map topic exists"
    echo "Checking if it has data..."
    timeout 2 ros2 topic hz /rtabmap/grid_map 2>&1 | head -3 || echo "✗ No data on /rtabmap/grid_map"
else
    echo "✗ /rtabmap/grid_map topic NOT found - RTAB-Map not configured for 2D occupancy grid!"
fi

echo ""
echo "=== Step 4: Check /map topic (from bridge) ==="
if ros2 topic list | grep -q "^/map$"; then
    echo "✓ /map topic exists"
    timeout 2 ros2 topic hz /map 2>&1 | head -3 || echo "✗ No data on /map"
else
    echo "✗ /map topic NOT found"
fi

echo ""
echo "=== Step 5: Check TF Tree ==="
if timeout 1 ros2 run tf2_ros tf2_echo map odom 2>&1 | grep -q "Translation"; then
    echo "✓ map->odom transform exists"
else
    echo "✗ map->odom transform NOT found - Frame [map] does not exist!"
fi

echo ""
echo "=== Step 6: Check RTAB-Map Grid/2D Parameter ==="
if ros2 param get /rtabmap Grid/2D 2>&1 | grep -q "true"; then
    echo "✓ Grid/2D is enabled"
else
    echo "✗ Grid/2D is NOT enabled - This is the problem!"
    echo "   RTAB-Map needs Grid/2D: true to publish /rtabmap/grid_map"
fi

echo ""
echo "=== Step 7: Check Sensor Topics ==="
echo "Checking for sensor data..."
ros2 topic list | grep -E "(oak|depth|camera|rgb|scan|odom)" | head -10

echo ""
echo "=== Summary ==="
echo "If Grid/2D is false/missing, RTAB-Map won't publish occupancy grid."
echo "If map->odom transform is missing, RViz can't display the map."

