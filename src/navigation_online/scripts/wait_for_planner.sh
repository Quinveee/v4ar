#!/bin/bash
# Script to wait for planner_server action server to be discoverable

echo "=== Waiting for planner_server action server to be ready ==="
echo ""
echo "This script will check every 5 seconds if the action server is discoverable."
echo "Press Ctrl+C to stop waiting."
echo ""

PLANNER_ACTION="/planner_server/compute_path_to_pose"
MAX_WAIT=120  # Maximum wait time in seconds
CHECK_INTERVAL=5  # Check every 5 seconds
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if action server exists
    if ros2 action list 2>&1 | grep -q "$PLANNER_ACTION"; then
        echo "✓ Action server $PLANNER_ACTION EXISTS"
        
        # Try to get action info (this tests if it's actually discoverable)
        if ros2 action info "$PLANNER_ACTION" > /dev/null 2>&1; then
            echo ""
            echo "✓✓✓ ACTION SERVER IS READY AND DISCOVERABLE! ✓✓✓"
            echo ""
            echo "You can now:"
            echo "  1. Play your rosbag (if not already playing)"
            echo "  2. Send goals to /goal topic"
            echo ""
            exit 0
        else
            echo "  ⏳ Server exists but not yet discoverable (DDS discovery in progress)..."
        fi
    else
        echo "  ⏳ Action server not found yet... (waited ${ELAPSED}s)"
    fi
    
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
    echo "  (Elapsed: ${ELAPSED}s / ${MAX_WAIT}s)"
done

echo ""
echo "⚠ Timeout after ${MAX_WAIT} seconds."
echo "The action server may still be initializing."
echo "Check planner_server logs for errors."
exit 1

