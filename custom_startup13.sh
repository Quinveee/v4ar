#!/bin/bash

export ROS_DOMAIN_ID=13
echo "ROS_DOMAIN_ID = $ROS_DOMAIN_ID"

# --- ugv_bringup ---
if ! pgrep -f "ros2 run ugv_bringup ugv_bringup" > /dev/null; then
    echo "Starting ugv_bringup..."
    ros2 run ugv_bringup ugv_bringup &
    PID=$!
    echo "ugv_bringup started with PID $PID"
else
    PID=$(pgrep -f "ros2 run ugv_bringup ugv_bringup")
    echo "ugv_bringup is already running with PID $PID"
fi


if ! pgrep -f "ros2 run ugv_bringup ugv_driver" > /dev/null; then 
   echo "Starting uvg_driver..."
   ros2 run ugv_bringup ugv_driver &
   PID=$!
   echo "ugv_driver started with PID $PID"
else
  PID=$(pgrep -f "ros2 run ugv_bringup ugv_driver")
  echo "ugv_driverp is alredy running with PID $PID"
fi

# --- camera.launch.py ---
if ! pgrep -f "ros2 launch ugv_vision camera.launch.py" > /dev/null; then
    echo "Starting camera.launch.py..."
    ros2 launch ugv_vision camera.launch.py &
    PID=$!
    echo "camera.launch.py started with PID $PID"
else
    PID=$(pgrep -f "ros2 launch ugv_vision camera.launch.py")
    echo "camera.launch.py is already running with PID $PID"
fi
