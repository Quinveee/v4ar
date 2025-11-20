#!/bin/bash
# Line following with confidence-based line selector
# This version uses the confidence selector for more stable tracking

ros2 run opencv_tools img_subscriber_uni & 
ros2 run line_detector line_detector_node --ros-args -p detector_type:=custom &
ros2 run line_follower line_follower --selector confidence &

