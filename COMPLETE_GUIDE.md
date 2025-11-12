# Complete Guide to v4ar

## Project Overview

v4ar is a ROS2-based autonomous robot vision system for line detection and following with modular architecture.

## New Package Structure

```
src/
├── image_tools/              # Image processing utilities
│   └── line_msgs/            # Custom ROS2 messages (CMake)
├── perception/               # Line detection (formerly image_detection)
├── control/                  # Control module (meta-package)
│   └── line_follower/        # Line following control
└── visualizations/           # Visualization nodes
```

## Build Instructions

```bash
cd /path/to/v4ar
colcon build
source install/setup.bash
```

## Running the System

### Quick Start (All-in-One)
```bash
ros2 launch line_follower line_following.launch.py
```

### Individual Nodes

| Node | Command |
|------|---------|
| Image Subscriber | `ros2 run image_tools img_subscriber_uni` |
| Line Detector | `ros2 run perception line_detector` |
| Line Follower | `ros2 run line_follower line_follower` |
| Visualization | `ros2 run visualizations visualization_node` |

### With Parameters

```bash
# Perception with custom detector
ros2 run perception line_detector --ros-args -p detector_type:=custom

# Line Follower with angle-based speed control
ros2 run line_follower line_follower --ros-args -p speed_control:=angle_based

# Complete system with all parameters
ros2 launch line_follower line_following.launch.py \
  detector_type:=custom \
  speed_control:=angle_based \
  selector:=confidence \
  forward_speed:=0.25 \
  enable_visualization:=true
```

## Available Parameters

### Perception
- `detector_type`: custom, canny, brightness, gradient, skeleton

### Line Follower
- `speed_control`: gradual, threshold, angle_based, none
- `selector`: closest, confidence, mean, tracking
- `smoothing_factor`: 0-1 (EMA smoothing)
- `forward_speed`: base velocity
- `k_angle`: steering gain

## ROS2 Topics

### Published
- `/detected_lines` (line_msgs/DetectedLines)
- `/cmd_vel` (geometry_msgs/Twist)

### Subscribed
- `/camera/image_raw` (sensor_msgs/Image)
- `/detected_lines` (line_msgs/DetectedLines)

## Documentation Files

- `README.md` - Project overview
- `QUICK_REFERENCE.md` - Command reference
- `RUNNING_NODES.md` - Detailed node instructions
- `PACKAGE_RESTRUCTURING.md` - Architecture details
- `BUILD_FIXES.md` - Build troubleshooting

