# Package Restructuring Summary

## New Project Structure

```
src/
├── image_tools/              # Image processing utilities
│   ├── image_tools/          # Python package
│   ├── launch/
│   │   └── image_subscriber.launch.py
│   ├── setup.py
│   └── package.xml
│
├── perception/               # Line detection (formerly image_detection)
│   ├── perception/           # Python package
│   ├── launch/
│   │   └── image_detection.launch.py
│   ├── setup.py
│   └── package.xml
│
├── control/                  # Control module (NEW)
│   ├── line_follower/        # Line following (moved here)
│   │   ├── line_follower/    # Python package
│   │   ├── launch/
│   │   │   ├── line_follower.launch.py
│   │   │   └── line_following.launch.py (main)
│   │   ├── setup.py
│   │   └── package.xml
│   ├── package.xml
│   └── CMakeLists.txt
│
├── visualizations/           # Visualization nodes
│   ├── visualizations/       # Python package
│   ├── setup.py
│   └── package.xml
│
├── line_msgs/                # Custom message definitions
│   ├── msg/
│   │   ├── DetectedLine.msg
│   │   └── DetectedLines.msg
│   ├── CMakeLists.txt
│   └── package.xml
│
└── line_detection/           # Legacy (can be removed)
```

## Key Changes

### 1. Renamed Packages
- `image_detection` → `perception`
- Updated all references in package.xml, setup.py, and launch files

### 2. New Control Package
- Created `control` as a meta-package (CMake-based)
- `line_follower` moved into `control/line_follower`
- Maintains all functionality and dependencies

### 3. Updated Dependencies
- `visualizations/package.xml`: Now depends on `perception` instead of `image_detection`
- `line_follower/package.xml`: Still depends on `line_msgs`
- `perception/package.xml`: Still depends on `line_msgs`

### 4. Launch Files Updated
- `perception/launch/image_detection.launch.py`: Uses `perception` package
- `control/line_follower/launch/line_following.launch.py`: References `perception` instead of `image_detection`

## Running the System

### Build
```bash
cd /path/to/v4ar
colcon build
source install/setup.bash
```

### Run Individual Nodes
```bash
ros2 run image_tools img_subscriber_uni
ros2 run perception line_detector
ros2 run line_follower line_follower
```

### Run Complete System
```bash
ros2 launch line_follower line_following.launch.py
```

## Package Dependencies

```
image_tools
  └── (no ROS dependencies)

perception
  ├── rclpy
  ├── sensor_msgs
  ├── cv_bridge
  └── line_msgs

control (meta-package)
  └── line_follower
      ├── rclpy
      ├── geometry_msgs
      └── line_msgs

visualizations
  ├── rclpy
  ├── line_msgs
  ├── perception
  └── line_follower

line_msgs
  ├── std_msgs
  └── rosidl_default_runtime
```

