# v4ar - Vision for Autonomous Robots

A ROS2-based autonomous robot vision system for line detection and following. This project provides modular line detection algorithms and intelligent line following control with multiple steering strategies.

## Project Structure

```
src/
├── image_tools/         # Image processing utilities for camera feed handling
│   └── line_msgs/       # Custom ROS2 message definitions
├── perception/          # Line detection algorithms (Canny, Brightness, Gradient, Custom)
├── control/             # Control module for robot navigation
│   └── line_follower/   # Line following control with multiple strategies
└── visualizations/      # Visualization and plotting nodes
```

## Building the Project

```bash
cd /path/to/v4ar
colcon build
source install/setup.bash
```

## Quick Start - Line Following System

The easiest way to run the complete line following system is using the launch file from the control package:

```bash
ros2 launch line_follower line_following.launch.py
```

With custom parameters:

```bash
ros2 launch line_follower line_following.launch.py \
  detector_type:=custom \
  speed_control:=angle_based \
  selector:=confidence \
  forward_speed:=0.25 \
  enable_visualization:=true
```

This launches the complete pipeline: image subscriber → perception → line follower → optional visualization

## Core Nodes

### 1. Image Subscriber (image_tools)

Subscribes to camera feed and displays processed images.

**Direct execution:**

```bash
ros2 run image_tools img_subscriber_uni
```

**Via launch file:**

```bash
ros2 launch image_tools image_subscriber.launch.py
```

### 2. Perception Node (perception)

Detects lines in camera feed using various algorithms.

**Direct execution:**

```bash
ros2 run perception line_detector --ros-args -p detector_type:=custom
```

**With detector type parameter:**

```bash
ros2 run perception line_detector --ros-args -p detector_type:=custom
```

Available detectors: `custom`, `canny`, `brightness`, `gradient`, `skeleton`

**Via launch file:**

```bash
ros2 launch perception image_detection.launch.py detector_type:=custom
```

### 3. Line Follower Node (line_follower)

Controls robot movement to follow detected lines.

**Direct execution:**

```bash
ros2 run line_follower line_follower
```

**With parameters:**

```bash
ros2 run line_follower line_follower \
  --speed_control angle_based \
  --selector confidence \
  --smoothing_factor 0.3 \
  --forward_speed 0.25
```

**Via launch file:**

```bash
ros2 launch line_follower line_follower.launch.py \
  speed_control:=angle_based \
  selector:=confidence
```

### 4. Visualization Node (visualizations)

Visualizes detected lines on the camera feed.

```bash
ros2 run visualizations visualization_node
```

### 5. Angle Plot Node (visualizations)

Plots line angle measurements over time.

```bash
ros2 run visualizations angle_plot_node
```

## Launch Files

### Complete Line Following System

```bash
ros2 launch line_follower line_following.launch.py
```

**Available parameters:**

- `detector_type` (default: custom) - Detection algorithm
- `speed_control` (default: gradual) - Speed control mode
- `selector` (default: closest) - Line selection strategy
- `smoothing_factor` (default: 0.3) - EMA smoothing factor
- `forward_speed` (default: 0.2) - Base forward velocity
- `enable_visualization` (default: false) - Enable visualization node

**Example with all parameters:**

```bash
ros2 launch line_follower line_following.launch.py \
  detector_type:=custom \
  speed_control:=angle_based \
  selector:=confidence \
  smoothing_factor:=0.4 \
  forward_speed:=0.3 \
  enable_visualization:=true
```

### Individual Launch Files

Image subscriber:

```bash
ros2 launch image_tools image_subscriber.launch.py
```

Image detection:

```bash
ros2 launch image_detection image_detection.launch.py detector_type:=custom
```

Line follower:

```bash
ros2 launch line_follower line_follower.launch.py speed_control:=angle_based
```

## Line Follower Control Modes

### Speed Control Modes

- **gradual**: Smooth inverse relationship between angle and speed. At 0 degrees: 100% speed, at 90 degrees: 50% speed.
- **threshold**: Simple threshold-based control. Half speed if angle > 30 degrees, else full speed.
- **angle_based**: Increase speed when angle is below threshold (e.g., ±5 degrees), reduce speed when above.
- **none**: Constant speed regardless of angle.

### Line Selection Strategies

- **closest**: Selects line closest to image center.
- **confidence**: Tracks lines based on detection confidence scores.
- **mean**: Uses average of all detected lines.
- **tracking**: Maintains tracking state across frames.

## ROS2 Topics

### Published Topics

- `/cmd_vel` (geometry_msgs/Twist): Robot velocity commands
- `/detected_lines` (line_msgs/DetectedLines): Detected line segments
- `/selected_line` (line_msgs/DetectedLine): Currently selected line for following
- `/line_follower/smoothed_angle` (std_msgs/Float32): Smoothed line angle
- `/processed_image` (sensor_msgs/Image): Processed image with detections

### Subscribed Topics

- `/camera/image_raw` (sensor_msgs/Image): Raw camera feed
- `/detected_lines` (line_msgs/DetectedLines): Line detection results

## Configuration

Key parameters for tuning:

- `smoothing_factor`: EMA smoothing (0.0-1.0). Higher values = more responsive but noisier.
- `forward_speed`: Base forward velocity (0.0-1.0).
- `k_angle`: Steering gain for angle error. Higher = more aggressive steering.
- `k_offset`: Steering gain for lateral offset. Higher = more aggressive centering.

## Message Definitions

### DetectedLine

```
float32 x1, y1, x2, y2  # Line endpoints
float32 offset_x         # Horizontal offset from image center
float32 angle            # Line angle in radians
float32 confidence       # Detection confidence (0-1)
```

### DetectedLines

```
DetectedLine[] lines     # Array of detected lines
```
