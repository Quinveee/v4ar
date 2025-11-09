# v4ar - Vision for Autonomous Robots

A ROS2-based autonomous robot vision system for line detection and following. This project provides modular line detection algorithms and intelligent line following control with multiple steering strategies.

## Project Structure

```
src/
├── line_msgs/           # Custom ROS2 message definitions
├── line_detector/       # Line detection algorithms (Canny, Brightness, Gradient, Custom)
├── line_follower/       # Line following control with multiple strategies
├── opencv_tools/        # OpenCV utilities for image processing
└── visualizations/      # Visualization and plotting nodes
```

## Building the Project

```bash
cd /path/to/v4ar
colcon build
source install/setup.bash
```

## Core Nodes

### 1. Image Subscriber (opencv_tools)

Subscribes to camera feed and displays processed images.

```bash
ros2 run opencv_tools img_subscriber_uni
```

### 2. Line Detector Node (line_detector)

Detects lines in camera feed using various algorithms.

**Basic usage:**

```bash
ros2 run line_detector line_detector_node
```

**With parameters:**

```bash
# Using custom detector (default)
ros2 run line_detector line_detector_node --ros-args -p detector_type:=custom

# Using Canny edge detection
ros2 run line_detector line_detector_node --ros-args -p detector_type:=canny

# Using brightness-based detection
ros2 run line_detector line_detector_node --ros-args -p detector_type:=brightness

# Using gradient-based detection
ros2 run line_detector line_detector_node --ros-args -p detector_type:=gradient

# Using skeleton detection
ros2 run line_detector line_detector_node --ros-args -p detector_type:=skeleton

# Enable display window
ros2 run line_detector line_detector_node --display_window

# Enable vignette masking (for custom detector)
ros2 run line_detector line_detector_node --vignette
```

### 3. Line Follower Node (line_follower)

Controls robot movement to follow detected lines.

**Basic usage:**

```bash
ros2 run line_follower line_follower
```

**With speed control modes:**

```bash
# Gradual speed control (default) - smooth slowdown based on angle
ros2 run line_follower line_follower --speed_control gradual

# Threshold speed control - half speed if angle > 30 degrees
ros2 run line_follower line_follower --speed_control threshold

# Angle-based speed control - increase speed when angle < threshold
ros2 run line_follower line_follower --speed_control angle_based --angle_threshold 5.0

# No speed adjustment - constant speed
ros2 run line_follower line_follower --speed_control none
```

**With line selection strategies:**

```bash
# Closest to center (default)
ros2 run line_follower line_follower --selector closest

# Confidence-based tracking
ros2 run line_follower line_follower --selector confidence

# Mean of all detected lines
ros2 run line_follower line_follower --selector mean

# Tracking-based selection
ros2 run line_follower line_follower --selector tracking
```

**With smoothing and control parameters:**

```bash
# EMA smoothing factor (0-1, higher = more weight to recent values)
ros2 run line_follower line_follower --smoothing_factor 0.5

# Forward speed (default: 0.2)
ros2 run line_follower line_follower --forward_speed 0.3

# Proportional gain for angle (default: 0.01)
ros2 run line_follower line_follower --k_angle 0.015

# Proportional gain for offset (default: 0.005)
ros2 run line_follower line_follower --k_offset 0.008

# Enable line extension to full screen
ros2 run line_follower line_follower --extend_lines
```

**Combined example:**

```bash
ros2 run line_follower line_follower \
  --speed_control angle_based \
  --angle_threshold 5.0 \
  --selector confidence \
  --smoothing_factor 0.3 \
  --forward_speed 0.25 \
  --k_angle 0.01
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

## Complete System Startup

Run all nodes together for a complete line following system:

```bash
# Terminal 1: Image subscriber
ros2 run opencv_tools img_subscriber_uni

# Terminal 2: Line detector
ros2 run line_detector line_detector_node --ros-args -p detector_type:=skeleton

# Terminal 3: Line follower
ros2 run line_follower line_follower --speed_control gradual --k_angle 0.4 --selector tracking --forward_speed 0.35

# Terminal 4 (optional): Visualization
ros2 run visualizations visualization_node

# Terminal 5 (optional): Angle plotting
ros2 run visualizations angle_plot_node
```

Or use the provided shell scripts:

```bash
# Basic line following
bash ros_cmds/line_following.sh

# Line following with confidence selector
bash ros_cmds/line_following_confidence.sh

# Camera only
bash ros_cmds/cam.sh
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
- `angle_threshold`: Threshold for angle-based speed control (in degrees).

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
