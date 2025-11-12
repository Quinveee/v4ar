# Quick Reference

## Setup (One Time)

```bash
cd /path/to/v4ar
colcon build
source install/setup.bash
```

## Run Everything (Recommended)

```bash
ros2 launch line_follower line_following.launch.py
```

## Run Individual Nodes

| Component | Command |
|-----------|---------|
| Image Subscriber | `ros2 run image_tools img_subscriber_uni` |
| Line Detector | `ros2 run perception line_detector` |
| Line Follower | `ros2 run line_follower line_follower` |
| Visualization | `ros2 run visualizations visualization_node` |
| Angle Plot | `ros2 run visualizations angle_plot_node` |

## Run via Launch Files

| Component | Command |
|-----------|---------|
| Image Subscriber | `ros2 launch image_tools image_subscriber.launch.py` |
| Perception | `ros2 launch perception image_detection.launch.py` |
| Line Follower | `ros2 launch line_follower line_follower.launch.py` |
| Complete System | `ros2 launch line_follower line_following.launch.py` |

## Common Parameters

### Perception (Line Detector)
```bash
--ros-args -p detector_type:=custom
```
Options: `custom`, `canny`, `brightness`, `gradient`, `skeleton`

### Line Follower
```bash
--ros-args \
  -p speed_control:=angle_based \
  -p selector:=confidence \
  -p forward_speed:=0.25 \
  -p smoothing_factor:=0.3
```

### Complete System
```bash
ros2 launch line_follower line_following.launch.py \
  detector_type:=custom \
  speed_control:=angle_based \
  selector:=confidence \
  forward_speed:=0.25 \
  enable_visualization:=true
```

## Package Locations

- **image_tools**: `src/image_tools/`
- **perception**: `src/perception/`
- **control**: `src/control/`
  - **line_follower**: `src/control/line_follower/`
- **visualizations**: `src/visualizations/`
- **line_msgs**: `src/line_msgs/`

## Topics

### Published
- `/detected_lines` (line_msgs/DetectedLines) - from perception
- `/cmd_vel` (geometry_msgs/Twist) - from line_follower

### Subscribed
- `/camera/image_raw` (sensor_msgs/Image) - by image_tools and perception
- `/detected_lines` (line_msgs/DetectedLines) - by line_follower and visualizations

