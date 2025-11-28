# Nav2 Launch File - Usage Guide

## Overview

The `nav2_with_detector_launch.py` launch file starts:
- **Nav2 Stack**: Complete navigation stack (bt_navigator, planner_server, controller_server, costmap_2d, lifecycle_manager)
- **Obstacle Detector**: Camera-based obstacle detection
- **Nav2 Strategy** (optional): Integration node for sending goals to Nav2
- **Static TF Transforms**: Robot structure transforms (base_link → laser_frame, base_link → camera_frame)

## Basic Usage

### Minimal Launch (Default Parameters)
```bash
ros2 launch control nav2_with_detector_launch.py
```

### With Custom Speed Parameters
```bash
ros2 launch control nav2_with_detector_launch.py \
  max_vel_x:=0.5 \
  max_vel_theta:=1.5 \
  acc_lim_x:=0.8
```

### With Nav2 Strategy and Auto-Navigation
```bash
ros2 launch control nav2_with_detector_launch.py \
  launch_nav2_strategy:=true \
  nav2_strategy_target_x:=5.0 \
  nav2_strategy_target_y:=4.0 \
  nav2_strategy_auto_navigate:=true
```

## Speed Configuration Parameters

Nav2 speed is controlled by these parameters:

### Linear Velocity
- `max_vel_x` (default: 0.3 m/s): Maximum forward/backward speed
- `min_vel_x` (default: 0.05 m/s): Minimum forward speed

### Angular Velocity
- `max_vel_theta` (default: 1.0 rad/s): Maximum rotation speed (~57°/s)
- `min_vel_theta` (default: 0.1 rad/s): Minimum rotation speed

### Acceleration Limits
- `acc_lim_x` (default: 0.5 m/s²): Linear acceleration/deceleration limit
- `acc_lim_theta` (default: 1.0 rad/s²): Angular acceleration/deceleration limit

### Example Speed Configurations

**Slow/Precise Navigation:**
```bash
ros2 launch control nav2_with_detector_launch.py \
  max_vel_x:=0.15 \
  min_vel_x:=0.02 \
  max_vel_theta:=0.5 \
  acc_lim_x:=0.3
```

**Fast Navigation:**
```bash
ros2 launch control nav2_with_detector_launch.py \
  max_vel_x:=0.6 \
  min_vel_x:=0.1 \
  max_vel_theta:=2.0 \
  acc_lim_x:=1.0
```

**Curling Field (Recommended):**
```bash
ros2 launch control nav2_with_detector_launch.py \
  max_vel_x:=0.3 \
  min_vel_x:=0.05 \
  max_vel_theta:=1.0 \
  acc_lim_x:=0.5
```

## All Launch Parameters

### Nav2 Configuration
- `use_sim_time` (default: 'false'): Use simulation time
- `autostart` (default: 'true'): Auto-start Nav2 lifecycle nodes
- `use_lifecycle_mgr` (default: 'true'): Use lifecycle manager
- `map` (default: ''): Path to map YAML file (if using static map)
- `params_file` (default: ''): Path to custom Nav2 parameters file
- `use_nav2_bringup` (default: 'true'): Use Nav2 bringup launch (true) or launch nodes individually (false)

### Speed Parameters (see above)
- `max_vel_x`, `min_vel_x`
- `max_vel_theta`, `min_vel_theta`
- `acc_lim_x`, `acc_lim_theta`

### Obstacle Detector
- `detector_no_gui` (default: 'true'): Disable GUI window
- `detector_max_distance` (default: '4.0'): Maximum detection distance (meters)
- `use_detector_node` (default: 'true'): Use detector as ROS node (true) or Python script (false)

### Nav2 Strategy
- `launch_nav2_strategy` (default: 'false'): Launch Nav2 strategy node
- `nav2_strategy_target_x` (default: '0.0'): Target x position
- `nav2_strategy_target_y` (default: '0.0'): Target y position
- `nav2_strategy_auto_navigate` (default: 'false'): Auto-navigate on startup

### TF Transforms
- `publish_static_tf` (default: 'true'): Publish static transforms for robot structure

## Prerequisites

Before launching, ensure:

1. **Nav2 is installed:**
   ```bash
   sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
   ```

2. **TF Tree is set up:**
   - `map` → `odom` transform (from your localization)
   - `odom` → `base_link` transform (from odometry, e.g., rf2o_laser_odometry)
   - Static transforms will be published by this launch file

3. **Laser scan is publishing:**
   - Topic: `/scan` (sensor_msgs/LaserScan)
   - Nav2 uses this for obstacle avoidance

4. **Robot pose is available:**
   - Topic: `/odom_pose_processed` (or configure via `robot_pose_topic` parameter)

## Troubleshooting

### Nav2 nodes not starting
- Check if Nav2 is installed: `ros2 pkg list | grep nav2`
- Check lifecycle manager status: `ros2 lifecycle get /lifecycle_manager_navigation`

### Detector not running
- If using node mode, ensure entry point exists in perception package
- If using script mode, check path: `ros2 pkg prefix perception`
- Check detector output: `ros2 topic echo /detected_rovers`

### Speed too fast/slow
- Adjust `max_vel_x` and `max_vel_theta` parameters
- Lower values = slower, more precise
- Higher values = faster, less precise

### TF errors
- Verify transforms: `ros2 run tf2_tools view_frames`
- Check static transforms are being published
- Ensure localization is publishing `map` → `odom`

## Example Complete Workflow

```bash
# Terminal 1: Launch Nav2 with detector
ros2 launch control nav2_with_detector_launch.py \
  max_vel_x:=0.3 \
  launch_nav2_strategy:=true \
  nav2_strategy_target_x:=5.0 \
  nav2_strategy_target_y:=4.0

# Terminal 2: Send goal via service (if not using auto-navigate)
ros2 service call /nav2/navigate_to_goal std_srvs/srv/Empty

# Terminal 3: Visualization (optional)
ros2 run visualizations control_visualization --ros-args \
  -p target_x:=5.0 \
  -p target_y:=4.0
```

## Notes

- **Speed Parameters**: These affect both path planning and path following. Lower speeds = safer navigation but slower arrival.
- **Acceleration Limits**: Control how quickly the robot accelerates/decelerates. Lower values = smoother motion.
- **Detector Mode**: If detector node doesn't work, set `use_detector_node:=false` to run as Python script.
- **Nav2 Bringup**: By default uses Nav2's bringup launch file. Set `use_nav2_bringup:=false` to launch nodes individually for more control.

