# Nav2 Launch - Quick Start Guide

## How to Run

### Basic Launch (RF2O Odometry - Default)
```bash
ros2 launch control nav2_with_detector_launch.py
```

### With Dead Reckoning (No Laser)
```bash
ros2 launch control nav2_with_detector_launch.py \
  odometry_source:=dead_reckoning
```

### Complete Example with Navigation
```bash
ros2 launch control nav2_with_detector_launch.py \
  # Odometry
  odometry_source:=rf2o \
  # Starting position
  start_x:=2.0 \
  start_y:=1.0 \
  start_theta:=0.0 \
  # Target position
  nav2_strategy_target_x:=5.0 \
  nav2_strategy_target_y:=4.0 \
  # Enable navigation
  launch_nav2_strategy:=true \
  nav2_strategy_auto_navigate:=true \
  # Enable visualization
  launch_visualization:=true \
  # Speed (optional)
  max_vel_x:=0.3 \
  max_vel_theta:=1.0
```

## Key Parameters

### Odometry Source
- `odometry_source:=rf2o` - Use laser-based odometry (default, requires `/scan`)
- `odometry_source:=dead_reckoning` - Use cmd_vel-based odometry (no laser needed)

### Navigation
- `launch_nav2_strategy:=true` - Enable Nav2 strategy node
- `nav2_strategy_target_x:=5.0` - Target x position
- `nav2_strategy_target_y:=4.0` - Target y position
- `nav2_strategy_auto_navigate:=true` - Auto-navigate on startup

### Visualization
- `launch_visualization:=true` - Show field visualization with robot, target, obstacles

### Starting Position
- `start_x:=2.0` - Starting x (meters)
- `start_y:=1.0` - Starting y (meters)
- `start_theta:=0.0` - Starting orientation (radians)

## What Gets Launched

1. **Nav2 Stack** - Complete navigation system
2. **Obstacle Detector** - Camera-based obstacle detection
3. **Odometry** - RF2O (laser) or Dead Reckoning (cmd_vel)
4. **Nav2 Strategy** (optional) - Sends goals to Nav2
5. **Visualization** (optional) - Field visualization

## TF Tree Requirements

Your TF tree should have:
- `map` → `odom` (from localization)
- `odom` → `base_footprint` (from odometry - RF2O or dead reckoning)
- `base_footprint` → `base_link` (from URDF)
- `base_link` → `base_lidar_link` (from URDF)
- `base_link` → `3d_camera_link` (from URDF)

All of these are already in your system! ✅

## Verification

**Check odometry:**
```bash
# RF2O
ros2 topic echo /odom_rf2o --once

# Dead reckoning
ros2 topic echo /odom --once
```

**Check TF:**
```bash
ros2 run tf2_ros tf2_echo odom base_footprint
```

**Check Nav2:**
```bash
ros2 action list | grep navigate_to_pose
```

**Check obstacles:**
```bash
ros2 topic echo /detected_rovers --once
```

## Common Issues

**No odometry:**
- RF2O: Check `/scan` topic is publishing
- Dead reckoning: Check `/cmd_vel` topic is publishing

**Nav2 not working:**
- Check TF tree: `ros2 run tf2_tools view_frames`
- Verify `map` → `odom` transform exists

**Visualization not showing:**
- Make sure X11 forwarding is enabled (for remote)
- Check window appears (may be behind other windows)

