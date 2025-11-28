# Nav2 Launch File - Complete Usage Guide

## Quick Start

### Basic Launch (RF2O Odometry - Default)
```bash
ros2 launch control nav2_with_detector_launch.py
```

### With Dead Reckoning Odometry
```bash
ros2 launch control nav2_with_detector_launch.py \
  odometry_source:=dead_reckoning
```

### Complete Example with All Options
```bash
ros2 launch control nav2_with_detector_launch.py \
  # Starting position
  start_x:=2.0 \
  start_y:=1.0 \
  start_theta:=0.0 \
  # Target position
  nav2_strategy_target_x:=5.0 \
  nav2_strategy_target_y:=4.0 \
  nav2_strategy_auto_navigate:=true \
  launch_nav2_strategy:=true \
  # Odometry source
  odometry_source:=rf2o \
  # Speed
  max_vel_x:=0.3 \
  max_vel_theta:=1.0 \
  # Visualization
  launch_visualization:=true
```

## Odometry Source Options

### Option 1: RF2O Laser Odometry (Default)

**What it is:**
- Uses laser scan matching to estimate motion
- More accurate than dead reckoning
- Requires working laser scanner

**When to use:**
- You have a working 2D laser scanner
- You want more accurate odometry
- Laser scans are available at reasonable frequency (10+ Hz)

**Launch:**
```bash
ros2 launch control nav2_with_detector_launch.py \
  odometry_source:=rf2o
```

**Requirements:**
- `/scan` topic publishing (sensor_msgs/LaserScan)
- `rf2o_laser_odometry` package installed:
  ```bash
  sudo apt install ros-humble-rf2o-laser-odometry
  ```

**Output:**
- Publishes to `/odom_rf2o` (nav_msgs/Odometry)
- Publishes TF: `odom` → `base_footprint`

### Option 2: Dead Reckoning from cmd_vel

**What it is:**
- Integrates velocity commands to estimate position
- Simpler, no laser required
- Less accurate (drifts over time)

**When to use:**
- No laser scanner available
- Testing/debugging
- Short-term navigation only
- Laser scanner is not working

**Launch:**
```bash
ros2 launch control nav2_with_detector_launch.py \
  odometry_source:=dead_reckoning
```

**Requirements:**
- `/cmd_vel` topic publishing (geometry_msgs/Twist)
- Nav2 or other node publishing velocity commands

**Output:**
- Publishes to `/odom` (nav_msgs/Odometry)
- Publishes TF: `odom` → `base_footprint`

**Note:** Dead reckoning will drift over time. For long-term navigation, use RF2O or combine with localization updates.

## Complete Parameter Reference

### Odometry Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `odometry_source` | `rf2o` | Odometry source: `rf2o` or `dead_reckoning` |
| `launch_rf2o` | `true` | Launch RF2O node (only if `odometry_source=rf2o`) |
| `rf2o_laser_scan_topic` | `/scan` | Laser scan topic for RF2O |
| `rf2o_odom_topic` | `/odom_rf2o` | RF2O output odometry topic |
| `dead_reckoning_odom_topic` | `/odom` | Dead reckoning output odometry topic |
| `dead_reckoning_cmd_vel_topic` | `/cmd_vel` | Input cmd_vel topic for dead reckoning |

### Navigation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_x` | `0.0` | Starting x position (meters) |
| `start_y` | `0.0` | Starting y position (meters) |
| `start_theta` | `0.0` | Starting orientation (radians) |
| `nav2_strategy_target_x` | `5.0` | Target x position (meters) |
| `nav2_strategy_target_y` | `4.0` | Target y position (meters) |
| `nav2_strategy_auto_navigate` | `true` | Auto-navigate on startup |
| `launch_nav2_strategy` | `false` | Launch Nav2 strategy node |

### Speed Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_vel_x` | `0.3` | Maximum linear velocity (m/s) |
| `min_vel_x` | `0.05` | Minimum linear velocity (m/s) |
| `max_vel_theta` | `1.0` | Maximum angular velocity (rad/s) |
| `min_vel_theta` | `0.1` | Minimum angular velocity (rad/s) |
| `acc_lim_x` | `0.5` | Linear acceleration limit (m/s²) |
| `acc_lim_theta` | `1.0` | Angular acceleration limit (rad/s²) |

### Visualization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `launch_visualization` | `false` | Launch control visualization node |

### Other Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `autostart` | `true` | Auto-start Nav2 lifecycle nodes |
| `publish_static_tf` | `false` | Publish static transforms (disable if URDF already does) |

## Common Use Cases

### Use Case 1: Navigation with Laser Odometry

```bash
ros2 launch control nav2_with_detector_launch.py \
  odometry_source:=rf2o \
  launch_nav2_strategy:=true \
  nav2_strategy_target_x:=5.0 \
  nav2_strategy_target_y:=4.0 \
  launch_visualization:=true
```

### Use Case 2: Navigation with Dead Reckoning (No Laser)

```bash
ros2 launch control nav2_with_detector_launch.py \
  odometry_source:=dead_reckoning \
  launch_nav2_strategy:=true \
  nav2_strategy_target_x:=5.0 \
  nav2_strategy_target_y:=4.0 \
  launch_visualization:=true
```

### Use Case 3: Testing Nav2 Stack Only

```bash
ros2 launch control nav2_with_detector_launch.py \
  odometry_source:=dead_reckoning \
  launch_nav2_strategy:=false \
  launch_visualization:=false
```

### Use Case 4: Full Stack with Custom Speed

```bash
ros2 launch control nav2_with_detector_launch.py \
  odometry_source:=rf2o \
  launch_nav2_strategy:=true \
  nav2_strategy_target_x:=10.0 \
  nav2_strategy_target_y:=8.0 \
  max_vel_x:=0.5 \
  max_vel_theta:=1.5 \
  launch_visualization:=true
```

## Verification

### Check Odometry is Publishing

**RF2O:**
```bash
ros2 topic echo /odom_rf2o --once
```

**Dead Reckoning:**
```bash
ros2 topic echo /odom --once
```

### Check TF Tree

```bash
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo odom base_footprint
```

### Check Nav2 Status

```bash
ros2 lifecycle get /lifecycle_manager_navigation
ros2 action list | grep navigate_to_pose
```

### Check Obstacle Detector

```bash
ros2 topic echo /detected_rovers --once
```

## Troubleshooting

### RF2O Not Working

**Symptoms:**
- No odometry published
- TF transform missing

**Solutions:**
1. Check laser scan is publishing: `ros2 topic echo /scan --once`
2. Check RF2O is installed: `ros2 pkg list | grep rf2o`
3. Try dead reckoning: `odometry_source:=dead_reckoning`

### Dead Reckoning Not Working

**Symptoms:**
- No odometry published
- Robot position not updating

**Solutions:**
1. Check cmd_vel is publishing: `ros2 topic echo /cmd_vel --once`
2. Verify Nav2 is sending commands
3. Check dead_reckoning_odom node is running: `ros2 node list | grep dead_reckoning`

### Nav2 Not Navigating

**Symptoms:**
- Goal sent but robot doesn't move
- Nav2 reports errors

**Solutions:**
1. Check TF tree: `ros2 run tf2_tools view_frames`
2. Verify `map` → `odom` transform exists (from localization)
3. Verify `odom` → `base_footprint` transform exists (from odometry)
4. Check Nav2 lifecycle: `ros2 lifecycle get /lifecycle_manager_navigation`

## Architecture Overview

```
┌─────────────────┐
│  Localization   │ → map → odom (TF)
│  (AprilTags)    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Odometry       │ → odom → base_footprint (TF)
│  (RF2O or DR)   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Nav2 Stack     │ → /cmd_vel
│  (Navigation)   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Obstacle       │ → /detected_rovers
│  Detector       │
└─────────────────┘
```

## Notes

- **Odometry Source:** Choose based on available sensors. RF2O is more accurate but requires laser.
- **Dead Reckoning Drift:** Dead reckoning will drift over time. Use with localization updates for long-term navigation.
- **TF Tree:** Ensure `map` → `odom` → `base_footprint` chain exists for Nav2 to work.
- **Visualization:** Enable visualization to see robot position, target, and obstacles on field.

