# Quick Start Guide - Navigation Online

## Overview

This package implements online navigation where:
- **Laptop** does all computation (mapping, planning, command generation)
- **Rover** simply executes commands

## Quick Setup

### 1. Build the Package

```bash
cd /path/to/v4ar
colcon build --packages-select navigation_online
source install/setup.bash
```

### 2. Configure Network

Ensure laptop and rover can communicate:
- Same network
- Same `ROS_DOMAIN_ID` (e.g., `export ROS_DOMAIN_ID=0`)

### 3. Start Laptop Side

On your laptop:

```bash
ros2 launch navigation_online laptop_side.launch.py use_rviz:=true
```

This starts:
- RTAB-Map SLAM (mapping)
- RTAB-Map bridge (converts to occupancy grid)
- Online navigator (plans and generates commands)

### 4. Start Rover Side

On the rover:

```bash
ros2 launch navigation_online rover_side.launch.py
```

This starts:
- Command executor (listens and executes commands)

### 5. Send Navigation Goal

Send a goal to navigate:

```bash
# Using command line
ros2 topic pub --once /goal geometry_msgs/PoseStamped "
{
  header: {frame_id: 'map'},
  pose: {
    position: {x: 2.0, y: 3.0, z: 0.0},
    orientation: {w: 1.0}
  }
}"
```

Or use RViz "2D Nav Goal" tool.

## What Happens

1. **RTAB-Map** creates a map from sensor data
2. **Bridge** converts map to occupancy grid
3. **Navigator** plans path using Dijkstra algorithm
4. **Navigator** generates velocity commands
5. **Rover** receives and executes commands

## Troubleshooting

### Check Topics

```bash
# On laptop - should see:
ros2 topic list | grep -E "(map|goal|navigation_commands|planned_path)"

# On rover - should see:
ros2 topic list | grep navigation_commands
```

### Check Map

```bash
ros2 topic echo /map --once
```

### Check Commands

```bash
ros2 topic echo /navigation_commands
```

## Testing on Same Machine

For testing, you can run both sides on the same machine:

```bash
# Terminal 1: Laptop side
ros2 launch navigation_online laptop_side.launch.py

# Terminal 2: Rover side  
ros2 launch navigation_online rover_side.launch.py

# Terminal 3: Send goal
ros2 topic pub --once /goal geometry_msgs/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 3.0}, orientation: {w: 1.0}}}"
```

## Next Steps

- Tune parameters in launch files for your robot
- Adjust control gains for smoother motion
- Configure RTAB-Map for your environment
- See README.md for detailed documentation

