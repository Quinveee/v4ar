# Control Module

Control strategies for robot navigation from start to target position.

## Overview

The control module provides a strategy-based approach to robot navigation, similar to the odometry module structure.

## Available Strategies

### `go_to_target` - Navigate to Target Position
- Takes fixed start position and orientation
- Takes fixed target position
- First orients robot to face target
- Then drives straight to target
- Optionally updates position using cmd_vel (dead reckoning)

## Usage

### Basic Usage

```bash
# Run control node with default parameters
ros2 run control control_node
```

### With Custom Parameters

```bash
ros2 run control control_node --ros-args \
  -p start_x:=1.0 \
  -p start_y:=2.0 \
  -p start_theta:=0.0 \
  -p target_x:=5.0 \
  -p target_y:=4.0 \
  -p use_odometry_update:=true \
  -p pose_topic:=/odom_pose_processed
```

### With Visualization

**Terminal 1: Start control node**
```bash
ros2 run control control_node --ros-args \
  -p start_x:=1.0 \
  -p start_y:=2.0 \
  -p start_theta:=0.0 \
  -p target_x:=5.0 \
  -p target_y:=4.0
```

**Terminal 2: Start visualization**
```bash
ros2 run visualizations control_visualization --ros-args \
  -p start_x:=1.0 \
  -p start_y:=2.0 \
  -p start_theta:=0.0 \
  -p target_x:=5.0 \
  -p target_y:=4.0
```

## Parameters

### Control Node (`control_node`)

- `start_x` (float, default: 0.0): Starting x position in world frame (meters)
- `start_y` (float, default: 0.0): Starting y position in world frame (meters)
- `start_theta` (float, default: 0.0): Starting orientation in world frame (radians)
- `target_x` (float, default: 3.0): Target x position in world frame (meters)
- `target_y` (float, default: 4.5): Target y position in world frame (meters)
- `use_odometry_update` (bool, default: true): Update position estimate using cmd_vel
- `update_rate` (float, default: 20.0): Control loop frequency (Hz)
- `pose_topic` (string, default: "/odom_pose_processed"): Topic for current robot pose

### Visualization Node (`control_visualization`)

- `start_x`, `start_y`, `start_theta`: Same as control node
- `target_x`, `target_y`: Same as control node
- `field_image`: Path to field image (optional)
- `draw_field`: Whether to draw field programmatically (default: true)
- `scale_px_per_mm`: Field drawing scale (default: 0.1)

## Topics

### Subscriptions

- `/odom_pose_processed` (or `pose_topic`): Current robot pose from odometry

### Publications

- `/cmd_vel` (geometry_msgs/Twist): Velocity commands to robot
- `/control/temp_pose` (geometry_msgs/PoseStamped): Current robot pose estimate (for visualization)
- `/control/temp_trajectory` (geometry_msgs/PoseStamped): Trajectory points (for visualization)

## How It Works

1. **Initialization**: Control strategy is initialized with start and target positions
2. **Phase 1 - Orientation**: Robot rotates to face target (if not already facing it)
3. **Phase 2 - Driving**: Robot drives straight to target
4. **Position Update**: If `use_odometry_update=true`, position is updated using cmd_vel for dead reckoning
5. **Completion**: When within tolerance (0.1m), robot stops

## Visualization

The visualization shows:
- **Green circle with arrow**: Start position and orientation
- **Red circle**: Target position
- **Blue line**: Trajectory (path taken by robot)
- **Yellow circle with arrow**: Current robot pose and orientation

## Example Workflow

```bash
# Terminal 1: Start odometry (if using pose updates)
ros2 run perception odometry_node --ros-args -p strategy_type:=cmd_vel

# Terminal 2: Start control
ros2 run control control_node --ros-args \
  -p start_x:=1.0 \
  -p start_y:=2.0 \
  -p start_theta:=0.0 \
  -p target_x:=5.0 \
  -p target_y:=4.0

# Terminal 3: Start visualization
ros2 run visualizations control_visualization --ros-args \
  -p start_x:=1.0 \
  -p start_y:=2.0 \
  -p start_theta:=0.0 \
  -p target_x:=5.0 \
  -p target_y:=4.0
```

