# Nav2 Navigation Strategy - Usage Guide

## Overview

The `nav2_strategy` node integrates with Nav2 stack to navigate to goals while visualizing obstacles detected by the obstacle detector. It:
- Subscribes to detected obstacles from `/detected_rovers`
- Transforms obstacles from camera frame to world frame
- Sends navigation goals to Nav2 via action interface
- Publishes obstacles for visualization on `/nav2/obstacles_world`

## Prerequisites

1. **Nav2 Stack Running**: Nav2 must be running with:
   - `bt_navigator` (behavior tree navigator)
   - `planner_server` (path planner)
   - `controller_server` (path follower)
   - `costmap_2d` (obstacle detection from laser scan)
   - `lifecycle_manager` (manages Nav2 nodes)

2. **Laser Scan**: Nav2 needs `/scan` topic (2D laser scan) for obstacle avoidance

3. **TF Tree**: Required transforms:
   - `map` → `odom` (from your localization)
   - `odom` → `base_link` (from odometry, e.g., rf2o_laser_odometry)
   - `base_link` → `laser_frame` (static transform)

4. **Obstacle Detector**: Run the obstacle detector to publish `/detected_rovers`

## Usage

### Basic Setup

**Terminal 1: Start Nav2 Stack**
```bash
# Launch Nav2 (you'll need a Nav2 launch file configured for your robot)
ros2 launch nav2_bringup bringup_launch.py
```

**Terminal 2: Start Obstacle Detector**
```bash
ros2 run perception obstacle_detector
# Or whatever command runs your obstacle detector
```

**Terminal 3: Start Nav2 Strategy Node**
```bash
ros2 run control nav2_strategy --ros-args \
  -p robot_pose_topic:=/odom_pose_processed \
  -p obstacle_topic:=/detected_rovers \
  -p target_x:=5.0 \
  -p target_y:=4.0 \
  -p auto_navigate:=true
```

**Terminal 4: Start Visualization (Optional)**
```bash
ros2 run visualizations control_visualization --ros-args \
  -p start_x:=1.0 \
  -p start_y:=2.0 \
  -p start_theta:=0.0 \
  -p target_x:=5.0 \
  -p target_y:=4.0
```

### Sending Goals

**Method 1: Via Parameters (Auto-navigate)**
```bash
ros2 run control nav2_strategy --ros-args \
  -p target_x:=5.0 \
  -p target_y:=4.0 \
  -p target_yaw:=1.57 \
  -p auto_navigate:=true
```

**Method 2: Via Service (After node is running)**
```bash
# Set goal via parameters
ros2 param set /nav2_navigation_strategy target_x 5.0
ros2 param set /nav2_navigation_strategy target_y 4.0

# Trigger navigation
ros2 service call /nav2/navigate_to_goal std_srvs/srv/Empty
```

**Method 3: Direct Python API**
```python
# In your code
from control.curling_control.navigation_strategies.nav2_strategy import Nav2NavigationStrategy

node = Nav2NavigationStrategy()
node.navigate_to_goal(target_x=5.0, target_y=4.0, target_yaw=1.57)
```

## Parameters

- `goal_tolerance` (float, default: 0.2): Distance tolerance for goal reached (meters)
- `obstacle_radius` (float, default: 0.3): Radius around each obstacle (for future costmap integration)
- `robot_pose_topic` (string, default: "/odom_pose_processed"): Topic for robot pose
- `obstacle_topic` (string, default: "/detected_rovers"): Topic for detected obstacles
- `target_x` (float, default: 0.0): Target x position in world frame (meters)
- `target_y` (float, default: 0.0): Target y position in world frame (meters)
- `target_yaw` (float, optional): Target orientation (radians)
- `auto_navigate` (bool, default: false): Auto-navigate on startup if true

## Topics

### Subscriptions
- `/odom_pose_processed` (or `robot_pose_topic`): Current robot pose
- `/detected_rovers` (or `obstacle_topic`): Detected obstacles from obstacle detector

### Publications
- `/nav2/obstacles_world` (ObjectPoseArray): Obstacles transformed to world frame (for visualization)
- `/nav2/current_goal` (PoseStamped): Current Nav2 goal (for visualization)

### Services
- `/nav2/navigate_to_goal` (std_srvs/Empty): Trigger navigation to parameter-specified goal

## Visualization

The control visualization shows:
- **Green circle with arrow**: Start position
- **Red circle**: Target position
- **Blue line**: Trajectory
- **Yellow circle with arrow**: Current robot pose
- **Red squares**: Detected obstacles (from camera)

## How It Works

1. **Obstacle Detection**: Obstacle detector publishes rovers in camera frame
2. **Transformation**: Nav2 strategy transforms obstacles to world frame using robot pose
3. **Goal Sending**: Goals are sent to Nav2 via `NavigateToPose` action
4. **Navigation**: Nav2 plans path and follows it, avoiding obstacles from laser scan
5. **Visualization**: Obstacles and goals are visualized on the field

## Notes

- **Laser Scan**: Nav2 uses `/scan` for obstacle avoidance. The camera-detected obstacles are for visualization and could be added to costmap in the future.
- **Frame Convention**: Nav2 typically uses `map` frame. Make sure your localization publishes `map` → `odom` transform.
- **Coordinate System**: Uses the same world frame as your odometry/localization system.

## Troubleshooting

**Nav2 action server not available:**
- Make sure Nav2 stack is running
- Check: `ros2 action list | grep navigate_to_pose`

**Obstacles not showing:**
- Check obstacle detector is publishing: `ros2 topic echo /detected_rovers`
- Check robot pose is available: `ros2 topic echo /odom_pose_processed`

**Navigation not working:**
- Verify TF tree: `ros2 run tf2_tools view_frames`
- Check Nav2 status: `ros2 topic echo /bt_navigator/status`

