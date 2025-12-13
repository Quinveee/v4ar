# Navigation Online Package

Online navigation system with RTAB-Map mapping and Dijkstra planning. This package implements a client-server architecture where the laptop performs all computation (mapping, planning, command generation) and the rover simply executes commands.

## Architecture

### Laptop Side (Computation)
- **RTAB-Map SLAM**: Creates maps from sensor data
- **RTAB-Map Bridge**: Converts RTAB-Map maps to standard occupancy grids
- **Online Navigator**: Plans paths using Dijkstra algorithm and generates movement commands

### Rover Side (Execution)
- **Command Executor**: Receives commands from laptop and executes them on the rover

## Package Structure

```
navigation_online/
├── navigation_online/
│   ├── rtabmap_bridge.py      # RTAB-Map to occupancy grid converter
│   ├── online_navigator.py    # Main navigation node (laptop side)
│   └── command_executor.py   # Command execution node (rover side)
├── launch/
│   ├── laptop_side.launch.py # Launch file for laptop
│   ├── rover_side.launch.py  # Launch file for rover
│   └── online_navigation.launch.py  # Complete system (for testing)
└── README.md
```

## Building

```bash
cd /path/to/v4ar
colcon build --packages-select navigation_online
source install/setup.bash
```

## Usage

### Setup

1. **Network Configuration**: Ensure laptop and rover are on the same network and ROS2 can communicate between them (same `ROS_DOMAIN_ID`).

2. **Prerequisites**:
   - Robot sensors publishing (camera, depth, odometry)
   - Robot odometry on `/odom` topic
   - RTAB-Map configured to publish occupancy grids (see RTAB-Map configuration)

### Laptop Side

Run on the laptop to perform mapping, planning, and command generation:

```bash
ros2 launch navigation_online laptop_side.launch.py
```

With options:
```bash
# Enable RViz visualization
ros2 launch navigation_online laptop_side.launch.py use_rviz:=true

# Use simulation time (for bag playback)
ros2 launch navigation_online laptop_side.launch.py use_sim_time:=true

# Custom depth filter level
ros2 launch navigation_online laptop_side.launch.py depth_filter_level:=2
```

### Rover Side

Run on the rover to execute commands:

```bash
ros2 launch navigation_online rover_side.launch.py
```

With options:
```bash
# Custom command topic
ros2 launch navigation_online rover_side.launch.py cmd_vel_topic:=/ugv/cmd_vel

# Custom safety limits
ros2 launch navigation_online rover_side.launch.py \
    max_linear_speed:=0.4 \
    max_angular_speed:=1.2
```

### Sending Goals

Once both sides are running, send navigation goals:

```bash
# Using ros2 topic pub
ros2 topic pub --once /goal geometry_msgs/PoseStamped "
{
  header: {frame_id: 'map'},
  pose: {
    position: {x: 2.0, y: 3.0, z: 0.0},
    orientation: {w: 1.0}
  }
}"

# Or use RViz "2D Nav Goal" tool
```

## Topics

### Laptop Side Topics

**Subscribes:**
- `/map` (nav_msgs/OccupancyGrid): Map from RTAB-Map bridge
- `/odom` (nav_msgs/Odometry): Robot odometry from rover
- `/goal` (geometry_msgs/PoseStamped): Goal pose for navigation

**Publishes:**
- `/navigation_commands` (geometry_msgs/Twist): Movement commands for rover
- `/planned_path` (nav_msgs/Path): Planned path for visualization

### Rover Side Topics

**Subscribes:**
- `/navigation_commands` (geometry_msgs/Twist): Commands from laptop

**Publishes:**
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands to rover base controller

## RTAB-Map Configuration

For RTAB-Map to publish occupancy grids, ensure it's configured with:

```yaml
Grid/FromDepth: "true"
Grid/2D: "true"  # Enable 2D occupancy grid
Grid/CellSize: "0.05"  # 5cm resolution
```

The bridge node subscribes to `/rtabmap/grid_map` if available, or can extract from `/rtabmap/mapData`.

## Parameters

### Online Navigator Parameters

- `occupied_threshold` (default: 50): Occupancy value threshold for obstacles
- `k_linear` (default: 0.5): Proportional gain for linear velocity
- `k_angular` (default: 2.0): Proportional gain for angular velocity
- `max_linear_speed` (default: 0.3): Maximum linear speed (m/s)
- `max_angular_speed` (default: 1.0): Maximum angular speed (rad/s)
- `waypoint_threshold` (default: 0.15): Distance to advance to next waypoint (m)
- `goal_threshold` (default: 0.1): Distance to consider goal reached (m)
- `control_frequency` (default: 10.0): Control loop frequency (Hz)

### Command Executor Parameters

- `cmd_vel_topic` (default: `/cmd_vel`): Topic for rover velocity commands
- `max_linear_speed` (default: 0.5): Safety limit for linear speed (m/s)
- `max_angular_speed` (default: 1.5): Safety limit for angular speed (rad/s)
- `enable_safety_limits` (default: true): Enable safety speed limits

## RViz Visualization

When you run with `use_rviz:=true`, RViz will show:

1. **Map Display**: 
   - Add "Map" display
   - Topic: `/map`
   - Shows occupancy grid (white=free, black=obstacles, gray=unknown)
   - Updates in real-time as RTAB-Map builds the map

2. **Path Display**:
   - Add "Path" display  
   - Topic: `/planned_path`
   - Shows the planned path from Nav2 planner (green/yellow line)
   - Updates when new goals are set

3. **Robot Model**:
   - Add "RobotModel" display
   - Shows robot's current position and orientation
   - Uses TF transforms

4. **RTAB-Map Visualization**:
   - RTAB-Map's own visualization shows 3D point clouds and graph
   - Topic: `/rtabmap/mapData` and `/rtabmap/mapGraph`

**To see everything:**
```bash
ros2 launch navigation_online laptop_side.launch.py use_rviz:=true
```

Then in RViz:
- Add "Map" → Topic: `/map`
- Add "Path" → Topic: `/planned_path`  
- Add "RobotModel"
- Add RTAB-Map displays if needed

## Integration with Existing Packages

This package integrates with:
- **var_mapping**: Uses RTAB-Map launch files for mapping
- **navigation**: Uses Nav2's planner server for path planning (not custom Dijkstra)
- **navigation_dijkstra**: Similar concept but this uses Nav2 instead

## Troubleshooting

### No map received
- Check RTAB-Map is running and publishing `/rtabmap/grid_map`
- Verify RTAB-Map is configured to publish 2D occupancy grids
- Check bridge node logs

### No commands received on rover
- Verify network connectivity between laptop and rover
- Check `ROS_DOMAIN_ID` matches on both machines
- Verify `/navigation_commands` topic is being published

### Path planning fails
- Ensure map is received (check `/map` topic)
- Verify robot odometry is available on `/odom`
- Check goal is within map bounds

## Bonus Features

This package implements the bonus requirements:
1. ✅ **RTAB-Map Integration**: Bridge converts RTAB-Map maps to occupancy grids
2. ✅ **Online Navigation**: Navigate while actively mapping
3. ✅ **Dynamic Map Updates**: Replans when map updates
4. ✅ **Client-Server Architecture**: Laptop computes, rover executes
5. ✅ **Nav2 Integration**: Uses Nav2's planner server (not custom Dijkstra)
6. ✅ **RViz Visualization**: Map and path visualization in real-time

