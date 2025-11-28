# Odometry Strategies

This directory contains multiple odometry strategies for fusing sensor data to estimate robot pose.

## Available Strategies

### 1. `cmd_vel` - Dead Reckoning
- Uses velocity commands (`/cmd_vel`) for dead reckoning
- Simple, but accumulates error over time
- Good for short-term odometry

### 2. `lidar` - RF2O Laser Odometry
- Uses `rf2o_laser_odometry` package
- Subscribes to `/odom_rf2o` (rf2o's output)
- Transforms rf2o's odometry to world frame
- Good accuracy from laser scan matching

### 3. `sensor_fusion` - Custom EKF
- Custom Extended Kalman Filter implementation
- Fuses: rf2o odometry, IMU, triangulation
- All processing in Python
- More control, but less tested than robot_localization

### 4. `robot_localization` - Production EKF (RECOMMENDED)
- Uses ROS2's `robot_localization` package
- Industry-standard sensor fusion
- Well-tested and maintained
- Requires separate EKF node to be running

## File Structure

### Core Files
- `odometry.py` - Main odometry node that uses strategies
- `odometry_strategies/` - Directory containing all strategy implementations
  - `base_strategy.py` - Base class for all strategies
  - `cmd_vel_strategy.py` - Dead reckoning strategy
  - `lidar_strategy.py` - RF2O laser odometry strategy
  - `sensor_fusion_strategy.py` - Custom EKF implementation
  - `robot_localization_strategy.py` - Wrapper for robot_localization

### Configuration Files
- `config/odometry_robot_localization.yaml` - Configuration for robot_localization's EKF node
  - Defines sensor inputs (rf2o, IMU)
  - Sets noise models
  - Configures frame IDs

### Launch Files
- `launch/odometry_robot_localization.launch.py` - Launches robot_localization's EKF node
  - Starts `ekf_localization_node`
  - Loads configuration
  - Remaps topics

## Usage

### Using robot_localization (Recommended)

**Step 1: Launch robot_localization EKF node**
```bash
ros2 launch perception odometry_robot_localization.launch.py
```

**Step 2: Run odometry node with robot_localization strategy**
```bash
ros2 run perception odometry_node --ros-args \
  -p strategy_type:=robot_localization \
  -p robot_localization_topic:=/odometry/filtered \
  -p triangulation_topic:=/robot_pose_raw
```

### Using sensor_fusion (Custom EKF)

```bash
ros2 run perception odometry_node --ros-args \
  -p strategy_type:=sensor_fusion \
  -p rf2o_odom_topic:=/odom_rf2o \
  -p imu_topic:=/imu/data \
  -p triangulation_topic:=/robot_pose_raw
```

### Using lidar (RF2O only)

```bash
ros2 run perception odometry_node --ros-args \
  -p strategy_type:=lidar \
  -p rf2o_odom_topic:=/odom_rf2o \
  -p triangulation_topic:=/robot_pose_raw
```

### Using cmd_vel (Dead reckoning)

```bash
ros2 run perception odometry_node --ros-args \
  -p strategy_type:=cmd_vel \
  -p triangulation_topic:=/robot_pose_raw
```

## How It Works

1. **Initialization**: First triangulation message (`/robot_pose_raw`) sets the world frame origin
2. **Sensor Updates**: Strategy-specific sensors update the odometry estimate
3. **World Frame Transformation**: All strategies transform sensor data to world frame
4. **Publishing**: Filtered odometry published to `/odom_pose_processed`

## Topic Subscriptions (by strategy)

### robot_localization
- `/odometry/filtered` - Filtered odometry from robot_localization EKF
- `/robot_pose_raw` - Triangulation (for initialization + corrections)

### sensor_fusion
- `/odom_rf2o` - RF2O laser odometry
- `/imu/data` - IMU data
- `/robot_pose_raw` - Triangulation (for initialization + corrections)

### lidar
- `/odom_rf2o` - RF2O laser odometry
- `/robot_pose_raw` - Triangulation (for initialization)

### cmd_vel
- `/cmd_vel` - Velocity commands
- `/robot_pose_raw` - Triangulation (for initialization)

## Output

All strategies publish to:
- `/odom_pose_processed` - `geometry_msgs/PoseStamped` in world frame

## Rebuilding Package

After making changes, rebuild:
```bash
cd /home/jetson/ugv_ws/v4ar
colcon build --packages-select perception
source install/setup.bash
```

