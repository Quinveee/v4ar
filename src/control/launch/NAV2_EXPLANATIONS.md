# Nav2 Launch File - Explanations

## Nav2 Planners Explained

### nav2_navfn_planner/NavfnPlanner

**What it is:**
- Classic grid-based path planner using Dijkstra or A* algorithms
- Part of the core Nav2 stack (always available)
- Uses potential fields for path smoothing

**How it works:**
- Creates a grid-based costmap from obstacles
- Finds optimal path using Dijkstra (guaranteed shortest) or A* (faster heuristic)
- Smooths path using potential fields

**Best for:**
- Simple environments
- Quick path planning
- Holonomic or non-holonomic robots
- When you need guaranteed optimal paths

**Installation:**
- Usually comes with `ros-humble-navigation2`
- Check: `ros2 pkg list | grep navfn`

### nav2_smac_planner/SmacPlanner

**What it is:**
- State Lattice planner with motion models
- More advanced than NavfnPlanner
- Supports Reeds-Shepp and Dubins curves

**How it works:**
- Searches in state space (x, y, theta) instead of just (x, y)
- Uses motion models (Reeds-Shepp, Dubins, State Lattice)
- Generates smoother, more realistic paths for non-holonomic robots
- Respects robot kinematics (can't turn in place, etc.)

**Best for:**
- Complex environments
- Non-holonomic robots (differential drive, car-like)
- When you need smooth curved paths
- When robot can't turn in place

**Installation:**
- May need: `sudo apt install ros-humble-nav2-smac-planner`
- Check: `ros2 pkg list | grep smac`

**Motion Models:**
- `REEDS_SHEPP`: Allows forward and backward motion, smooth curves
- `DUBIN`: Forward-only motion, smooth curves
- `STATE_LATTICE`: Pre-computed motion primitives

### Which Planner to Use?

**Use NavfnPlanner if:**
- You want simple, fast planning
- Your robot can turn in place easily
- You don't need complex motion models

**Use SmacPlanner if:**
- Your robot has non-holonomic constraints
- You need smooth curved paths
- You want better path quality

**Default in Launch File:**
- Both planners are configured
- Nav2 will use the one specified in the planner plugin parameter
- NavfnPlanner is the fallback (always available)

## Autostart Explained

**What is `autostart`?**

`autostart` is a Nav2 lifecycle parameter that controls whether Nav2 nodes automatically transition to their active state.

**Nav2 Lifecycle States:**
1. **Unconfigured** → Node just started
2. **Inactive** → Node configured but not running
3. **Active** → Node running and processing

**When `autostart=true`:**
- Lifecycle manager automatically transitions all Nav2 nodes: Unconfigured → Inactive → Active
- Nav2 is ready to use immediately after launch
- No manual intervention needed
- **This is the default and recommended setting**

**When `autostart=false`:**
- Nodes stay in Inactive state
- You must manually activate them using lifecycle commands:
  ```bash
  ros2 lifecycle set /controller_server configure
  ros2 lifecycle set /controller_server activate
  ros2 lifecycle set /planner_server configure
  ros2 lifecycle set /planner_server activate
  # ... repeat for all nodes
  ```
- Useful for debugging or when you want manual control

**Why use autostart=false?**
- Debugging: See what happens during activation
- Manual control: Activate nodes in specific order
- Testing: Verify each node individually

**Default:** `autostart=true` (automatic activation)

## Default Values Explained

### Why `nav2_strategy_target_x/y` defaults to 5.0/4.0?

**Previous:** Default was `0.0/0.0` (origin)

**Problem:**
- If target is at origin (0,0) and robot starts at origin, there's no navigation needed
- Easy to forget to set target, leading to confusion
- Not a realistic example

**New Default:** `5.0/4.0` (example coordinates)

**Benefits:**
- Provides a realistic example
- Makes it obvious if you forgot to set your own target
- More intuitive for testing

**You can override:**
```bash
ros2 launch control nav2_with_detector_launch.py \
  nav2_strategy_target_x:=10.0 \
  nav2_strategy_target_y:=8.0
```

### Why `nav2_strategy_auto_navigate` default changed to `true`?

**Previous:** Default was `false` (manual navigation)

**Problem:**
- Users had to manually trigger navigation via service
- Extra step that's easy to forget
- Not intuitive for first-time users

**New Default:** `true` (auto-navigate)

**Benefits:**
- More intuitive: Set target and it navigates automatically
- Less manual steps
- Better for demos and testing

**When to use `false`:**
- When you want to set target but navigate later
- When you need to do other setup before navigation
- When you want manual control via service

**You can override:**
```bash
ros2 launch control nav2_with_detector_launch.py \
  nav2_strategy_auto_navigate:=false
```

## Starting Point Parameters

**Why add starting point?**

Nav2 needs to know where the robot starts for:
- Initial pose estimation
- Path planning from current position
- Visualization

**New Parameters:**
- `start_x` (default: 0.0): Starting x position in world frame (meters)
- `start_y` (default: 0.0): Starting y position in world frame (meters)
- `start_theta` (default: 0.0): Starting orientation (radians)

**Usage:**
```bash
ros2 launch control nav2_with_detector_launch.py \
  start_x:=2.0 \
  start_y:=1.0 \
  start_theta:=0.0
```

**Note:** These are for reference/visualization. Your actual localization (AprilTags, odometry) determines the real starting pose. Nav2 will use the pose from `/odom_pose_processed` or the TF tree.

## RF2O Laser Odometry

**What is RF2O?**

RF2O (Range Flow 2D Odometry) generates odometry from laser scan motion.

**Why use it?**

1. **Smooth Odometry:** Provides high-frequency odometry (20+ Hz) between low-frequency localization updates (AprilTags at ~5-30 Hz)
2. **No Wheel Encoders Needed:** Works purely from laser scans
3. **Short-term Accuracy:** Excellent for local motion estimation
4. **Complements Localization:** AprilTags correct long-term drift, RF2O handles short-term motion

**How it works:**
- Compares consecutive laser scans
- Estimates motion from scan matching
- Publishes odometry on `/odom_rf2o`
- Publishes TF: `odom` → `base_link`

**TF Tree with RF2O:**
```
map → odom → base_link → laser_frame
  ↑       ↑
  |       └─ RF2O publishes this
  |
  └─ Your localization publishes this
```

**Parameters:**
- `launch_rf2o` (default: true): Enable/disable RF2O
- `rf2o_laser_scan_topic` (default: '/scan'): Input laser scan
- `rf2o_odom_topic` (default: '/odom_rf2o'): Output odometry
- `rf2o_base_frame` (default: 'base_link'): Base frame
- `rf2o_odom_frame` (default: 'odom'): Odometry frame

**Installation:**
```bash
sudo apt install ros-humble-rf2o-laser-odometry
```

## Detector Node Fix

**What changed:**

1. **Added entry point** in `perception/setup.py`:
   ```python
   'rover_detector_with_pose = obstacle_detector.detector:main'
   ```

2. **Removed Python script fallback** from launch file (no longer needed)

3. **Simplified launch** to only use ROS node:
   ```python
   Node(
       package='perception',
       executable='rover_detector_with_pose',
       ...
   )
   ```

**Benefits:**
- Proper ROS2 integration
- Better parameter handling
- Cleaner launch file
- Standard ROS2 practice

**Usage:**
Now you can run:
```bash
ros2 run perception rover_detector_with_pose
```

Or via launch file (automatic).

## Summary of Changes

1. ✅ **Added RF2O laser odometry** - Provides smooth odometry from laser scans
2. ✅ **Fixed detector.py** - Added entry point, now proper ROS node
3. ✅ **Removed Python script fallback** - No longer needed
4. ✅ **Added starting point parameters** - `start_x`, `start_y`, `start_theta`
5. ✅ **Updated default values** - More intuitive defaults (target: 5.0/4.0, auto_navigate: true)
6. ✅ **Added planner explanations** - Comments explaining NavfnPlanner vs SmacPlanner
7. ✅ **Enhanced autostart documentation** - Clear explanation of lifecycle management

## Complete Launch Example

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
  # Speed
  max_vel_x:=0.3 \
  max_vel_theta:=1.0 \
  # Enable Nav2 strategy
  launch_nav2_strategy:=true \
  # RF2O (enabled by default)
  launch_rf2o:=true
```

