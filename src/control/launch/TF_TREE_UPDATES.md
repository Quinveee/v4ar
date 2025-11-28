# TF Tree Updates for Nav2 Launch

## Your TF Tree Structure

Based on your `/tf` and `/tf_static` topics:

```
map (from your localization)
  └─ odom (from RF2O or odometry)
      └─ base_footprint (robot base on ground)
          └─ base_link (robot center)
              ├─ base_lidar_link (laser scanner)
              ├─ base_imu_link (IMU)
              ├─ 3d_camera_link (camera)
              ├─ pt_base_link (pan-tilt base)
              └─ pt_camera_link (pan-tilt camera)
```

## Changes Made to Launch File

### 1. Nav2 Costmap Frames

**Changed from:** `robot_base_frame: 'base_link'`  
**Changed to:** `robot_base_frame: 'base_footprint'`

**Why:**
- Nav2 typically uses `base_footprint` as the robot base frame
- `base_footprint` represents the robot's contact point with the ground
- Your TF tree has `odom` → `base_footprint`, which is the standard Nav2 convention

**Affects:**
- `global_costmap`: Now uses `base_footprint`
- `local_costmap`: Now uses `base_footprint`

### 2. RF2O Base Frame

**Changed from:** `base_frame_id: 'base_link'`  
**Changed to:** `base_frame_id: 'base_footprint'`

**Why:**
- RF2O publishes `odom` → `base_frame_id` transform
- Your TF tree shows `odom` → `base_footprint`
- This ensures RF2O's output matches your existing TF structure

**Note:** RF2O will publish `odom` → `base_footprint`, which matches your current TF tree.

### 3. Static Transforms

**Changed:** `publish_static_tf` default from `true` to `false`

**Why:**
- Your robot description/URDF already publishes all static transforms
- No need to duplicate them
- Prevents TF conflicts

**If you need to override:**
```bash
ros2 launch control nav2_with_detector_launch.py \
  publish_static_tf:=true
```

**Updated static transform names:**
- `base_link` → `base_lidar_link` (matches your actual frame)
- `base_link` → `3d_camera_link` (matches your actual frame)

### 4. Nav2 Strategy Frame Usage

**No changes needed!**

The `nav2_strategy` node:
- Uses `frame_id: "map"` for goals (correct for Nav2)
- Doesn't directly reference `base_link` or `base_footprint`
- Gets robot pose from `/odom_pose_processed` topic
- Nav2 handles all frame transformations internally

**Why it works:**
- Nav2 receives goals in `map` frame
- Nav2 uses its costmap configuration (`base_footprint`) to plan paths
- Nav2 publishes commands that work with your TF tree structure
- No changes needed in `nav2_strategy.py`

## Visualization Added

**New parameter:** `launch_visualization` (default: `false`)

**Usage:**
```bash
ros2 launch control nav2_with_detector_launch.py \
  launch_visualization:=true \
  start_x:=2.0 \
  start_y:=1.0 \
  start_theta:=0.0 \
  nav2_strategy_target_x:=5.0 \
  nav2_strategy_target_y:=4.0
```

**What it shows:**
- Green circle with arrow: Start position
- Red circle: Target position
- Blue line: Trajectory
- Yellow circle with arrow: Current robot pose
- Red squares: Detected obstacles

## Complete TF Tree for Nav2

**Required transforms:**
1. `map` → `odom` (from your localization/AprilTags)
2. `odom` → `base_footprint` (from RF2O or odometry)
3. `base_footprint` → `base_link` (from your URDF/robot description)
4. `base_link` → `base_lidar_link` (from your URDF/robot description)

**Your current setup:**
- ✅ `map` → `odom`: From your localization (AprilTags)
- ✅ `odom` → `base_footprint`: From RF2O (when launched)
- ✅ `base_footprint` → `base_link`: From your URDF (static)
- ✅ `base_link` → `base_lidar_link`: From your URDF (static)

**Everything is correctly configured!**

## Frame Naming Convention

**Nav2 Standard:**
- `map`: Global reference frame (from localization)
- `odom`: Local reference frame (from odometry)
- `base_footprint`: Robot base on ground (Nav2's robot frame)
- `base_link`: Robot center (for sensors, actuators)

**Your Setup:**
- Matches Nav2 convention perfectly
- `base_footprint` is the robot base for Nav2
- `base_link` is used for sensor frames
- All transforms are properly connected

## Verification

**Check your TF tree:**
```bash
ros2 run tf2_tools view_frames
# Generates frames.pdf showing your TF tree
```

**Check specific transform:**
```bash
ros2 run tf2_ros tf2_echo map base_footprint
ros2 run tf2_ros tf2_echo odom base_footprint
ros2 run tf2_ros tf2_echo base_footprint base_link
```

**Check Nav2 is using correct frames:**
```bash
ros2 param get /global_costmap robot_base_frame
ros2 param get /local_costmap robot_base_frame
```

Should both return: `base_footprint`

## Summary

✅ **Nav2 costmap**: Now uses `base_footprint` (matches your TF tree)  
✅ **RF2O**: Now uses `base_footprint` (matches your TF tree)  
✅ **Static transforms**: Disabled by default (you already have them)  
✅ **Nav2 strategy**: No changes needed (works with any TF tree)  
✅ **Visualization**: Added with `launch_visualization` parameter  

**Everything should work correctly with your existing TF tree!**

