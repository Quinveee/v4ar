# Step-by-Step Debugging Guide

## Problem: "Frame [map] does not exist" and no mapping visible

### Root Causes Found:
1. ✅ **FIXED**: RTAB-Map was missing `Grid/2D: "true"` parameter
   - **Fix Applied**: Added `Grid/2D: "true"` to `var_mapping/launch/map_generator.launch.py`
   - This enables RTAB-Map to publish `/rtabmap/grid_map` (2D occupancy grid)

2. **Still Need to Check**: RTAB-Map may not be publishing map->odom transform
   - RTAB-Map should publish this automatically in SLAM mode
   - But it only publishes after it has enough data to create a map

## Debugging Steps (Run in Docker Container)

### Step 1: Verify RTAB-Map is Running
```bash
ros2 node list | grep rtabmap
```
**Expected**: Should see `/rtabmap` node

---

### Step 2: Check if Grid/2D is Enabled
```bash
ros2 param get /rtabmap Grid/2D
```
**Expected**: Should return `true` (we just fixed this)

**If false**: The fix didn't apply - rebuild var_mapping package:
```bash
cd ~/v4ar
colcon build --packages-select var_mapping
source install/setup.bash
```

---

### Step 3: Check if /rtabmap/grid_map Topic Exists
```bash
ros2 topic list | grep grid_map
```
**Expected**: Should see `/rtabmap/grid_map`

**If missing**: RTAB-Map hasn't created a map yet, or Grid/2D is still false

---

### Step 4: Check if Map Has Data
```bash
# Check if topic has messages
ros2 topic hz /rtabmap/grid_map

# Or check one message
ros2 topic echo /rtabmap/grid_map --once
```
**Expected**: Should see occupancy grid data

**If empty**: RTAB-Map is running but hasn't created a map yet. This is normal if:
- Rosbag just started playing
- Not enough sensor data yet
- RTAB-Map is still initializing

---

### Step 5: Check if Bridge is Receiving Map
```bash
ros2 topic echo /map --once
```
**Expected**: Should see occupancy grid message with `header.frame_id = "map"`

**If empty**: Bridge isn't receiving from RTAB-Map. Check bridge logs.

---

### Step 6: Check TF Tree for Map Frame
```bash
ros2 run tf2_ros tf2_echo map odom
```
**Expected**: Should see transform from map to odom

**If error "Frame [map] does not exist"**: 
- RTAB-Map hasn't published the transform yet
- This is normal if RTAB-Map hasn't created a map yet
- Wait for RTAB-Map to process some sensor data

**Alternative check:**
```bash
ros2 run tf2_tools view_frames
# Opens frames.pdf - check if 'map' frame exists
```

---

### Step 7: Check Sensor Data is Available
```bash
# Check camera topics
ros2 topic list | grep -E "(oak|depth|camera|rgb)"

# Check if topics have data
ros2 topic hz /oak/rgb/image_rect
ros2 topic hz /oak/stereo/image_raw

# Check odometry
ros2 topic hz /odom

# Check laser scan (if using)
ros2 topic hz /scan
```

**Expected**: All topics should be publishing at reasonable rates

**If topics are empty**: 
- Rosbag might not be playing
- Topics names might not match
- Check rosbag playback command

---

### Step 8: Check RTAB-Map Logs
Look for RTAB-Map output in the terminal where you launched. Look for:
- "RTAB-Map started"
- "Creating new map"
- "Map updated"
- Any error messages

---

## Common Issues & Solutions

### Issue 1: Grid/2D is false
**Solution**: Already fixed in code. Rebuild var_mapping:
```bash
cd ~/v4ar
colcon build --packages-select var_mapping
source install/setup.bash
```

### Issue 2: Map frame doesn't exist in TF
**Cause**: RTAB-Map only publishes map->odom transform after it creates a map
**Solution**: Wait for RTAB-Map to process sensor data. It needs:
- At least a few seconds of camera/depth data
- Or some laser scan data
- Then it will create the first map node and publish the transform

### Issue 3: No sensor data
**Solution**: 
- Make sure rosbag is playing: `ros2 bag play <bag_file> --clock`
- Check topic names match what RTAB-Map expects
- Check if topics are publishing: `ros2 topic hz <topic_name>`

### Issue 4: Bridge not receiving map
**Check**:
```bash
# Check bridge is running
ros2 node list | grep bridge

# Check bridge logs for errors
# Look for: "Subscribed to /rtabmap/grid_map"
# Look for: "Published occupancy grid: ..."
```

---

## Quick Diagnostic Script

Run this script to check everything at once:
```bash
cd ~/v4ar
source install/setup.bash
ros2 run navigation_online debug_mapping.sh
```

Or run commands manually:
```bash
# 1. Check Grid/2D
ros2 param get /rtabmap Grid/2D

# 2. Check topics
ros2 topic list | grep -E "(rtabmap|map)"

# 3. Check TF
timeout 2 ros2 run tf2_ros tf2_echo map odom 2>&1 | head -5

# 4. Check sensor data
ros2 topic hz /odom
```

---

## Expected Behavior

**When everything works:**
1. RTAB-Map starts and subscribes to sensor topics
2. After processing sensor data, RTAB-Map creates first map node
3. RTAB-Map publishes `/rtabmap/grid_map` (2D occupancy grid)
4. RTAB-Map publishes `map->odom` transform in TF
5. Bridge receives `/rtabmap/grid_map` and publishes `/map`
6. RViz can display the map (frame exists, map topic has data)
7. Navigator can plan paths using the map

**Timeline**: This usually takes 5-30 seconds after rosbag starts playing, depending on sensor data rate.

