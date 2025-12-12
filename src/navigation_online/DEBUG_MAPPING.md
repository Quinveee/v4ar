# Debugging Mapping Issues - Step by Step

## Problem: "Frame [map] does not exist" in RViz

This means RTAB-Map isn't publishing the map frame or the map itself.

## Step 1: Check if RTAB-Map is Running

**In a new terminal (inside Docker):**
```bash
ros2 node list | grep rtabmap
```

**Expected:** Should see `rtabmap` node

**If not found:** RTAB-Map didn't start. Check launch file logs for errors.

---

## Step 2: Check RTAB-Map Topics

**Check what topics RTAB-Map is publishing:**
```bash
ros2 topic list | grep rtabmap
```

**Expected topics:**
- `/rtabmap/grid_map` (occupancy grid - **CRITICAL**)
- `/rtabmap/mapData`
- `/rtabmap/mapGraph`
- `/rtabmap/cloud_map` (3D point cloud)

**If `/rtabmap/grid_map` is missing:** RTAB-Map isn't configured to publish 2D occupancy grids.

---

## Step 3: Check if Map is Being Published

**Check if grid_map topic has data:**
```bash
ros2 topic echo /rtabmap/grid_map --once
```

**Expected:** Should see occupancy grid message with header.frame_id = "map"

**If empty or no data:** RTAB-Map hasn't created a map yet, or it's not configured correctly.

---

## Step 4: Check TF Tree

**Check if map frame exists in TF tree:**
```bash
ros2 run tf2_ros tf2_echo map odom
```

**Expected:** Should see transform from map to odom

**If error "Frame [map] does not exist":** RTAB-Map isn't publishing the map->odom transform.

**Alternative check:**
```bash
ros2 run tf2_tools view_frames
# Then open frames.pdf to see TF tree
```

---

## Step 5: Check Bridge Node

**Check if bridge is receiving map:**
```bash
ros2 topic echo /map --once
```

**Expected:** Should see occupancy grid message

**If empty:** Bridge isn't receiving from RTAB-Map, or RTAB-Map isn't publishing.

**Check bridge logs:**
```bash
# Look for messages like:
# "Subscribed to /rtabmap/grid_map"
# "Published occupancy grid: ..."
```

---

## Step 6: Check RTAB-Map Configuration

**RTAB-Map needs these parameters to publish 2D occupancy grid:**

```yaml
Grid/FromDepth: "true"
Grid/2D: "true"        # ← CRITICAL: Enables 2D occupancy grid
Grid/CellSize: "0.05"
Grid/RangeMax: "3.0"
```

**Check RTAB-Map parameters:**
```bash
ros2 param list /rtabmap | grep Grid
```

**If Grid/2D is false or missing:** That's the problem! RTAB-Map won't publish `/rtabmap/grid_map`.

---

## Step 7: Check Sensor Data

**RTAB-Map needs sensor data to create maps:**

```bash
# Check if camera/depth topics exist
ros2 topic list | grep -E "(oak|depth|camera|rgb)"

# Check if odometry exists
ros2 topic echo /odom --once

# Check if laser scan exists (if using scan)
ros2 topic echo /scan --once
```

**If topics are empty:** The rosbag might not be playing, or topics don't match.

---

## Common Issues & Fixes

### Issue 1: RTAB-Map not publishing grid_map
**Fix:** Add `Grid/2D: "true"` to RTAB-Map parameters

### Issue 2: Map frame not in TF tree
**Fix:** RTAB-Map needs to publish map->odom transform. Check if it's in localization mode vs SLAM mode.

### Issue 3: No sensor data
**Fix:** Check rosbag is playing with `--clock` flag, or check topic names match.

### Issue 4: Bridge not receiving data
**Fix:** Check topic name - might be `/rtabmap/grid_map` vs `/rtabmap/grid_map_2d`

---

## Quick Diagnostic Commands

Run these in order to diagnose:

```bash
# 1. Check nodes
ros2 node list

# 2. Check topics
ros2 topic list

# 3. Check if grid_map exists
ros2 topic hz /rtabmap/grid_map

# 4. Check TF
ros2 run tf2_ros tf2_echo map odom

# 5. Check bridge output
ros2 topic hz /map

# 6. Check RTAB-Map params
ros2 param get /rtabmap Grid/2D
```

