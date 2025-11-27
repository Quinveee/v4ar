# A2 Setup and Execution Guide

## Camera Configuration

Configure camera parameters using v4l2-ctl:

```bash
v4l2-ctl -d /dev/video0 -c auto_exposure=1
v4l2-ctl -d /dev/video0 -c exposure_time_absolute=100
v4l2-ctl -d /dev/video0 -c brightness=10
```

### View all available camera controls:
```bash
v4l2-ctl -d /dev/video0 --list-ctrls
```

---

## Mono Camera Pipeline

### Start image streaming (with adjustable frame rate for efficiency):
```bash
ros2 run image_tools img --custom_rect --topic "/image_raw" --frame_rate 10
```

### Run AprilTag detection with multiscale support:
```bash
ros2 run perception apriltag_vis_node --ros-args -p enable_gui:=false -p enable_multiscale:=true -p multiscale_scales:="1.0,1.5,2.5,3.5"
```

### Run triangulation (single camera):
```bash
ros2 run perception triangulation --topic '/detected_markers'
```

### Run field visualization:
```bash
ros2 run visualizations field
```

---

## OAK-D Camera Pipeline

### Start OAK stereo camera:
```bash
ros2 launch depthai_examples stereo_inertial_node.launch.py
```

### Run OAK AprilTag detection with buffering:
```bash
ros2 run perception oak_apriltag --no_gui --enable_buffer
```

### Run dual-camera fusion triangulation:
```bash
ros2 run perception oak_mono_triangulation
```

---

## Topic Mapping

| Node | Publishes | Subscribes |
|------|-----------|-----------|
| `apriltag_vis_node` | `/detected_markers` | `/image_raw` |
| `oak_apriltag` | `/oak/detected_markers` | OAK camera topics |
| `triangulation` | `/robot_pose_raw` | `/detected_markers` |
| `oak_mono_triangulation` | `/robot_pose_raw` | `/detected_markers`, `/oak/detected_markers` |
| `localization` | `/robot_pose` | `/robot_pose_raw`, `/cmd_vel` |
| `field_visualization` | N/A | `/robot_pose`, `/detected_markers` |

---

## Quick Start Examples

### **Option 1: Mono Camera Only**
```bash
# Terminal 1: Camera
ros2 run image_tools img --custom_rect --topic "/image_raw" --frame_rate 10

# Terminal 2: Detection
ros2 run perception apriltag_vis_node --ros-args -p enable_gui:=false -p enable_multiscale:=true

# Terminal 3: Triangulation
ros2 run perception triangulation --topic '/detected_markers'

# Terminal 4: Localization
ros2 run perception localization

# Terminal 5: Visualization
ros2 run visualizations field
```

### **Option 2: OAK Camera Only**
```bash
# Terminal 1: OAK Camera
ros2 launch depthai_examples stereo_inertial_node.launch.py

# Terminal 2: Detection
ros2 run perception oak_apriltag --no_gui --enable_buffer

# Terminal 3: Triangulation:
ros2 run perception triangulation --topic '/oak/detected_markers'

# Terminal 4: Localization (optional, for filtered pose)
ros2 run perception localization

# Terminal 5: Visualization
ros2 run visualizations field
```

### **Option 3: Dual Camera Fusion (Mono + OAK)**
```bash
# Terminal 1: Mono Camera
ros2 run image_tools img --custom_rect --topic "/image_raw" --frame_rate 10

# Terminal 2: Mono Detection
ros2 run perception apriltag_vis_node --ros-args -p enable_gui:=false -p enable_multiscale:=true

# Terminal 3: OAK Camera
ros2 launch depthai_examples stereo_inertial_node.launch.py

# Terminal 4: OAK Detection
ros2 run perception oak_apriltag --no_gui --enable_buffer

# Terminal 5: Dual-camera Fusion Triangulation
ros2 run perception oak_mono_triangulation

# Terminal 6: Localization (optional)
ros2 run perception localization

# Terminal 7: Visualization
ros2 run visualizations field
```

---

## Tuning Parameters

### AprilTag Detection (Mono)
```bash
ros2 run perception apriltag_vis_node --ros-args \
  -p enable_gui:=false \
  -p enable_multiscale:=true \
  -p multiscale_scales:="0.7,1.0,1.5,2.0" \
  -p enable_superres:=true \
  -p superres_scale:=4
```

### OAK Detection
```bash
ros2 run perception oak_apriltag \
  --no_gui \
  --enable_buffer \
  --enable_low_res_tracking
```

### Dual-camera Fusion Weights
```bash
ros2 run perception oak_mono_triangulation \
  --ros-args \
  --oak_weight 2.0 \
  --mono_weight 1.0
```

---

## Debugging

### Check active topics:
```bash
ros2 topic list
```

### Monitor marker detections:
```bash
ros2 topic echo /detected_markers
ros2 topic echo /oak/detected_markers
```

### Monitor robot pose:
```bash
ros2 topic echo /robot_pose_raw
ros2 topic echo /robot_pose
```

### View node graphs:
```bash
rqt_graph
```
