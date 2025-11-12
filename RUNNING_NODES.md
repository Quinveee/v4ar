# Running the Nodes

After building with `colcon build` and sourcing with `source install/setup.bash`, you can run the nodes as follows:

## Individual Nodes

### 1. Image Subscriber (image_tools)

Subscribes to camera feed and displays processed images.

```bash
ros2 run image_tools img_subscriber_uni
```

Or via launch file:

```bash
ros2 launch image_tools image_subscriber.launch.py
```

### 2. Line Detector (perception)

Detects lines in camera feed using various algorithms.

```bash
ros2 run perception line_detector
```

With custom detector type:

```bash
ros2 run perception line_detector --ros-args -p detector_type:=custom
```

Available detectors: `custom`, `canny`, `brightness`, `gradient`, `skeleton`

Or via launch file:

```bash
ros2 launch perception image_detection.launch.py detector_type:=custom
```

### 3. Line Follower (control/line_follower)

Controls robot movement to follow detected lines.

```bash
ros2 run line_follower line_follower
```

With custom parameters:

```bash
ros2 run line_follower line_follower --ros-args \
  -p speed_control:=angle_based \
  -p selector:=confidence \
  -p forward_speed:=0.25
```

Or via launch file:

```bash
ros2 launch line_follower line_follower.launch.py speed_control:=angle_based
```

## Complete System (Recommended)

Run all nodes together with the main launch file:

```bash
ros2 launch line_follower line_following.launch.py
```

With custom parameters:

```bash
ros2 launch line_follower line_following.launch.py \
  detector_type:=custom \
  speed_control:=angle_based \
  selector:=confidence \
  forward_speed:=0.25 \
  enable_visualization:=true
```

## Running in Separate Terminals

Terminal 1 - Image Subscriber:
```bash
ros2 run image_tools img_subscriber_uni
```

Terminal 2 - Line Detector:
```bash
ros2 run perception line_detector --ros-args -p detector_type:=custom
```

Terminal 3 - Line Follower:
```bash
ros2 run line_follower line_follower --ros-args -p speed_control:=angle_based
```

Terminal 4 (Optional) - Visualization:
```bash
ros2 run visualizations visualization_node
```

