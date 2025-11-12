# Build Fixes Applied

## Issues Fixed

### 1. Resource Directory Names
Fixed resource directory names to match package names:

```bash
# image_tools
src/image_tools/resource/opencv_tools → src/image_tools/resource/image_tools

# perception
src/perception/resource/line_detector → src/perception/resource/perception

# control (created)
mkdir -p src/control/resource
touch src/control/resource/control
```

### 2. image_tools setup.py
Updated to exclude line_msgs from Python packages (it's a CMake package):

```python
packages=find_packages(exclude=['test', 'line_msgs']),
```

## Build Command

On the rover or any system with ROS2 installed:

```bash
cd /home/ws/ugv_ws/v4ar
colcon build
source install/setup.bash
```

## Expected Build Output

```
Starting >>> line_msgs
Starting >>> control
Starting >>> image_tools
Starting >>> line_detection
Finished <<< line_msgs [1.85s]
Finished <<< control [2.69s]
Finished <<< image_tools [3.17s]
Finished <<< line_detection [3.29s]
Starting >>> perception
Finished <<< perception [1.63s]
Starting >>> visualizations
Finished <<< visualizations [2.50s]
Starting >>> line_follower
Finished <<< line_follower [2.30s]

Summary: 7 packages finished [15.00s]
```

## If Build Still Fails

Check for:
1. Missing ROS2 dependencies: `sudo apt install ros-humble-*` (or your distro)
2. Python package issues: `pip install -r requirements.txt` (if exists)
3. CMake issues: Ensure CMake 3.5+ is installed

## After Successful Build

```bash
source install/setup.bash
ros2 launch line_follower line_following.launch.py
```

