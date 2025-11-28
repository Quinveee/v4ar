ros2 launch depthai_examples stereo_inertial_node.launch.py &
ros2 run perception oak_apriltag --no_gui --enable_buffer &
ros2 run perception triangulation --topic '/oak/detected_markers' &
ros2 run perception localization &
ros2 run visualizations field