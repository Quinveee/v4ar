ros2 run perception apriltag_vis_node --ros-args -p enable_gui:=false -p enable_multiscale:=true -p multiscale_scales:="1.0,1.5,2.5,3.5" &
ros2 run perception triangulation &
ros2 run perception localization --ros-args -p strategy_type:=kalman &
ros2 run visualizations field