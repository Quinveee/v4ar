#!/usr/bin/env python3
"""
Launch file for AprilTag marker visualization on the UGV.

This launch file:
  - Starts ugv_bringup (robot bringup)
  - Starts ugv_driver (hardware driver)
  - Starts the camera pipeline (ugv_vision)
  - Starts your AprilTag visualization node subscribing to /image_rect
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # --------- Launch arguments for your AprilTag node ---------
    tag_family_arg = DeclareLaunchArgument(
        'tag_family',
        default_value='tagStandard41h12',
        description='AprilTag family'
    )

    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='apriltag_screenshots',
        description='Directory for saving screenshots'
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/image_rect',
        description='Image topic to subscribe to'
    )

    # --------- 1) Bringup and driver (equivalent to ros2 run ... & ) ---------
    ugv_bringup_proc = ExecuteProcess(
        cmd=['ros2', 'run', 'ugv_bringup', 'ugv_bringup'],
        name='ugv_bringup',
        output='screen'
    )

    ugv_driver_proc = ExecuteProcess(
        cmd=['ros2', 'run', 'ugv_bringup', 'ugv_driver'],
        name='ugv_driver',
        output='screen'
    )

    # --------- 2) Camera pipeline (equivalent to ros2 launch ugv_vision camera.launch.py) ---------
    camera_proc = ExecuteProcess(
        cmd=['ros2', 'launch', 'ugv_vision', 'camera.launch.py'],
        name='ugv_camera',
        output='screen'
    )

    # --------- 3) Your AprilTag visualization node ---------
    # Using Node action instead of ExecuteProcess - this is the ROS2 way!
    apriltag_node = Node(
        package='perception',
        executable='apriltag_vis_node_2',  # This should match the entry point in setup.py
        name='apriltag_visualization',
        output='screen',
        parameters=[{
            'tag_family': LaunchConfiguration('tag_family'),
            'output_dir': LaunchConfiguration('output_dir'),
            'image_topic': LaunchConfiguration('image_topic'),
        }]
    )

    return LaunchDescription([
        tag_family_arg,
        output_dir_arg,
        image_topic_arg,
        ugv_bringup_proc,
        ugv_driver_proc,
        camera_proc,
        apriltag_node,
    ])