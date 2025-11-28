#!/usr/bin/env python3
"""
Rover Detector Launch File

Launches the rover detector node and optionally the visualization node.

Usage:
  ros2 launch perception rover_detector.launch.py
  ros2 launch perception rover_detector.launch.py enable_viz:=true no_gui:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    low_res_fps_arg = DeclareLaunchArgument(
        'low_res_fps',
        default_value='15.0',
        description='Low resolution detection frequency (Hz)'
    )
    
    high_res_fps_arg = DeclareLaunchArgument(
        'high_res_fps',
        default_value='5.0',
        description='High resolution detection frequency (Hz)'
    )
    
    max_distance_arg = DeclareLaunchArgument(
        'max_distance',
        default_value='4.0',
        description='Maximum detection distance (meters)'
    )
    
    min_rover_area_arg = DeclareLaunchArgument(
        'min_rover_area',
        default_value='200',
        description='Minimum rover area in pixels for detection'
    )
    
    depth_roi_size_arg = DeclareLaunchArgument(
        'depth_roi_size',
        default_value='10',
        description='Depth ROI size around center (pixels)'
    )
    
    black_threshold_arg = DeclareLaunchArgument(
        'black_threshold',
        default_value='60',
        description='HSV value threshold for black detection (0-255)'
    )
    
    enable_buffer_arg = DeclareLaunchArgument(
        'enable_buffer',
        default_value='true',
        description='Enable temporal buffering for smoother tracking'
    )
    
    buffer_alpha_arg = DeclareLaunchArgument(
        'buffer_alpha',
        default_value='0.7',
        description='Buffer blending factor (0-1, higher = more weight to new data)'
    )
    
    no_gui_arg = DeclareLaunchArgument(
        'no_gui',
        default_value='false',
        description='Disable detector GUI window'
    )
    
    depth_method_arg = DeclareLaunchArgument(
        'depth_method',
        default_value='closest_average',
        choices=['median', 'closest_average', 'mean'],
        description='Depth calculation method'
    )
    
    closest_percentile_arg = DeclareLaunchArgument(
        'closest_percentile',
        default_value='0.1',
        description='Percentile of closest points to use for averaging (0-1)'
    )
    
    rover_length_arg = DeclareLaunchArgument(
        'rover_length_m',
        default_value='0.25',
        description='Known rover length in meters (25cm)'
    )
    
    rover_width_arg = DeclareLaunchArgument(
        'rover_width_m',
        default_value='0.25',
        description='Known rover width in meters (25cm)'
    )
    
    rover_height_arg = DeclareLaunchArgument(
        'rover_height_m',
        default_value='0.25',
        description='Known rover height in meters (25cm)'
    )
    
    size_filter_enabled_arg = DeclareLaunchArgument(
        'size_filter_enabled',
        default_value='true',
        description='Enable size-based filtering to reduce false positives'
    )
    
    min_size_ratio_arg = DeclareLaunchArgument(
        'min_size_ratio',
        default_value='0.5',
        description='Minimum size relative to expected rover (0.5 = 50% of expected size)'
    )
    
    max_size_ratio_arg = DeclareLaunchArgument(
        'max_size_ratio',
        default_value='2.0',
        description='Maximum size relative to expected rover (2.0 = 200% of expected size)'
    )
    
    aspect_ratio_tolerance_arg = DeclareLaunchArgument(
        'aspect_ratio_tolerance',
        default_value='0.5',
        description='Aspect ratio tolerance (0.5 = 50% deviation allowed)'
    )
    
    enable_viz_arg = DeclareLaunchArgument(
        'enable_viz',
        default_value='true',
        description='Enable visualization node'
    )
    
    # Detector node
    detector_node = Node(
        package='perception',
        executable='rover_detector_with_pose',
        name='rover_detector',
        output='screen',
        parameters=[{
            'low_res_fps': LaunchConfiguration('low_res_fps'),
            'high_res_fps': LaunchConfiguration('high_res_fps'),
            'max_distance': LaunchConfiguration('max_distance'),
            'min_rover_area': LaunchConfiguration('min_rover_area'),
            'depth_roi_size': LaunchConfiguration('depth_roi_size'),
            'black_threshold': LaunchConfiguration('black_threshold'),
            'enable_buffer': LaunchConfiguration('enable_buffer'),
            'buffer_alpha': LaunchConfiguration('buffer_alpha'),
            'no_gui': LaunchConfiguration('no_gui'),
            'depth_method': LaunchConfiguration('depth_method'),
            'closest_percentile': LaunchConfiguration('closest_percentile'),
            'rover_length_m': LaunchConfiguration('rover_length_m'),
            'rover_width_m': LaunchConfiguration('rover_width_m'),
            'rover_height_m': LaunchConfiguration('rover_height_m'),
            'size_filter_enabled': LaunchConfiguration('size_filter_enabled'),
            'min_size_ratio': LaunchConfiguration('min_size_ratio'),
            'max_size_ratio': LaunchConfiguration('max_size_ratio'),
            'aspect_ratio_tolerance': LaunchConfiguration('aspect_ratio_tolerance'),
        }]
    )
    
    # Visualization node (optional)
    viz_node = Node(
        package='visualizations',
        executable='rover_detection_visualization_node',
        name='rover_detection_viz',
        output='screen',
        parameters=[{
            'show_robot_pose': True,  # Boolean, not string
            'rover_color_bgr': [0, 0, 255],  # Red
            'rover_size_m': LaunchConfiguration('rover_length_m'),
        }],
        condition=IfCondition(LaunchConfiguration('enable_viz'))
    )
    
    return LaunchDescription([
        low_res_fps_arg,
        high_res_fps_arg,
        max_distance_arg,
        min_rover_area_arg,
        depth_roi_size_arg,
        black_threshold_arg,
        enable_buffer_arg,
        buffer_alpha_arg,
        no_gui_arg,
        depth_method_arg,
        closest_percentile_arg,
        rover_length_arg,
        rover_width_arg,
        rover_height_arg,
        size_filter_enabled_arg,
        min_size_ratio_arg,
        max_size_ratio_arg,
        aspect_ratio_tolerance_arg,
        enable_viz_arg,
        detector_node,
        viz_node,
    ])

