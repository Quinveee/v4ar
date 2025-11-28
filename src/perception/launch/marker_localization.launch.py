#!/usr/bin/env python3
"""
Complete marker-based localization launch file.

This launch file starts:
  - AprilTag marker detection
  - Triangulation node
  - Localization node (with selectable strategy)

Usage:
  ros2 launch perception marker_localization.launch.py
  ros2 launch perception marker_localization.launch.py detector:=oak_apriltag strategy_type:=adaptive_kalman
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get config path
    config_path = os.path.join(
        get_package_share_directory('perception'),
        'config',
        'markers.yaml'
    )

    # Launch arguments
    detector_arg = DeclareLaunchArgument(
        'detector',
        default_value='apriltag',
        choices=['apriltag', 'oak_apriltag'],
        description='Marker detector type'
    )

    strategy_type_arg = DeclareLaunchArgument(
        'strategy_type',
        default_value='adaptive_kalman',
        choices=['identity', 'kalman', 'adaptive_kalman', 'particle_filter', 'sliding_window', 'complementary', 'robust'],
        description='Localization strategy'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='20.0',
        description='Localization publish rate (Hz)'
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/image_rect',
        description='Input image topic'
    )

    enable_gui_arg = DeclareLaunchArgument(
        'enable_gui',
        default_value='false',
        description='Enable marker detection GUI'
    )

    # Marker detection node (conditional based on detector type)
    apriltag_node = Node(
        package='perception',
        executable='apriltag',
        name='apriltag_detector',
        output='screen',
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'enable_gui': LaunchConfiguration('enable_gui'),
        }],
        condition=lambda context: context.launch_configurations['detector'] == 'apriltag'
    )

    oak_apriltag_node = Node(
        package='perception',
        executable='oak_apriltag',
        name='oak_apriltag_detector',
        output='screen',
        parameters=[{
            'enable_gui': LaunchConfiguration('enable_gui'),
        }],
        condition=lambda context: context.launch_configurations['detector'] == 'oak_apriltag'
    )

    # Triangulation node
    triangulation_node = Node(
        package='perception',
        executable='triangulation',
        name='triangulation_node',
        output='screen',
        parameters=[{
            'marker_config': config_path,
            'solver_type': 'least_squares',
        }]
    )

    # Localization node
    localization_node = Node(
        package='perception',
        executable='localization',
        name='localization_node',
        output='screen',
        parameters=[{
            'strategy_type': LaunchConfiguration('strategy_type'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }]
    )

    return LaunchDescription([
        detector_arg,
        strategy_type_arg,
        publish_rate_arg,
        image_topic_arg,
        enable_gui_arg,
        apriltag_node,
        oak_apriltag_node,
        triangulation_node,
        localization_node,
    ])