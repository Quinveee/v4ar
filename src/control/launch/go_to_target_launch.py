#!/usr/bin/env python3
"""
Launch file for go_to_target_strategy control node and visualization.

Launches:
- Control node (control_node) with go_to_target_strategy
- Control visualization node (optional)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for go_to_target_strategy."""
    
    # Declare launch arguments with defaults
    start_x = LaunchConfiguration('start_x', default='2.0')
    start_y = LaunchConfiguration('start_y', default='1.0')
    start_theta = LaunchConfiguration('start_theta', default='0.0')
    target_x = LaunchConfiguration('target_x', default='4.0')
    target_y = LaunchConfiguration('target_y', default='1.0')
    use_odometry_update = LaunchConfiguration('use_odometry_update', default='true')
    pose_topic = LaunchConfiguration('pose_topic', default='/odom_pose_processed')
    launch_visualization = LaunchConfiguration('launch_visualization', default='true')
    
    # Control node
    control_node = Node(
        package='control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{
            'start_x': start_x,
            'start_y': start_y,
            'start_theta': start_theta,
            'target_x': target_x,
            'target_y': target_y,
            'use_odometry_update': use_odometry_update,
            'pose_topic': pose_topic,
        }]
    )
    
    # Control visualization node (optional)
    control_visualization = Node(
        package='visualizations',
        executable='control_visualization',
        name='control_visualization',
        output='screen',
        parameters=[{
            'start_x': start_x,
            'start_y': start_y,
            'start_theta': start_theta,
            'target_x': target_x,
            'target_y': target_y,
        }],
        condition=IfCondition(launch_visualization)
    )
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument('start_x', default_value='2.0',
                             description='Start x position (meters)'),
        DeclareLaunchArgument('start_y', default_value='1.0',
                             description='Start y position (meters)'),
        DeclareLaunchArgument('start_theta', default_value='0.0',
                             description='Start orientation (radians)'),
        DeclareLaunchArgument('target_x', default_value='4.0',
                             description='Target x position (meters)'),
        DeclareLaunchArgument('target_y', default_value='1.0',
                             description='Target y position (meters)'),
        DeclareLaunchArgument('use_odometry_update', default_value='true',
                             description='Use odometry updates for position estimation'),
        DeclareLaunchArgument('pose_topic', default_value='/odom_pose_processed',
                             description='Topic name for pose updates'),
        DeclareLaunchArgument('launch_visualization', default_value='true',
                             description='Launch visualization node'),
        
        # Nodes
        control_node,
        control_visualization,
    ])

