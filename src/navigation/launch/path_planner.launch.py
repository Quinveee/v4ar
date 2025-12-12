#!/usr/bin/env python3
"""
Simple navigation launch - just point and go!

Usage:
    # Navigate to a specific point:
    ros2 launch navigation simple_nav.launch.py goal_x:=2.0 goal_y:=3.0
    
    # With custom speed and obstacle avoidance:
    ros2 launch navigation simple_nav.launch.py goal_x:=2.0 goal_y:=3.0 max_speed:=0.2 obstacle_distance:=0.7

Prerequisites:
    - Localization running (publishing /robot_pose)
    - Optionally: obstacle detection (publishing /detected_rovers)
    - Robot accepting /cmd_vel commands
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for simple navigation."""
    
    # Launch arguments
    declare_goal_x = DeclareLaunchArgument(
        'goal_x', default_value='2.0',
        description='Goal X coordinate in map frame (meters)')
    
    declare_goal_y = DeclareLaunchArgument(
        'goal_y', default_value='3.0',
        description='Goal Y coordinate in map frame (meters)')
    
    declare_goal_tolerance = DeclareLaunchArgument(
        'goal_tolerance', default_value='0.15',
        description='Distance threshold to consider goal reached (meters)')
    
    declare_max_speed = DeclareLaunchArgument(
        'max_linear_speed', default_value='0.3',
        description='Maximum linear velocity (m/s)')
    
    declare_max_angular = DeclareLaunchArgument(
        'max_angular_speed', default_value='1.0',
        description='Maximum angular velocity (rad/s)')
    
    declare_obstacle_dist = DeclareLaunchArgument(
        'obstacle_distance', default_value='0.5',
        description='Stop distance from obstacles (meters)')
    
    # Simple navigator node
    navigator_node = Node(
        package='navigation',
        executable='simple_navigator',
        name='simple_navigator',
        output='screen',
        parameters=[{
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'goal_tolerance': LaunchConfiguration('goal_tolerance'),
            'max_linear_speed': LaunchConfiguration('max_linear_speed'),
            'max_angular_speed': LaunchConfiguration('max_angular_speed'),
            'obstacle_distance': LaunchConfiguration('obstacle_distance'),
        }]
    )
    
    return LaunchDescription([
        declare_goal_x,
        declare_goal_y,
        declare_goal_tolerance,
        declare_max_speed,
        declare_max_angular,
        declare_obstacle_dist,
        navigator_node,
    ])
