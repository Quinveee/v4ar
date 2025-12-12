#!/usr/bin/env python3
"""
Rover Side Launch File for Online Navigation

This launch file starts the command executor node on the rover.
The rover listens to commands from the laptop and executes them.

Usage:
    # On rover:
    ros2 launch navigation_online rover_side.launch.py
    
    # With custom command topic:
    ros2 launch navigation_online rover_side.launch.py cmd_vel_topic:=/ugv/cmd_vel

Prerequisites:
    - Laptop must be running laptop_side.launch.py
    - Network connection between laptop and rover
    - ROS2 topics accessible across network (ROS_DOMAIN_ID must match)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for rover-side command execution."""
    
    # Launch arguments
    declare_cmd_vel_topic = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel',
        description='Topic name for rover velocity commands'
    )
    
    declare_max_linear_speed = DeclareLaunchArgument(
        'max_linear_speed',
        default_value='0.5',
        description='Maximum linear speed safety limit (m/s)'
    )
    
    declare_max_angular_speed = DeclareLaunchArgument(
        'max_angular_speed',
        default_value='1.5',
        description='Maximum angular speed safety limit (rad/s)'
    )
    
    declare_enable_safety = DeclareLaunchArgument(
        'enable_safety_limits',
        default_value='true',
        description='Enable safety speed limits'
    )
    
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    max_linear = LaunchConfiguration('max_linear_speed')
    max_angular = LaunchConfiguration('max_angular_speed')
    enable_safety = LaunchConfiguration('enable_safety_limits')
    
    # Command Executor Node
    command_executor_node = Node(
        package='navigation_online',
        executable='command_executor',
        name='command_executor',
        output='screen',
        parameters=[{
            'cmd_vel_topic': cmd_vel_topic,
            'max_linear_speed': max_linear,
            'max_angular_speed': max_angular,
            'enable_safety_limits': enable_safety,
        }]
    )
    
    return LaunchDescription([
        # Launch arguments
        declare_cmd_vel_topic,
        declare_max_linear_speed,
        declare_max_angular_speed,
        declare_enable_safety,
        
        # Node
        command_executor_node,
    ])

