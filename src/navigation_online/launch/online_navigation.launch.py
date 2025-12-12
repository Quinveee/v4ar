#!/usr/bin/env python3
"""
Complete Online Navigation Launch File

This launch file can be used to start both laptop and rover sides together
if running on the same machine (for testing).

For production use:
    - Run laptop_side.launch.py on laptop
    - Run rover_side.launch.py on rover

Usage:
    # For testing (both sides on same machine):
    ros2 launch navigation_online online_navigation.launch.py
    
    # Specify which side to run:
    ros2 launch navigation_online online_navigation.launch.py side:=laptop
    ros2 launch navigation_online online_navigation.launch.py side:=rover
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for complete online navigation system."""
    
    navigation_online_dir = get_package_share_directory('navigation_online')
    
    # Launch arguments
    declare_side = DeclareLaunchArgument(
        'side',
        default_value='both',
        description='Which side to run: laptop, rover, or both'
    )
    
    side = LaunchConfiguration('side')
    
    # Laptop side launch
    laptop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_online_dir, 'launch', 'laptop_side.launch.py')
        )
    )
    
    # Rover side launch
    rover_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_online_dir, 'launch', 'rover_side.launch.py')
        )
    )
    
    return LaunchDescription([
        declare_side,
        laptop_launch,
        rover_launch,
    ])

