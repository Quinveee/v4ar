#!/usr/bin/env python3
"""
Nav2 bringup launch file for UGV rover.

This launch file starts all Nav2 nodes needed for autonomous navigation:
- Map server (loads your saved map)
- AMCL (localization using laser scan)
- Controller server (local planner)
- Planner server (global planner)
- Behavior server (recovery behaviors)
- BT Navigator (behavior tree coordinator)
- Waypoint follower
- Lifecycle manager (manages all node lifecycles)

Usage:
    ros2 launch navigation nav2_rover_bringup.launch.py

    # With custom params file:
    ros2 launch navigation nav2_rover_bringup.launch.py params_file:=/path/to/params.yaml

    # With custom map:
    ros2 launch navigation nav2_rover_bringup.launch.py map:=/path/to/map.yaml

Prerequisites:
    - Robot must be publishing /odom and TF (odom -> base_footprint)
    - LiDAR must be publishing to /scan
    - Robot must accept velocity commands on /cmd_vel
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for Nav2 stack."""
    
    # Get package directory
    pkg_navigation = FindPackageShare('navigation')
    
    # Launch configuration variables
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    
    # Declare launch arguments
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([pkg_navigation, 'config', 'nav2_params.yaml']),
        description='Full path to the ROS2 parameters file to use for all launched nodes')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')
    
    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack')
    
    # Map server node
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )
    
    # AMCL (localization) node
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )
    
    # Controller server (local planner)
    controller_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static'),
                    ('cmd_vel', '/cmd_vel')]
    )
    
    # Planner server (global planner)
    planner_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )
    
    # Behavior server (recovery behaviors)
    behavior_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static'),
                    ('cmd_vel', '/cmd_vel')]
    )
    
    # BT Navigator (behavior tree coordinator)
    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )
    
    # Waypoint follower
    waypoint_follower_node = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )
    
    # Lifecycle manager - manages all Nav2 node lifecycles
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[params_file,
                    {'use_sim_time': use_sim_time},
                    {'autostart': autostart}]
    )
    
    # Create launch description and populate
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_autostart_cmd)
    
    # Add all Nav2 nodes
    ld.add_action(map_server_node)
    ld.add_action(amcl_node)
    ld.add_action(controller_node)
    ld.add_action(planner_node)
    ld.add_action(behavior_node)
    ld.add_action(bt_navigator_node)
    ld.add_action(waypoint_follower_node)
    ld.add_action(lifecycle_manager_node)
    
    return ld
