#!/usr/bin/env python3
"""
Launch file for Dijkstra-based navigation system.

This launch file starts:
    1. Map server (publishes static map from PGM+YAML files)
    2. Dijkstra planner (plans paths using the map)
    3. Path follower (executes the planned path)

Usage:
    # With map from topic (map server or RTAB-Map publishes /map):
    ros2 launch navigation_dijkstra dijkstra_navigation.launch.py
    
    # With map from file:
    ros2 launch navigation_dijkstra dijkstra_navigation.launch.py \\
        map_file:=/path/to/map.yaml
    
    # With custom parameters:
    ros2 launch navigation_dijkstra dijkstra_navigation.launch.py \\
        map_file:=/path/to/map.yaml \\
        k_linear:=0.6 \\
        k_angular:=2.5 \\
        max_linear_speed:=0.4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for Dijkstra navigation."""
    
    # Launch arguments
    declare_map_file = DeclareLaunchArgument(
        'map_file',
        default_value='',
        description='Path to map YAML file (leave empty to use /map topic)'
    )
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    
    declare_occupied_threshold = DeclareLaunchArgument(
        'occupied_threshold',
        default_value='50',
        description='Occupancy threshold for obstacles (0-100)'
    )
    
    declare_k_linear = DeclareLaunchArgument(
        'k_linear',
        default_value='0.5',
        description='Proportional gain for linear velocity'
    )
    
    declare_k_angular = DeclareLaunchArgument(
        'k_angular',
        default_value='2.0',
        description='Proportional gain for angular velocity'
    )
    
    declare_max_linear_speed = DeclareLaunchArgument(
        'max_linear_speed',
        default_value='0.3',
        description='Maximum linear velocity (m/s)'
    )
    
    declare_max_angular_speed = DeclareLaunchArgument(
        'max_angular_speed',
        default_value='1.0',
        description='Maximum angular velocity (rad/s)'
    )
    
    declare_waypoint_threshold = DeclareLaunchArgument(
        'waypoint_threshold',
        default_value='0.15',
        description='Distance to advance to next waypoint (m)'
    )
    
    declare_goal_threshold = DeclareLaunchArgument(
        'goal_threshold',
        default_value='0.1',
        description='Distance to consider goal reached (m)'
    )
    
    declare_use_map_server = DeclareLaunchArgument(
        'use_map_server',
        default_value='true',
        description='Launch map_server to publish /map topic from file'
    )
    
    # Get launch configurations
    map_file = LaunchConfiguration('map_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_map_server = LaunchConfiguration('use_map_server')
    occupied_threshold = LaunchConfiguration('occupied_threshold')
    k_linear = LaunchConfiguration('k_linear')
    k_angular = LaunchConfiguration('k_angular')
    max_linear_speed = LaunchConfiguration('max_linear_speed')
    max_angular_speed = LaunchConfiguration('max_angular_speed')
    waypoint_threshold = LaunchConfiguration('waypoint_threshold')
    goal_threshold = LaunchConfiguration('goal_threshold')
    
    # Dijkstra planner node
    dijkstra_planner_node = Node(
        package='navigation_dijkstra',
        executable='dijkstra_planner',
        name='dijkstra_planner',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'map_file': map_file,
            'occupied_threshold': occupied_threshold,
            'use_map_topic': True,  # Always subscribe to /map topic
        }]
    )
    
    # Path follower node
    path_follower_node = Node(
        package='navigation_dijkstra',
        executable='path_follower',
        name='path_follower',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'k_linear': k_linear,
            'k_angular': k_angular,
            'max_linear_speed': max_linear_speed,
            'max_angular_speed': max_angular_speed,
            'waypoint_threshold': waypoint_threshold,
            'goal_threshold': goal_threshold,
        }]
    )
    
    # Map server node (publishes /map from file)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'yaml_filename': map_file,
        }],
        condition=IfCondition(use_map_server)
    )
    
    # Lifecycle manager for map server (required for Nav2 map server)
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server']
        }],
        condition=IfCondition(use_map_server)
    )
    
    return LaunchDescription([
        # Declare arguments
        declare_map_file,
        declare_use_sim_time,
        declare_use_map_server,
        declare_occupied_threshold,
        declare_k_linear,
        declare_k_angular,
        declare_max_linear_speed,
        declare_max_angular_speed,
        declare_waypoint_threshold,
        declare_goal_threshold,
        
        # Launch nodes
        map_server_node,
        lifecycle_manager_node,
        dijkstra_planner_node,
        path_follower_node,
    ])
