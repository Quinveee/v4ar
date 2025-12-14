#!/usr/bin/env python3
"""
Laptop Side Launch File for Online Navigation

This launch file starts all nodes that run on the laptop:
1. RTAB-Map SLAM (mapping)
2. RTAB-Map bridge (converts to occupancy grid)
3. Online navigator (plans paths and generates commands)

Usage:
    # On laptop:
    ros2 launch navigation_online laptop_side.launch.py
    
    # With custom parameters:
    ros2 launch navigation_online laptop_side.launch.py use_rviz:=true use_sim_time:=true

Prerequisites:
    - Robot sensors publishing to topics (camera, depth, odometry)
    - Robot odometry available on /odom topic
    - Goal poses can be published to /goal topic
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for laptop-side online navigation."""
    
    # Get package directories
    var_mapping_dir = get_package_share_directory('var_mapping')
    navigation_online_dir = get_package_share_directory('navigation_online')
    
    # Launch arguments
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz2 for visualization'
    )
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    
    declare_depth_filter_level = DeclareLaunchArgument(
        'depth_filter_level',
        default_value='-1',
        description='Depth filter level: -1 (no filter), 0-3 (filter level)'
    )
    
    declare_subscribe_scan = DeclareLaunchArgument(
        'subscribe_scan',
        default_value='false',
        description='Subscribe to laser scan topic (/scan) for RTAB-Map'
    )

    declare_clean_map = DeclareLaunchArgument(
        'clean_map',
        default_value='false',
        description='Enable map noise cleaning (removes small obstacles, closes gaps, prunes spikes)'
    )

    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    depth_filter_level = LaunchConfiguration('depth_filter_level')
    subscribe_scan = LaunchConfiguration('subscribe_scan')
    clean_map = LaunchConfiguration('clean_map')
    
    # 1. RTAB-Map SLAM (from var_mapping package)
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(var_mapping_dir, 'launch', 'map_generator.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_rviz': use_rviz,
            'depth_filter_level': depth_filter_level,
            'subscribe_scan': subscribe_scan,
            'is_online': 'true',  # Use online navigation RViz config
        }.items()
    )
    
    # 2. RTAB-Map Bridge (converts RTAB-Map output to occupancy grid)
    rtabmap_bridge_node = Node(
        package='navigation_online',
        executable='rtabmap_bridge',
        name='rtabmap_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'map_frame_id': 'map',
            'use_rtabmap_grid': True,  # Use RTAB-Map's grid_map if available
        }]
    )
    
    # 3. Nav2 Planner Server (for path planning)
    # Note: Planner server creates its own global_costmap internally
    navigation_dir = get_package_share_directory('navigation')
    nav2_params_file = os.path.join(navigation_dir, 'config', 'nav2_params.yaml')
    
    planner_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params_file, {'use_sim_time': use_sim_time}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )
    
    # 4. Lifecycle Manager (required to activate Nav2 lifecycle nodes)
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['planner_server']
        }]
    )
    
    # 5. Online Navigator (uses Nav2 planner and generates commands)
    online_navigator_node = Node(
        package='navigation_online',
        executable='online_navigator',
        name='online_navigator',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'planner_server': '',  # Empty string = use /compute_path_to_pose (Nav2 default action name)
            'k_linear': 0.5,
            'k_angular': 2.0,
            'max_linear_speed': 0.3,
            'max_angular_speed': 1.0,
            'waypoint_threshold': 0.15,
            'goal_threshold': 0.1,
            'control_frequency': 10.0,
            # Map cleaning parameters
            'clean_map': clean_map,
            'min_blob_size': 30,
            'connect_gap_size': 6,
            'prune_size': 3,
            'prune_iters': 1,
        }]
    )
    
    return LaunchDescription([
        # Launch arguments
        declare_use_rviz,
        declare_use_sim_time,
        declare_depth_filter_level,
        declare_subscribe_scan,
        declare_clean_map,

        # Nodes
        rtabmap_launch,
        rtabmap_bridge_node,
        planner_node,
        lifecycle_manager_node,
        online_navigator_node,
    ])

