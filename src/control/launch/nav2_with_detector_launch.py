#!/usr/bin/env python3
"""
Launch file for Nav2 navigation with obstacle detector.

Launches:
- Nav2 stack (bt_navigator, planner_server, controller_server, costmap_2d, lifecycle_manager)
- Obstacle detector (detector.py)
- Optional: Nav2 strategy node
- Static TF transforms for robot structure
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for Nav2 with obstacle detector."""
    
    # Get package directories
    try:
        control_package_dir = FindPackageShare('control').find('control')
    except:
        control_package_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    autostart = LaunchConfiguration('autostart', default='true')
    
    # Nav2 speed parameters
    max_vel_x = LaunchConfiguration('max_vel_x', default='0.3')
    min_vel_x = LaunchConfiguration('min_vel_x', default='0.05')
    max_vel_theta = LaunchConfiguration('max_vel_theta', default='1.0')
    min_vel_theta = LaunchConfiguration('min_vel_theta', default='0.1')
    acc_lim_x = LaunchConfiguration('acc_lim_x', default='0.5')
    acc_lim_theta = LaunchConfiguration('acc_lim_theta', default='1.0')
    
    # Obstacle detector parameters
    detector_no_gui = LaunchConfiguration('detector_no_gui', default='true')
    detector_max_distance = LaunchConfiguration('detector_max_distance', default='4.0')
    
    # Nav2 strategy parameters
    launch_nav2_strategy = LaunchConfiguration('launch_nav2_strategy', default='false')
    nav2_strategy_target_x = LaunchConfiguration('nav2_strategy_target_x', default='5.0')
    nav2_strategy_target_y = LaunchConfiguration('nav2_strategy_target_y', default='4.0')
    nav2_strategy_auto_navigate = LaunchConfiguration('nav2_strategy_auto_navigate', default='true')
    
    # Starting position parameters (for initial pose)
    start_x = LaunchConfiguration('start_x', default='0.0')
    start_y = LaunchConfiguration('start_y', default='0.0')
    start_theta = LaunchConfiguration('start_theta', default='0.0')
    
    # Odometry source selection
    odometry_source = LaunchConfiguration('odometry_source', default='rf2o')
    
    # RF2O laser odometry parameters
    rf2o_laser_scan_topic = LaunchConfiguration('rf2o_laser_scan_topic', default='/scan')
    rf2o_odom_topic = LaunchConfiguration('rf2o_odom_topic', default='/odom_rf2o')
    rf2o_base_frame = LaunchConfiguration('rf2o_base_frame', default='base_footprint')
    rf2o_odom_frame = LaunchConfiguration('rf2o_odom_frame', default='odom')
    
    # Dead reckoning odometry parameters
    dead_reckoning_odom_topic = LaunchConfiguration('dead_reckoning_odom_topic', default='/odom')
    dead_reckoning_cmd_vel_topic = LaunchConfiguration('dead_reckoning_cmd_vel_topic', default='/cmd_vel')
    dead_reckoning_base_frame = LaunchConfiguration('dead_reckoning_base_frame', default='base_footprint')
    dead_reckoning_odom_frame = LaunchConfiguration('dead_reckoning_odom_frame', default='odom')
    dead_reckoning_publish_tf = LaunchConfiguration('dead_reckoning_publish_tf', default='true')
    
    # Visualization parameters
    launch_visualization = LaunchConfiguration('launch_visualization', default='false')
    
    # Robot frame parameters
    publish_static_tf = LaunchConfiguration('publish_static_tf', default='false')
    
    # Nav2 controller parameters (FLATTENED - single level nesting)
    controller_params = {
        'use_sim_time': use_sim_time,
        'controller_frequency': 20.0,
        'min_x_velocity_threshold': 0.001,
        'min_y_velocity_threshold': 0.5,
        'min_theta_velocity_threshold': 0.001,
        'failure_tolerance': 0.3,
        'progress_checker_plugin': 'progress_checker',
        'goal_checker_plugins': ['general_goal_checker'],
        'controller_plugins': ['FollowPath'],
        'general_goal_checker': {
            'plugin': 'nav2_controller::SimpleGoalChecker',
            'xy_goal_tolerance': 0.25,
            'yaw_goal_tolerance': 0.25,
            'stateful': True,
        },
        'progress_checker': {
            'plugin': 'nav2_controller::SimpleProgressChecker',
            'required_movement_radius': 0.5,
            'movement_time_allowance': 10.0,
        },
        'FollowPath': {
            'plugin': 'dwb_core::DWBLocalPlanner',
            'debug_trajectory_details': False,
            'min_vel_x': min_vel_x,
            'min_vel_y': 0.0,
            'max_vel_x': max_vel_x,
            'max_vel_y': 0.0,
            'max_vel_theta': max_vel_theta,
            'min_speed_xy': min_vel_x,
            'max_speed_xy': max_vel_x,
            'min_speed_theta': min_vel_theta,
            'acc_lim_x': acc_lim_x,
            'acc_lim_y': 0.0,
            'acc_lim_theta': acc_lim_theta,
            'decel_lim_x': acc_lim_x,
            'decel_lim_y': 0.0,
            'decel_lim_theta': acc_lim_theta,
            'vx_samples': 20,
            'vy_samples': 0,
            'vtheta_samples': 40,
            'sim_time': 1.7,
            'linear_granularity': 0.05,
            'angular_granularity': 0.025,
            'transform_tolerance': 0.2,
            'xy_goal_tolerance': 0.25,
            'trans_stopped_velocity': 0.25,
            'short_circuit_trajectory_evaluation': True,
            'stateful': True,
            'critics': ['RotateToGoal', 'Oscillation', 'BaseObstacle', 'GoalAlign', 'PathAlign', 'PathDist', 'GoalDist'],
            'BaseObstacle.scale': 0.02,
            'PathAlign.scale': 32.0,
            'PathAlign.forward_point_distance': 0.1,
            'GoalAlign.scale': 24.0,
            'GoalAlign.forward_point_distance': 0.1,
            'PathDist.scale': 32.0,
            'GoalDist.scale': 24.0,
            'RotateToGoal.scale': 32.0,
            'RotateToGoal.slowing_factor': 5.0,
            'RotateToGoal.lookahead_time': -1.0,
        },
    }
    
    # Nav2 planner parameters (FLATTENED)
    planner_params = {
        'use_sim_time': use_sim_time,
        'expected_planner_frequency': 20.0,
        'planner_plugins': ['GridBased'],
        'GridBased': {
            'plugin': 'nav2_navfn_planner/NavfnPlanner',
            'tolerance': 0.5,
            'use_astar': False,
            'allow_unknown': True,
        },
    }
    
    # Nav2 BT Navigator parameters (FLATTENED)
    bt_navigator_params = {
        'use_sim_time': use_sim_time,
        'global_frame': 'map',
        'robot_base_frame': 'base_footprint',
        'odom_topic': '/odom',
        'bt_loop_duration': 10,
        'default_server_timeout': 20,
        'enable_groot_monitoring': False,
        'groot_zmq_publisher_port': 1666,
        'groot_zmq_server_port': 1667,
        'plugin_lib_names': [
            'nav2_compute_path_to_pose_action_bt_node',
            'nav2_compute_path_through_poses_action_bt_node',
            'nav2_smooth_path_action_bt_node',
            'nav2_follow_path_action_bt_node',
            'nav2_spin_action_bt_node',
            'nav2_wait_action_bt_node',
            'nav2_back_up_action_bt_node',
            'nav2_drive_on_heading_bt_node',
            'nav2_clear_costmap_service_bt_node',
            'nav2_is_stuck_condition_bt_node',
            'nav2_goal_reached_condition_bt_node',
            'nav2_goal_updated_condition_bt_node',
            'nav2_globally_updated_goal_condition_bt_node',
            'nav2_is_path_valid_condition_bt_node',
            'nav2_initial_pose_received_condition_bt_node',
            'nav2_reinitialize_global_localization_service_bt_node',
            'nav2_rate_controller_bt_node',
            'nav2_distance_controller_bt_node',
            'nav2_speed_controller_bt_node',
            'nav2_truncate_path_action_bt_node',
            'nav2_truncate_path_local_action_bt_node',
            'nav2_goal_updater_node_bt_node',
            'nav2_recovery_node_bt_node',
            'nav2_pipeline_sequence_bt_node',
            'nav2_round_robin_node_bt_node',
            'nav2_transform_available_condition_bt_node',
            'nav2_time_expired_condition_bt_node',
            'nav2_path_expiring_timer_condition',
            'nav2_distance_traveled_condition_bt_node',
            'nav2_single_trigger_bt_node',
            'nav2_is_battery_low_condition_bt_node',
            'nav2_navigate_through_poses_action_bt_node',
            'nav2_navigate_to_pose_action_bt_node',
            'nav2_remove_passed_goals_action_bt_node',
            'nav2_planner_selector_bt_node',
            'nav2_controller_selector_bt_node',
            'nav2_goal_checker_selector_bt_node',
            'nav2_controller_cancel_bt_node',
            'nav2_path_longer_on_approach_bt_node',
            'nav2_wait_cancel_bt_node',
            'nav2_spin_cancel_bt_node',
            'nav2_back_up_cancel_bt_node',
            'nav2_drive_on_heading_cancel_bt_node',
        ],
    }
    
    # Global costmap parameters (FLATTENED - NO double nesting)
    global_costmap_params = {
        'use_sim_time': use_sim_time,
        'global_frame': 'map',
        'robot_base_frame': 'base_footprint',
        'update_frequency': 1.0,
        'publish_frequency': 1.0,
        'resolution': 0.05,
        'robot_radius': 0.15,
        'track_unknown_space': True,
        'rolling_window': False,
        'width': 10,
        'height': 10,
        'origin_x': -5.0,
        'origin_y': -5.0,
        'always_send_full_costmap': True,
        'plugins': ['obstacle_layer', 'inflation_layer'],
        'obstacle_layer': {
            'plugin': 'nav2_costmap_2d::ObstacleLayer',
            'enabled': True,
            'footprint_clearing_enabled': True,
            'max_obstacle_height': 2.0,
            'combination_method': 1,
            'observation_sources': 'scan',
            'scan': {
                'topic': '/scan',
                'max_obstacle_height': 2.0,
                'clearing': True,
                'marking': True,
                'data_type': 'LaserScan',
                'raytrace_max_range': 3.0,
                'raytrace_min_range': 0.0,
                'obstacle_max_range': 2.5,
                'obstacle_min_range': 0.0,
            },
        },
        'inflation_layer': {
            'plugin': 'nav2_costmap_2d::InflationLayer',
            'cost_scaling_factor': 3.0,
            'inflation_radius': 0.55,
        },
    }
    
    # Local costmap parameters (FLATTENED - NO double nesting)
    local_costmap_params = {
        'use_sim_time': use_sim_time,
        'global_frame': 'odom',
        'robot_base_frame': 'base_footprint',
        'update_frequency': 5.0,
        'publish_frequency': 2.0,
        'resolution': 0.05,
        'robot_radius': 0.15,
        'rolling_window': True,
        'width': 5,
        'height': 5,
        'always_send_full_costmap': True,
        'plugins': ['obstacle_layer', 'inflation_layer'],
        'obstacle_layer': {
            'plugin': 'nav2_costmap_2d::ObstacleLayer',
            'enabled': True,
            'footprint_clearing_enabled': True,
            'max_obstacle_height': 2.0,
            'combination_method': 1,
            'observation_sources': 'scan',
            'scan': {
                'topic': '/scan',
                'max_obstacle_height': 2.0,
                'clearing': True,
                'marking': True,
                'data_type': 'LaserScan',
                'raytrace_max_range': 3.0,
                'raytrace_min_range': 0.0,
                'obstacle_max_range': 2.5,
                'obstacle_min_range': 0.0,
            },
        },
        'inflation_layer': {
            'plugin': 'nav2_costmap_2d::InflationLayer',
            'cost_scaling_factor': 3.0,
            'inflation_radius': 0.55,
        },
    }
    
    # BT Navigator node
    nav2_bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[bt_navigator_params],
    )
    
    # Planner server node
    nav2_planner = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[planner_params],
    )
    
    # Controller server node
    nav2_controller = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[controller_params],
    )
    
    # Global costmap node (FIXED - proper structure)
    nav2_global_costmap = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='global_costmap',
        namespace='',
        output='screen',
        parameters=[global_costmap_params],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    )
    
    # Local costmap node (FIXED - proper structure)
    nav2_local_costmap = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        namespace='',
        output='screen',
        parameters=[local_costmap_params],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    )
    
    # Lifecycle manager - starts/manages all Nav2 nodes
    nav2_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'node_names': [
                'controller_server',
                'planner_server',
                'bt_navigator',
                'global_costmap',
                'local_costmap',
            ]
        }],
    )
    
    # Obstacle detector node
    obstacle_detector_node = Node(
        package='perception',
        executable='rover_detector_with_pose',
        name='obstacle_detector',
        output='screen',
        parameters=[{
            'no_gui': detector_no_gui,
            'max_distance': detector_max_distance,
            'enable_buffer': True,
            'buffer_alpha': 0.7,
        }],
    )
    
    # RF2O Laser Odometry node
    rf2o_odometry = Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        output='screen',
        parameters=[{
            'laser_scan_topic': rf2o_laser_scan_topic,
            'odom_topic': rf2o_odom_topic,
            'base_frame_id': rf2o_base_frame,
            'odom_frame_id': rf2o_odom_frame,
            'publish_tf': True,
            'freq': 20.0,
        }],
        condition=IfCondition(
            PythonExpression(["'", odometry_source, "' == 'rf2o'"])
        )
    )
    
    # Dead Reckoning Odometry node
    dead_reckoning_odometry = Node(
        package='control',
        executable='dead_reckoning_odom',
        name='dead_reckoning_odom',
        output='screen',
        parameters=[{
            'odom_topic': dead_reckoning_odom_topic,
            'cmd_vel_topic': dead_reckoning_cmd_vel_topic,
            'base_frame_id': dead_reckoning_base_frame,
            'odom_frame_id': dead_reckoning_odom_frame,
            'publish_tf': dead_reckoning_publish_tf,
            'publish_rate': 50.0,
        }],
        condition=IfCondition(
            PythonExpression(["'", odometry_source, "' == 'dead_reckoning'"])
        )
    )
    
    # Control visualization node
    control_visualization = Node(
        package='visualizations',
        executable='control_visualization',
        name='control_visualization',
        output='screen',
        parameters=[{
            'start_x': start_x,
            'start_y': start_y,
            'start_theta': start_theta,
            'target_x': nav2_strategy_target_x,
            'target_y': nav2_strategy_target_y,
            'draw_field': True,
            'scale_px_per_mm': 0.1,
        }],
        condition=IfCondition(launch_visualization)
    )
    
    # Nav2 strategy node (optional)
    nav2_strategy = Node(
        package='control',
        executable='nav2_strategy',
        name='nav2_navigation_strategy',
        output='screen',
        parameters=[{
            'robot_pose_topic': '/odom_pose_processed',
            'obstacle_topic': '/detected_rovers',
            'target_x': nav2_strategy_target_x,
            'target_y': nav2_strategy_target_y,
            'auto_navigate': nav2_strategy_auto_navigate,
            'goal_tolerance': 0.2,
        }],
        condition=IfCondition(launch_nav2_strategy)
    )
    
    # Static transform: map → odom (identity transform)
    static_tf_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    
    # Static transforms for robot structure (if needed)
    static_tf_base_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=[
            '0.0398145505519817', '0.0', '0.04', '0.0', '0.0', '0.7071080798594737', '0.7071054825112364',
            'base_link', 'base_lidar_link'
        ],
        condition=IfCondition(publish_static_tf)
    )
    
    static_tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '0.06531', '0.0', '0.021953', '0.0', '0.0', '0.0', '1.0',
            'base_link', '3d_camera_link'
        ],
        condition=IfCondition(publish_static_tf)
    )
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('max_vel_x', default_value='0.3'),
        DeclareLaunchArgument('min_vel_x', default_value='0.05'),
        DeclareLaunchArgument('max_vel_theta', default_value='1.0'),
        DeclareLaunchArgument('min_vel_theta', default_value='0.1'),
        DeclareLaunchArgument('acc_lim_x', default_value='0.5'),
        DeclareLaunchArgument('acc_lim_theta', default_value='1.0'),
        DeclareLaunchArgument('detector_no_gui', default_value='true'),
        DeclareLaunchArgument('detector_max_distance', default_value='4.0'),
        DeclareLaunchArgument('start_x', default_value='0.0'),
        DeclareLaunchArgument('start_y', default_value='0.0'),
        DeclareLaunchArgument('start_theta', default_value='0.0'),
        DeclareLaunchArgument('launch_nav2_strategy', default_value='false'),
        DeclareLaunchArgument('nav2_strategy_target_x', default_value='5.0'),
        DeclareLaunchArgument('nav2_strategy_target_y', default_value='4.0'),
        DeclareLaunchArgument('nav2_strategy_auto_navigate', default_value='true'),
        DeclareLaunchArgument('odometry_source', default_value='rf2o'),
        DeclareLaunchArgument('rf2o_laser_scan_topic', default_value='/scan'),
        DeclareLaunchArgument('rf2o_odom_topic', default_value='/odom_rf2o'),
        DeclareLaunchArgument('rf2o_base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('rf2o_odom_frame', default_value='odom'),
        DeclareLaunchArgument('dead_reckoning_odom_topic', default_value='/odom'),
        DeclareLaunchArgument('dead_reckoning_cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('dead_reckoning_base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('dead_reckoning_odom_frame', default_value='odom'),
        DeclareLaunchArgument('dead_reckoning_publish_tf', default_value='true'),
        DeclareLaunchArgument('launch_visualization', default_value='false'),
        DeclareLaunchArgument('publish_static_tf', default_value='false'),
        
        # Nodes
        static_tf_map_to_odom,
        nav2_bt_navigator,
        nav2_planner,
        nav2_controller,
        nav2_global_costmap,
        nav2_local_costmap,
        nav2_lifecycle,
        rf2o_odometry,
        dead_reckoning_odometry,
        obstacle_detector_node,
        nav2_strategy,
        control_visualization,
        static_tf_base_to_laser,
        static_tf_base_to_camera,
    ])