#!/usr/bin/env python3
"""
Launch file for Nav2 navigation with obstacle detector.

Launches:
- Nav2 stack (controller_server, planner_server with embedded costmaps)
- RF2O laser odometry or dead reckoning
- Obstacle detector
- Optional: Nav2 strategy node
- Optional: Visualization
- Static TF transforms

Nav2 uses laser scan for obstacle detection.
TF tree: map → base_footprint → base_link (RF2O publishes map→base_footprint)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Nav2 with obstacle detector."""
    
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
    nav2_strategy_target_yaw = LaunchConfiguration('nav2_strategy_target_yaw', default='0.0')
    nav2_strategy_auto_navigate = LaunchConfiguration('nav2_strategy_auto_navigate', default='true')
    
    # Starting position parameters
    start_x = LaunchConfiguration('start_x', default='0.0')
    start_y = LaunchConfiguration('start_y', default='0.0')
    start_theta = LaunchConfiguration('start_theta', default='0.0')
    
    # Odometry source selection
    odometry_source = LaunchConfiguration('odometry_source', default='rf2o')
    
    # RF2O laser odometry parameters
    rf2o_laser_scan_topic = LaunchConfiguration('rf2o_laser_scan_topic', default='/scan')
    rf2o_odom_topic = LaunchConfiguration('rf2o_odom_topic', default='/odom')
    rf2o_base_frame = LaunchConfiguration('rf2o_base_frame', default='base_footprint')
    
    # Dead reckoning odometry parameters
    dead_reckoning_odom_topic = LaunchConfiguration('dead_reckoning_odom_topic', default='/odom')
    dead_reckoning_cmd_vel_topic = LaunchConfiguration('dead_reckoning_cmd_vel_topic', default='/cmd_vel')
    dead_reckoning_base_frame = LaunchConfiguration('dead_reckoning_base_frame', default='base_footprint')
    dead_reckoning_publish_tf = LaunchConfiguration('dead_reckoning_publish_tf', default='true')
    
    # Visualization parameters
    launch_visualization = LaunchConfiguration('launch_visualization', default='false')
    
    # Robot frame parameters
    publish_static_tf = LaunchConfiguration('publish_static_tf', default='false')
    
    # ============================================================================
    # Nav2 Parameters Configuration
    # ============================================================================
    
    # Controller parameters (creates local_costmap internally as sub-node)
    controller_params = {
        'use_sim_time': use_sim_time,
        'controller_frequency': 20.0,
        'min_x_velocity_threshold': 0.001,
        'min_theta_velocity_threshold': 0.001,
        'failure_tolerance': 0.3,
        'progress_checker_plugin': 'progress_checker',
        'goal_checker_plugins': ['general_goal_checker'],
        'controller_plugins': ['FollowPath'],
        'progress_checker': {
            'plugin': 'nav2_controller::SimpleProgressChecker',
            'required_movement_radius': 0.5,
            'movement_time_allowance': 10.0,
        },
        'general_goal_checker': {
            'plugin': 'nav2_controller::SimpleGoalChecker',
            'xy_goal_tolerance': 0.25,
            'yaw_goal_tolerance': 0.25,
            'stateful': True,
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
        # Local costmap embedded in controller (sub-node)
        'local_costmap': {
            'local_costmap': {
                'ros__parameters': {
                    'use_sim_time': use_sim_time,
                    'global_frame': 'map',  # Uses map frame
                    'robot_base_frame': 'base_footprint',
                    'update_frequency': 5.0,
                    'publish_frequency': 2.0,
                    'rolling_window': True,
                    'width': 5,
                    'height': 5,
                    'resolution': 0.05,
                    'robot_radius': 0.15,
                    'plugins': ['obstacle_layer', 'inflation_layer'],
                    'obstacle_layer': {
                        'plugin': 'nav2_costmap_2d::ObstacleLayer',
                        'enabled': True,
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
            }
        }
    }
    
    # Planner parameters (creates global_costmap internally as sub-node)
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
        # Global costmap embedded in planner (sub-node)
        'global_costmap': {
            'global_costmap': {
                'ros__parameters': {
                    'use_sim_time': use_sim_time,
                    'global_frame': 'map',  # Uses map frame
                    'robot_base_frame': 'base_footprint',
                    'update_frequency': 1.0,
                    'publish_frequency': 1.0,
                    'rolling_window': False,
                    'width': 20,
                    'height': 20,
                    'origin_x': -10.0,
                    'origin_y': -10.0,
                    'resolution': 0.05,
                    'robot_radius': 0.15,
                    'track_unknown_space': True,
                    'plugins': ['obstacle_layer', 'inflation_layer'],
                    'obstacle_layer': {
                        'plugin': 'nav2_costmap_2d::ObstacleLayer',
                        'enabled': True,
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
            }
        }
    }
    
    # ============================================================================
    # Node Definitions
    # ============================================================================
    
    # REMOVED: static_tf_map_to_odom (was causing disconnected TF tree)
    # RF2O now publishes map→base_footprint directly
    
    # Static transform base_footprint → base_link
    # Nav2 internally looks for base_link even though costmaps use base_footprint
    static_tf_base_footprint_to_base_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_to_base_link_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_link']
    )
    
    # RF2O Laser Odometry node
    # CRITICAL FIX: odom_frame_id changed to 'map' to create proper TF tree
    # Creates: map → base_footprint
    rf2o_odometry = Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        output='screen',
        parameters=[{
            'laser_scan_topic': rf2o_laser_scan_topic,
            'odom_topic': rf2o_odom_topic,
            'base_frame_id': rf2o_base_frame,
            'odom_frame_id': 'map',  # CHANGED: Publish in map frame instead of odom
            'publish_tf': True,
            'freq': 20.0,
        }],
        condition=IfCondition(
            PythonExpression(["'", odometry_source, "' == 'rf2o'"])
        )
    )
    
    # Dead Reckoning Odometry node
    # CRITICAL FIX: odom_frame_id changed to 'map'
    dead_reckoning_odometry = Node(
        package='control',
        executable='dead_reckoning_odom',
        name='dead_reckoning_odom',
        output='screen',
        parameters=[{
            'odom_topic': dead_reckoning_odom_topic,
            'cmd_vel_topic': dead_reckoning_cmd_vel_topic,
            'base_frame_id': dead_reckoning_base_frame,
            'odom_frame_id': 'map',  # CHANGED: Publish in map frame instead of odom
            'publish_tf': dead_reckoning_publish_tf,
            'publish_rate': 50.0,
        }],
        condition=IfCondition(
            PythonExpression(["'", odometry_source, "' == 'dead_reckoning'"])
        )
    )
    
    # Nav2 Controller Server
    nav2_controller = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[controller_params],
    )
    
    # Nav2 Planner Server
    nav2_planner = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[planner_params],
    )
    
    # Nav2 Lifecycle Manager
    # BT Navigator removed - only managing controller and planner
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
            'target_yaw': nav2_strategy_target_yaw,
            'auto_navigate': nav2_strategy_auto_navigate,
            'goal_tolerance': 0.2,
        }],
        condition=IfCondition(launch_nav2_strategy)
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
            'target_x': nav2_strategy_target_x,
            'target_y': nav2_strategy_target_y,
            'draw_field': True,
            'scale_px_per_mm': 0.1,
        }],
        condition=IfCondition(launch_visualization)
    )
    
    # Static transforms for robot structure (optional)
    static_tf_base_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=[
            '0.0398145505519817', '0.0', '0.04', 
            '0.0', '0.0', '0.7071080798594737', '0.7071054825112364',
            'base_link', 'base_lidar_link'
        ],
        condition=IfCondition(publish_static_tf)
    )
    
    static_tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '0.06531', '0.0', '0.021953', 
            '0.0', '0.0', '0.0', '1.0',
            'base_link', '3d_camera_link'
        ],
        condition=IfCondition(publish_static_tf)
    )
    
    # ============================================================================
    # Launch Description
    # ============================================================================
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument('use_sim_time', default_value='false',
                             description='Use simulation time'),
        DeclareLaunchArgument('autostart', default_value='true',
                             description='Automatically startup Nav2 lifecycle nodes'),
        
        # Speed parameters
        DeclareLaunchArgument('max_vel_x', default_value='0.3',
                             description='Maximum linear velocity (m/s)'),
        DeclareLaunchArgument('min_vel_x', default_value='0.05',
                             description='Minimum linear velocity (m/s)'),
        DeclareLaunchArgument('max_vel_theta', default_value='1.0',
                             description='Maximum angular velocity (rad/s)'),
        DeclareLaunchArgument('min_vel_theta', default_value='0.1',
                             description='Minimum angular velocity (rad/s)'),
        DeclareLaunchArgument('acc_lim_x', default_value='0.5',
                             description='Linear acceleration limit (m/s²)'),
        DeclareLaunchArgument('acc_lim_theta', default_value='1.0',
                             description='Angular acceleration limit (rad/s²)'),
        
        # Obstacle detector parameters
        DeclareLaunchArgument('detector_no_gui', default_value='true',
                             description='Disable GUI for obstacle detector'),
        DeclareLaunchArgument('detector_max_distance', default_value='4.0',
                             description='Maximum detection distance (meters)'),
        
        # Starting position parameters
        DeclareLaunchArgument('start_x', default_value='0.0',
                             description='Starting x position (meters)'),
        DeclareLaunchArgument('start_y', default_value='0.0',
                             description='Starting y position (meters)'),
        DeclareLaunchArgument('start_theta', default_value='0.0',
                             description='Starting orientation (radians)'),
        
        # Nav2 strategy parameters
        DeclareLaunchArgument('launch_nav2_strategy', default_value='false',
                             description='Launch Nav2 strategy node'),
        DeclareLaunchArgument('nav2_strategy_target_x', default_value='5.0',
                             description='Target x position (meters)'),
        DeclareLaunchArgument('nav2_strategy_target_y', default_value='4.0',
                             description='Target y position (meters)'),
        DeclareLaunchArgument('nav2_strategy_target_yaw', default_value='0.0',
                             description='Target yaw angle (radians)'),
        DeclareLaunchArgument('nav2_strategy_auto_navigate', default_value='true',
                             description='Auto-navigate on startup'),
        
        # Odometry source selection
        DeclareLaunchArgument('odometry_source', default_value='rf2o',
                             description='Odometry source: "rf2o" or "dead_reckoning"'),
        
        # RF2O parameters
        DeclareLaunchArgument('rf2o_laser_scan_topic', default_value='/scan',
                             description='Laser scan topic for RF2O'),
        DeclareLaunchArgument('rf2o_odom_topic', default_value='/odom',
                             description='Output odometry topic from RF2O'),
        DeclareLaunchArgument('rf2o_base_frame', default_value='base_footprint',
                             description='Base frame ID for RF2O'),
        
        # Dead reckoning parameters
        DeclareLaunchArgument('dead_reckoning_odom_topic', default_value='/odom',
                             description='Output odometry topic for dead reckoning'),
        DeclareLaunchArgument('dead_reckoning_cmd_vel_topic', default_value='/cmd_vel',
                             description='Input cmd_vel topic for dead reckoning'),
        DeclareLaunchArgument('dead_reckoning_base_frame', default_value='base_footprint',
                             description='Base frame ID for dead reckoning'),
        DeclareLaunchArgument('dead_reckoning_publish_tf', default_value='true',
                             description='Publish TF transform for dead reckoning'),
        
        # Visualization parameters
        DeclareLaunchArgument('launch_visualization', default_value='false',
                             description='Launch control visualization node'),
        
        # TF parameters
        DeclareLaunchArgument('publish_static_tf', default_value='false',
                             description='Publish static transforms for robot structure'),
        
        # ========================================================================
        # Nodes (in launch order)
        # ========================================================================
        
        # 1. Static TF: base_footprint → base_link
        static_tf_base_footprint_to_base_link,
        
        # 2. Optional static TFs for robot structure
        static_tf_base_to_laser,
        static_tf_base_to_camera,
        
        # 3. Odometry source (RF2O or dead reckoning) - publishes map→base_footprint
        rf2o_odometry,
        dead_reckoning_odometry,
        
        # 4. Nav2 core nodes (BT Navigator removed)
        nav2_controller,
        nav2_planner,
        
        # 5. Nav2 lifecycle manager (only manages controller and planner)
        nav2_lifecycle,
        
        # 6. Perception (obstacle detector)
        obstacle_detector_node,
        
        # 7. Optional: Nav2 strategy (sends goals)
        nav2_strategy,
        
        # 8. Optional: Visualization
        control_visualization,
    ])