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
        nav2_bringup_dir = FindPackageShare('nav2_bringup').find('nav2_bringup')
    except:
        nav2_bringup_dir = '/opt/ros/humble/share/nav2_bringup'
    
    try:
        control_package_dir = FindPackageShare('control').find('control')
    except:
        control_package_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    autostart = LaunchConfiguration('autostart', default='true')
    use_lifecycle_mgr = LaunchConfiguration('use_lifecycle_mgr', default='true')
    map_yaml_file = LaunchConfiguration('map', default='')
    params_file = LaunchConfiguration('params_file', default='')
    
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
    odometry_source = LaunchConfiguration('odometry_source', default='rf2o')  # 'rf2o' or 'dead_reckoning'
    
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
    publish_static_tf = LaunchConfiguration('publish_static_tf', default='false')  # Set to false since you already have static TFs
    
    # Nav2 controller parameters (speed configuration)
    controller_params = {
        'controller_server': {
            'ros__parameters': {
                'controller_frequency': 20.0,
                'min_x_velocity_threshold': 0.001,
                'min_y_velocity_threshold': 0.5,
                'min_theta_velocity_threshold': 0.001,
                'failure_tolerance': 0.3,
                'progress_checker_plugin': 'progress_checker',
                'goal_checker_plugins': ['general_goal_checker', 'precise_goal_checker'],
                'general_goal_checker': {
                    'plugin': 'nav2_controller::SimpleGoalChecker',
                    'xy_goal_tolerance': 0.25,
                    'yaw_goal_tolerance': 0.25,
                },
                'precise_goal_checker': {
                    'plugin': 'nav2_controller::SimpleGoalChecker',
                    'xy_goal_tolerance': 0.1,
                    'yaw_goal_tolerance': 0.1,
                },
                'progress_checker': {
                    'plugin': 'nav2_controller::SimpleProgressChecker',
                    'required_movement_radius': 0.5,
                    'movement_time_allowance': 10.0,
                },
                'FollowPath': {
                    'plugin': 'nav2_controller::FollowPath',
                    'desired_linear_vel': max_vel_x,
                    'base_desired_linear_vel': max_vel_x,
                    'max_linear_acc': acc_lim_x,
                    'max_linear_decel': acc_lim_x,
                    'max_angular_acc': acc_lim_theta,
                    'max_angular_decel': acc_lim_theta,
                    'in_place_angular_vel': min_vel_theta,
                    'min_in_place_angular_vel': min_vel_theta,
                    'allow_backward': True,
                    'max_angular_vel': max_vel_theta,
                    'min_linear_vel': min_vel_x,
                    'max_linear_vel': max_vel_x,
                    'min_angular_vel': min_vel_theta,
                },
            }
        }
    }
    
    # Nav2 planner parameters (speed configuration)
    # 
    # Nav2 Planners Explained:
    # - nav2_navfn_planner/NavfnPlanner: Classic Dijkstra/A* based planner, fast and reliable
    #   - Good for: Simple environments, quick planning
    #   - Uses: Grid-based pathfinding with potential fields
    # - nav2_smac_planner/SmacPlanner: State Lattice planner with motion models
    #   - Good for: Complex environments, non-holonomic robots, smoother paths
    #   - Uses: State space search with Reeds-Shepp or Dubins curves
    #   - Better for: Robots that can't turn in place, need smooth curves
    planner_params = {
        'planner_server': {
            'ros__parameters': {
                'expected_planner_frequency': 20.0,
                # NavfnPlanner - Classic grid-based planner (always available)
                'GridBased': {
                    'plugin': 'nav2_navfn_planner/NavfnPlanner',
                    'tolerance': 0.5,
                    'use_astar': False,  # False = Dijkstra (guaranteed optimal), True = A* (faster)
                    'allow_unknown': True,  # Allow planning through unknown space
                },
                # SmacPlanner - State lattice planner (if nav2_smac_planner is installed)
                'SmacPlanner': {
                    'plugin': 'nav2_smac_planner/SmacPlanner',
                    'tolerance': 0.5,
                    'max_iterations': 1000000,
                    'max_planning_time': 5.0,
                    'motion_model_for_search': 'REEDS_SHEPP',  # Options: DUBIN, REEDS_SHEPP, STATE_LATTICE
                    'max_linear_acc': acc_lim_x,
                    'max_angular_acc': acc_lim_theta,
                    'max_linear_vel': max_vel_x,
                    'max_angular_vel': max_vel_theta,
                    'min_linear_vel': min_vel_x,
                    'min_angular_vel': min_vel_theta,
                },
            }
        }
    }
    
    # Nav2 costmap parameters
    costmap_params = {
                'global_costmap': {
            'ros__parameters': {
                'global_frame': 'map',
                'robot_base_frame': 'base_footprint',  # Nav2 typically uses base_footprint
                'update_frequency': 1.0,
                'publish_frequency': 1.0,
                'static_layer': {
                    'plugin': 'nav2_costmap_2d::StaticLayer',
                    'enabled': True,
                },
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
                        'expected_update_rate': 0.0,
                        'observation_persistence': 0.0,
                        'inf_is_valid': False,
                    },
                },
                'inflation_layer': {
                    'plugin': 'nav2_costmap_2d::InflationLayer',
                    'cost_scaling_factor': 3.0,
                    'inflation_radius': 0.55,
                },
            }
        },
        'local_costmap': {
            'ros__parameters': {
                'global_frame': 'odom',
                'robot_base_frame': 'base_footprint',  # Nav2 typically uses base_footprint
                'update_frequency': 5.0,
                'publish_frequency': 2.0,
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
                        'expected_update_rate': 0.0,
                        'observation_persistence': 0.0,
                        'inf_is_valid': False,
                    },
                },
                'inflation_layer': {
                    'plugin': 'nav2_costmap_2d::InflationLayer',
                    'cost_scaling_factor': 3.0,
                    'inflation_radius': 0.55,
                },
            }
        },
    }
    
    # Combine all parameters
    all_params = {**controller_params, **planner_params, **costmap_params}
    
    # Nav2 bringup launch (optional - disabled by default since individual nodes are used)
    # Only include if use_nav2_bringup is explicitly set to true
    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
        ),
        launch_arguments=[
            ('use_sim_time', use_sim_time),
            ('autostart', autostart),
            ('use_lifecycle_mgr', use_lifecycle_mgr),
            ('map', map_yaml_file),
            ('params_file', params_file),
        ],
        condition=IfCondition(LaunchConfiguration('use_nav2_bringup', default='false'))
    )
    
    # Launch Nav2 nodes individually (default - used when bringup is not available)
    # These nodes are required for Nav2 navigation:
    # - bt_navigator: Brain that coordinates everything
    # - planner_server: Plans path from start to goal
    # - controller_server: Follows the planned path
    # - nav2_costmap_2d: Obstacle detection/avoidance (global and local costmaps)
    # - lifecycle_manager: Starts/manages all Nav2 nodes
    nav2_bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[all_params],
        condition=UnlessCondition(LaunchConfiguration('use_nav2_bringup', default='false'))
    )
    
    nav2_planner = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[all_params],
        condition=UnlessCondition(LaunchConfiguration('use_nav2_bringup', default='false'))
    )
    
    nav2_controller = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[all_params],
        condition=UnlessCondition(LaunchConfiguration('use_nav2_bringup', default='false'))
    )
    
    # Global costmap
    nav2_global_costmap = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='global_costmap',
        output='screen',
        parameters=[all_params],
        condition=UnlessCondition(LaunchConfiguration('use_nav2_bringup', default='false'))
    )
    
    # Local costmap
    nav2_local_costmap = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        output='screen',
        parameters=[all_params],
        condition=UnlessCondition(LaunchConfiguration('use_nav2_bringup', default='false'))
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
        condition=UnlessCondition(LaunchConfiguration('use_nav2_bringup', default='false'))
    )
    
    # Obstacle detector node (now has entry point in perception/setup.py)
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
    # RF2O (Range Flow 2D Odometry) generates odometry from laser scan motion
    # This provides smooth odometry between AprilTag localization updates
    # Note: RF2O publishes odom → base_footprint transform
    rf2o_odometry = Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        output='screen',
        parameters=[{
            'laser_scan_topic': rf2o_laser_scan_topic,
            'odom_topic': rf2o_odom_topic,
            'base_frame_id': rf2o_base_frame,  # Uses base_footprint to match your TF tree
            'odom_frame_id': rf2o_odom_frame,
            'publish_tf': True,
            'freq': 20.0,  # Publication frequency
        }],
        condition=IfCondition(
            PythonExpression(["'", odometry_source, "' == 'rf2o'"])
        )
    )
    
    # Dead Reckoning Odometry node
    # Dead reckoning from commanded velocity (cmd_vel)
    # Use this when laser odometry is not available
    # Note: Publishes odom → base_footprint transform
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
            'publish_rate': 50.0,  # Hz
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
    
    # Static transforms for robot structure (if needed)
    # Note: Your TF tree already has these transforms, so publish_static_tf defaults to false
    # Only enable if you need to override or add missing transforms
    static_tf_base_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=[
            '0.0398145505519817', '0.0', '0.04', '0.0', '0.0', '0.7071080798594737', '0.7071054825112364',
            'base_link', 'base_lidar_link'  # Using your actual frame name
        ],
        condition=IfCondition(publish_static_tf)
    )
    
    static_tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '0.06531', '0.0', '0.021953', '0.0', '0.0', '0.0', '1.0',
            'base_link', '3d_camera_link'  # Using your actual frame name
        ],
        condition=IfCondition(publish_static_tf)
    )
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument('use_sim_time', default_value='false',
                             description='Use simulation time'),
        DeclareLaunchArgument('autostart', default_value='true',
                             description='Automatically startup Nav2 lifecycle nodes. '
                             'When true, lifecycle_manager automatically transitions Nav2 nodes '
                             'from unconfigured → inactive → active states. When false, you must '
                             'manually activate nodes using lifecycle commands.'),
        DeclareLaunchArgument('use_lifecycle_mgr', default_value='true',
                             description='Use lifecycle manager for Nav2 nodes'),
        DeclareLaunchArgument('map', default_value='',
                             description='Path to map yaml file'),
        DeclareLaunchArgument('params_file', default_value='',
                             description='Path to Nav2 parameters file'),
        
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
                             description='Starting x position in world frame (meters)'),
        DeclareLaunchArgument('start_y', default_value='0.0',
                             description='Starting y position in world frame (meters)'),
        DeclareLaunchArgument('start_theta', default_value='0.0',
                             description='Starting orientation in world frame (radians)'),
        
        # Nav2 strategy parameters
        DeclareLaunchArgument('launch_nav2_strategy', default_value='false',
                             description='Launch Nav2 strategy node'),
        DeclareLaunchArgument('nav2_strategy_target_x', default_value='5.0',
                             description='Nav2 strategy target x position (default: 5.0 for example)'),
        DeclareLaunchArgument('nav2_strategy_target_y', default_value='4.0',
                             description='Nav2 strategy target y position (default: 4.0 for example)'),
        DeclareLaunchArgument('nav2_strategy_auto_navigate', default_value='true',
                             description='Auto-navigate on startup. When true, automatically sends '
                             'navigation goal to Nav2 when strategy node starts. When false, you must '
                             'manually trigger navigation via service or code.'),
        
        # Odometry source selection
        DeclareLaunchArgument('odometry_source', default_value='rf2o',
                             description='Odometry source: "rf2o" (laser-based) or "dead_reckoning" (cmd_vel-based)'),
        
        # RF2O parameters
        DeclareLaunchArgument('rf2o_laser_scan_topic', default_value='/scan',
                             description='Laser scan topic for RF2O'),
        DeclareLaunchArgument('rf2o_odom_topic', default_value='/odom_rf2o',
                             description='Output odometry topic from RF2O'),
        DeclareLaunchArgument('rf2o_base_frame', default_value='base_footprint',
                             description='Base frame ID for RF2O (should match your TF tree)'),
        DeclareLaunchArgument('rf2o_odom_frame', default_value='odom',
                             description='Odometry frame ID for RF2O'),
        
        # Dead reckoning parameters
        DeclareLaunchArgument('dead_reckoning_odom_topic', default_value='/odom',
                             description='Output odometry topic for dead reckoning'),
        DeclareLaunchArgument('dead_reckoning_cmd_vel_topic', default_value='/cmd_vel',
                             description='Input cmd_vel topic for dead reckoning'),
        DeclareLaunchArgument('dead_reckoning_base_frame', default_value='base_footprint',
                             description='Base frame ID for dead reckoning'),
        DeclareLaunchArgument('dead_reckoning_odom_frame', default_value='odom',
                             description='Odometry frame ID for dead reckoning'),
        DeclareLaunchArgument('dead_reckoning_publish_tf', default_value='true',
                             description='Publish TF transform for dead reckoning'),
        
        # Visualization parameters
        DeclareLaunchArgument('launch_visualization', default_value='false',
                             description='Launch control visualization node'),
        
        # TF parameters
        DeclareLaunchArgument('publish_static_tf', default_value='false',
                             description='Publish static transforms. Set to false if your URDF/robot '
                             'description already publishes these transforms (default: false)'),
        DeclareLaunchArgument('use_nav2_bringup', default_value='false',
                             description='Use Nav2 bringup launch file (true) or launch nodes individually (false). '
                             'Default is false to use individual nodes.'),
        # Nodes
        nav2_bringup_launch,
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

