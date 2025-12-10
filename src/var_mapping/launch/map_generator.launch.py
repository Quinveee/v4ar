"""Launch file for RTAB-Map SLAM map generation.

Usage: 
    # In terminal 1: Launch the SLAM nodes
    ros2 launch var_mapping map_generator.launch.py use_rviz:=true use_sim_time:=true
    
    # In terminal 2: Play the bag file (IMPORTANT: play bag separately!)
    ros2 bag play /path/to/bag --clock
    
    # With depth filtering:
    ros2 launch var_mapping map_generator.launch.py depth_filter_level:=2 use_rviz:=true use_sim_time:=true
    # Then play bag separately: ros2 bag play /path/to/bag --clock
    
Depth Filter Levels:
    -1: No filter, use raw depth (/oak/stereo/image_raw)
     0: Passthrough (no filtering, but through filter node)
     1: Minimal filtering (median 5x5)
     2: Standard filtering (median + bilateral)
     3: Advanced filtering (statistical + bilateral + inpaint)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
from pathlib import Path


def generate_launch_description():
    """Generate launch description for RTAB-Map."""
    
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    queue_size = LaunchConfiguration('queue_size')
    qos = LaunchConfiguration('qos')
    localization = LaunchConfiguration('localization')
    delete_db = LaunchConfiguration('delete_db')
    depth_filter_level = LaunchConfiguration('depth_filter_level')
    subscribe_scan = LaunchConfiguration('subscribe_scan')
    
    # Create maps directory structure
    maps_dir = Path.home() / 'v4ar' / 'maps' / 'rtabmap'
    maps_dir.mkdir(parents=True, exist_ok=True)
    database_path = str(maps_dir / 'rtabmap.db')
    
    # Launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',  # Default to true for bag playback
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='Whether to launch RViz2'
    )
    
    declare_queue_size = DeclareLaunchArgument(
        'queue_size', default_value='20',
        description='Queue size'
    )
    
    declare_qos = DeclareLaunchArgument(
        'qos', default_value='2',
        description='QoS used for input sensor topics'
    )
        
    declare_localization = DeclareLaunchArgument(
        'localization', default_value='false',
        description='Launch in localization mode.'
    )
    
    declare_delete_db = DeclareLaunchArgument(
        'delete_db',
        default_value='false',
        description='Delete existing database on start (true/false)'
    )
    
    declare_depth_filter_level = DeclareLaunchArgument(
        'depth_filter_level',
        default_value='-1',
        description='Depth filter level: -1 (no filter, use raw), 0-3 (filter level)'
    )
    
    declare_subscribe_scan = DeclareLaunchArgument(
        'subscribe_scan',
        default_value='true',
        description='Subscribe to laser scan topic (/scan) for RTAB-Map'
    )
    
    # Get UGV model from environment and create URDF path
    UGV_MODEL = os.environ.get('UGV_MODEL', 'ugv_rover')  # Default to ugv_rover if not set
    urdf_file_name = UGV_MODEL + '.urdf'
    urdf_model_path = os.path.join(
        get_package_share_directory('robot_description'),
        'urdf', 
        urdf_file_name)
    
    # Create robot_state_publisher node with use_sim_time
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        arguments=[urdf_model_path],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Create static transform publisher from base_footprint to base_lidar_link
    static_tf_base_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_to_base_lidar_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_lidar_link'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Create static transform from base_footprint to base_link (if needed)
    static_tf_base_footprint_to_base_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_to_base_link_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_link'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Create static transform from 3d_camera_link to oak_rgb_camera_optical_frame
    # Optical frame is rotated -90 degrees around X-axis (typical for camera optical frames)
    static_tf_camera_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_3d_to_oak_optical_tf',
        arguments=['0', '0', '0', '0.7071068', '-0.7071068', '0', '0', '3d_camera_link', 'oak_rgb_camera_optical_frame'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Node to republish odom->base_footprint transform from /odom topic
    odom_to_tf_node = Node(
        package='var_mapping',
        executable='odom_to_tf',
        name='odom_to_tf',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Depth filter node (only launched if depth_filter_level >= 0)
    # Use OpaqueFunction to evaluate the condition properly
    def create_depth_filter_node(context):
        """Create depth filter node only if filter level >= 0."""
        filter_level_val = context.perform_substitution(depth_filter_level)
        try:
            filter_level_int = int(filter_level_val)
            if filter_level_int >= 0:
                return [Node(
                    package='var_mapping',
                    executable='depth_filter',
                    name='depth_filter_node',
                    parameters=[{
                        'filter_level': filter_level_int,
                        'use_sim_time': use_sim_time
                    }],
                    output='screen'
                )]
        except (ValueError, TypeError):
            pass
        return []
    
    depth_filter_node = OpaqueFunction(function=create_depth_filter_node)
    
    # Launch the robot pose publisher launch file (optional - only if package exists)
    robot_pose_publisher_launch = None
    try:
        robot_pose_publisher_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('robot_pose_publisher'), 'launch'),
             '/robot_pose_publisher_launch.py'])
        )
    except:
        # robot_pose_publisher package not available - skip it (not required for RTABMap)
        pass
    
    # Get RViz config from var_mapping package
    var_mapping_dir = get_package_share_directory('var_mapping')
    rviz_slam_3d_config = os.path.join(var_mapping_dir, 'rviz', 'view_slam_3d.rviz')
    
    # Create rviz2 node
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_slam_3d_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )
    
    # Parameters for the SLAM node
    parameters = {
        "frame_id": 'base_footprint',
        'queue_size': queue_size,
        "subscribe_rgb": True,
        "subscribe_depth": True,
        'subscribe_scan': subscribe_scan,
        "subscribe_odom_info": False,
        "approx_sync": True,
        "Rtabmap/DetectionRate": "3.5",
        "use_sim_time": use_sim_time,
        "database_path": database_path,
        "Vis/MinInliers": "20",  # Require more feature matches
        "Vis/MaxDepth": "3.0",   # Limit depth to 4 meters (adjust based on your environment)
        "Vis/MinDepth": "0.3",   # Minimum depth
    }

    # Dynamic RTAB-Map nodes with conditional depth remapping
    def create_rtabmap_nodes(context):
        """Create RTAB-Map nodes with dynamic depth remapping based on filter level."""
        filter_level_val = context.perform_substitution(depth_filter_level)
        localization_val = context.perform_substitution(localization)
        use_rviz_val = context.perform_substitution(LaunchConfiguration('use_rviz'))
        
        # Determine depth topic based on filter level
        depth_topic = "oak/stereo/image_raw"
        try:
            filter_level_int = int(filter_level_val)
            if filter_level_int >= 0:
                depth_topic = "/oak/stereo/image_raw/compressedDepth/filtered"
        except (ValueError, TypeError):
            pass
        
        # Create remappings with dynamic depth topic
        remappings = [
            ("rgb/image", "oak/rgb/image_rect"),
            ("rgb/camera_info", "oak/rgb/camera_info"),
            ("depth/image", depth_topic),
            ("scan", "/scan"),
        ]
        
        nodes_to_launch = []
        
        # SLAM mode node
        if localization_val.lower() == 'false':
            nodes_to_launch.append(
                Node(
                    package='rtabmap_slam', 
                    executable='rtabmap', 
                    output='screen',
                    parameters=[parameters],
                    remappings=remappings,
                    arguments=['-d']
                )
            )
        
        # Localization mode node
        if localization_val.lower() == 'true':
            nodes_to_launch.append(
                Node(
                    package='rtabmap_slam', 
                    executable='rtabmap', 
                    output='screen',
                    parameters=[
                        parameters,
                        {'Mem/IncrementalMemory': 'False',
                         'Mem/InitWMWithAllNodes': 'True'}
                    ],
                    remappings=remappings
                )
            )
        
        # RTAB-Map viz node (only if not using RViz)
        if use_rviz_val.lower() == 'false':
            nodes_to_launch.append(
                Node(
                    package='rtabmap_viz', 
                    executable='rtabmap_viz', 
                    output='screen',
                    parameters=[parameters],
                    remappings=remappings,
                )
            )
        
        return nodes_to_launch
    
    rtabmap_nodes = OpaqueFunction(function=create_rtabmap_nodes)
                        
    # Build launch description list
    launch_nodes = [
        declare_use_sim_time,
        declare_use_rviz,
        declare_queue_size,
        declare_qos,
        declare_localization,
        declare_delete_db,
        declare_depth_filter_level,
        declare_subscribe_scan,
        robot_state_publisher_node,
        static_tf_base_lidar,
        static_tf_base_footprint_to_base_link,
        static_tf_camera_optical,
        odom_to_tf_node,
        depth_filter_node,
        rtabmap_nodes,  # Dynamic RTAB-Map nodes
        rviz2_node
    ]
    
    # Add robot_pose_publisher only if available
    if robot_pose_publisher_launch is not None:
        launch_nodes.append(robot_pose_publisher_launch)
    
    return LaunchDescription(launch_nodes)