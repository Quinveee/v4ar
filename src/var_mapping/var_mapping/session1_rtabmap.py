from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    
    # Path to rosbag
    bag_path = os.path.join(
        os.path.expanduser('~'),
        'var_mapping',
        'ROS_BAGS',
        'rtabmap_bag_clean'
    )
    
    # Database path (where RTAB-Map saves the map)
    database_path = os.path.join(
        os.path.expanduser('~'),
        'slam_test',
        'data',
        'session1',
        'rtabmap.db'
    )
    
    return LaunchDescription([
        
        # Play rosbag
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_path],
            output='screen'
        ),
        
        # RTAB-Map node
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                'database_path': database_path,
                'frame_id': 'base_link',
                'subscribe_depth': True,
                'subscribe_rgb': True,
                'approx_sync': True,
            }],
            remappings=[
                ('rgb/image', '/camera/rgb/image_raw'),
                ('rgb/camera_info', '/camera/rgb/camera_info'),
                ('depth/image', '/camera/depth/image_raw'),
            ],
            arguments=['--delete_db_on_start']  # Start fresh
        ),
        
        # RTAB-Map visualization (optional)
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            name='rtabmap_viz',
            output='screen',
        ),
    ])