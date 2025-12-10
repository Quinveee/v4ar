#!/usr/bin/env python3
"""
RTAB-MAP Offline SLAM Launch File for UGV Rover rosbag

Assumes:
- ros2 bag play is publishing /clock (use_sim_time = true)
- Bag contains at least:
    /oak/rgb/image_rect
    /oak/rgb/camera_info
    /odom
    /tf
    /tf_static

We add a minimal static TF from base_footprint to oak_rgb_camera_optical_frame,
then run RTAB-Map in RGB + odom mode (no scan) to avoid laser TF issues.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    database_path = LaunchConfiguration(
        'database_path',
        default='~/.ros/rtabmap_offline.db'
    )

    import launch
    launch.logging.get_logger("rtabmap_with_static_tf.launch.py").info("[RTABMAP LAUNCH] Updated build: 2025-12-05")
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulated time from /clock (rosbag)'
        ),
        DeclareLaunchArgument(
            'database_path',
            default_value='~/.ros/rtabmap_offline.db',
            description='Output RTAB-Map database path'
        ),

        # Static TF: base_footprint -> oak_rgb_camera_optical_frame
        Node(   
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_oak_rgb_optical',
            arguments=[
                '0.0', '0.0', '0.30',          # x y z [m] (rough mount guess)
                '0.0', '0.0', '0.0',           # roll pitch yaw [rad]
                'base_footprint',              # parent frame
                'oak_rgb_camera_optical_frame' # child frame (matches image frame_id)
            ],
            output='screen'
        ),

        # Static TF: base_footprint -> base_lidar_link (from URDF)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_lidar',
            arguments=[
                '0.040', '0.0', '0.12',        # x y z [m] (from ugv_rover.urdf)
                '0.0', '0.0', '1.5708',        # roll pitch yaw [rad] (90° yaw)
                'base_footprint',              # parent frame
                'base_lidar_link'              # child frame (matches /scan frame_id)
            ],
            output='screen'
        ),

        # RTAB-MAP SLAM node (RGB + odom + scan)
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'database_path': database_path,

                # Base frame should match what your odom topic uses as child_frame_id
                'frame_id': 'base_footprint',

                # Use odom from the /odom topic only, do not require TF odom->base_footprint
                'odom_frame_id': '',
                'wait_for_transform': 0.5,
                'wait_for_transform_duration': 0.5,

                # Subscriptions
                'subscribe_rgb': True,
                'subscribe_depth': False,
                'subscribe_scan': True,
                'subscribe_odom_info': False,

                # Synchronization
                'approx_sync': True,
                'topic_queue_size': 30,
                'sync_queue_size': 30,

                # Some reasonable RTAB-Map tuning defaults
                'Rtabmap/DetectionRate': '1.0',
                'RGBD/NeighborLinkRefining': 'true',
                'RGBD/ProximityBySpace': 'true',
                'RGBD/OptimizeFromGraphEnd': 'false',
                'RGBD/LinearUpdate': '0.1',
                'RGBD/AngularUpdate': '0.1',

                'Mem/IncrementalMemory': 'true',
                'Mem/InitWMWithAllNodes': 'false',
                'Mem/STMSize': '30',

                'RGBD/ProximityPathMaxNeighbors': '10',
                'Kp/MaxFeatures': '400',
                'Kp/DetectorStrategy': '0',  # ORB

                'Grid/3D': 'true',
                'RGBD/CreateOccupancyGrid': 'true',
            }],
            remappings=[
                ('rgb/image',       '/oak/rgb/image_rect'),
                ('rgb/camera_info', '/oak/rgb/camera_info'),
                ('odom',            '/odom/odom_raw'),
                ('scan',            '/scan'),
            ],
            arguments=['-d'],
        ),
    ])