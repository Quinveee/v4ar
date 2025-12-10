#!/usr/bin/env python3
"""
RTAB-MAP Offline SLAM Launch File for UGV Rover rosbag

Assumes:
- ros2 bag play is publishing /clock (use_sim_time = true)
- Bag contains:
    /oak/rgb/image_rect
    /oak/rgb/camera_info
    /scan
    /odom
    /tf
    /tf_static
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

        # RTAB-MAP SLAM node (RGB + scan + odom)
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                # ROS-side params
                'use_sim_time': use_sim_time,
                'database_path': database_path,
                # Frame setup: base frame should match what is in TF for the rover
                'frame_id': 'base_footprint',      # change to 'base_link' if TF complains
                # Subscriptions
                'subscribe_rgb': True,
                'subscribe_depth': False,          # keep False for now; bag has compressedDepth only
                'subscribe_scan': True,
                'subscribe_odom_info': False,
                'approx_sync': True,
                'queue_size': 30,
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
                ('scan',            '/scan'),
                ('odom',            '/odom'),
            ],
            # -d: delete previous database at startup so you always get a fresh run
            arguments=['-d'],
        ),
    ])
