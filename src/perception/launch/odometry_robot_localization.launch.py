"""
Launch file for robot_localization EKF node.
This launches robot_localization's ekf_node to fuse:
- rf2o_laser_odometry (/odom_rf2o)
- IMU (/imu/data)

The filtered odometry is published to /odometry/filtered

NOTE: Make sure robot_localization is installed:
  sudo apt-get install ros-humble-robot-localization
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                
                # Frame IDs
                'map_frame': 'map',
                'odom_frame': 'odom',
                'base_link_frame': 'base_footprint',
                'world_frame': 'odom',
                
                # Filter settings
                'frequency': 30.0,
                'two_d_mode': True,
                
                # rf2o_laser_odometry input
                'odom0': '/odom_rf2o',
                'odom0_config': [True,  True,  True,   # x, y, z
                                  False, False, False,  # roll, pitch, yaw
                                  False, False, False,  # vx, vy, vz
                                  False, False, False,  # vroll, vpitch, vyaw
                                  False, False, False], # ax, ay, az
                'odom0_differential': False,
                'odom0_relative': False,
                'odom0_queue_size': 10,
                
                # IMU input
                'imu0': '/imu/data',
                'imu0_config': [False, False, False,   # x, y, z
                                 False, False, True,    # roll, pitch, yaw
                                 False, False, False,   # vx, vy, vz
                                 False, False, True,    # vroll, vpitch, vyaw
                                 True,  True,  False],  # ax, ay, az
                'imu0_differential': False,
                'imu0_relative': False,
                'imu0_remove_gravitational_acceleration': True,
                'imu0_queue_size': 10,
                
                # Process noise (15x15 matrix for 2D mode)
                'process_noise_covariance': [
                    0.05, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0.05, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0.06, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0.03, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0.03, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0.06, 0,    0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0.025, 0,    0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0.025, 0,    0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0.04, 0,    0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.01, 0,    0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.01, 0,    0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.015, 0,    0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.01, 0,    0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.01, 0,
                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0.015
                ],
                
                # Diagnostics
                'print_diagnostics': True,
                'diagnostic_updater_period': 1.0,
            }],
            remappings=[
                # Output topic
                ('odometry/filtered', '/odometry/filtered'),
            ]
        )
    ])

