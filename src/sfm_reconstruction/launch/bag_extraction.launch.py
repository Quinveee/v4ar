from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os

def generate_launch_description():
    """Launch file for SfM rosbag extraction."""
    
    # Arguments
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value='',
        description='Path to ROS2 rosbag'
    )
    
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value=['./sfm_data'],
        description='Output directory for extracted data'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value=['/camera/color/image_raw'],
        description='Image topic name'
    )
    
    camera_info_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value=['/camera/color/camera_info'],
        description='Camera info topic name'
    )
    
    skip_camera_info_arg = DeclareLaunchArgument(
        'skip_camera_info',
        default_value=['false'],
        choices=['true', 'false'],
        description='Skip camera info extraction'
    )
    
    # Extraction command
    extract_cmd = ExecuteProcess(
        cmd=['ros2', 'run', 'sfm_reconstruction', 'bag_extractor',
             LaunchConfiguration('bag_path'),
             '-o', LaunchConfiguration('output_dir'),
             '-i', LaunchConfiguration('image_topic'),
             '-c', LaunchConfiguration('camera_info_topic')],
        output='screen'
    )
    
    return LaunchDescription([
        bag_path_arg,
        output_dir_arg,
        image_topic_arg,
        camera_info_arg,
        skip_camera_info_arg,
        extract_cmd,
    ])
