from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    """Launch file for SfM processing with COLMAP."""
    
    # Arguments
    project_dir_arg = DeclareLaunchArgument(
        'project_dir',
        default_value='./sfm_data',
        description='SfM project directory'
    )
    
    matcher_arg = DeclareLaunchArgument(
        'matcher',
        default_value='exhaustive',
        choices=['exhaustive', 'sequential'],
        description='Feature matching method'
    )
    
    camera_model_arg = DeclareLaunchArgument(
        'camera_model',
        default_value='PINHOLE',
        description='Camera model'
    )
    
    sparse_only_arg = DeclareLaunchArgument(
        'sparse_only',
        default_value='false',
        choices=['true', 'false'],
        description='Only perform sparse reconstruction'
    )
    
    # SfM processing command
    sfm_cmd = ['ros2', 'run', 'sfm_reconstruction', 'sfm_processor',
               LaunchConfiguration('project_dir'),
               '-m', LaunchConfiguration('matcher'),
               '-c', LaunchConfiguration('camera_model')]
    
    # Add sparse-only flag if needed
    sfm_process = ExecuteProcess(
        cmd=sfm_cmd,
        output='screen'
    )
    
    return LaunchDescription([
        project_dir_arg,
        matcher_arg,
        camera_model_arg,
        sparse_only_arg,
        sfm_process,
    ])
