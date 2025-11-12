from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the line follower node with configurable parameters."""
    
    speed_control_arg = DeclareLaunchArgument(
        'speed_control',
        default_value='gradual',
        description='Speed control mode: gradual, threshold, angle_based, none'
    )
    
    selector_arg = DeclareLaunchArgument(
        'selector',
        default_value='closest',
        description='Line selection strategy: closest, confidence, mean, tracking'
    )
    
    smoothing_factor_arg = DeclareLaunchArgument(
        'smoothing_factor',
        default_value='0.3',
        description='EMA smoothing factor (0-1)'
    )
    
    forward_speed_arg = DeclareLaunchArgument(
        'forward_speed',
        default_value='0.2',
        description='Base forward velocity'
    )
    
    k_angle_arg = DeclareLaunchArgument(
        'k_angle',
        default_value='0.01',
        description='Proportional gain for angle'
    )
    
    k_offset_arg = DeclareLaunchArgument(
        'k_offset',
        default_value='0.005',
        description='Proportional gain for offset'
    )
    
    return LaunchDescription([
        speed_control_arg,
        selector_arg,
        smoothing_factor_arg,
        forward_speed_arg,
        k_angle_arg,
        k_offset_arg,
        Node(
            package='line_follower',
            executable='line_follower',
            name='line_follower',
            output='screen',
            arguments=[
                '--speed_control', LaunchConfiguration('speed_control'),
                '--selector', LaunchConfiguration('selector'),
                '--smoothing_factor', LaunchConfiguration('smoothing_factor'),
                '--forward_speed', LaunchConfiguration('forward_speed'),
                '--k_angle', LaunchConfiguration('k_angle'),
                '--k_offset', LaunchConfiguration('k_offset'),
            ],
        ),
    ])

