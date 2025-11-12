from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the image detection node with configurable detector type."""
    
    detector_type_arg = DeclareLaunchArgument(
        'detector_type',
        default_value='custom',
        description='Detection algorithm: custom, canny, brightness, gradient, skeleton'
    )
    
    display_window_arg = DeclareLaunchArgument(
        'display_window',
        default_value='false',
        description='Enable OpenCV display window'
    )
    
    vignette_arg = DeclareLaunchArgument(
        'vignette',
        default_value='false',
        description='Enable vignette masking for custom detector'
    )
    
    return LaunchDescription([
        detector_type_arg,
        display_window_arg,
        vignette_arg,
        Node(
            package='perception',
            executable='line_detector',
            name='line_detector',
            output='screen',
            parameters=[
                {'detector_type': LaunchConfiguration('detector_type')},
            ],
            arguments=[
                '--display_window' if LaunchConfiguration('display_window') == 'true' else '',
                '--vignette' if LaunchConfiguration('vignette') == 'true' else '',
            ],
        ),
    ])

