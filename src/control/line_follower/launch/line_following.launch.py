from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch the complete line following system.
    Starts: image subscriber, image detection, line follower, and optional visualization.
    """
    
    # Arguments
    detector_type_arg = DeclareLaunchArgument(
        'detector_type',
        default_value='custom',
        description='Detection algorithm: custom, canny, brightness, gradient, skeleton'
    )
    
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
    
    enable_visualization_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='false',
        description='Enable visualization node'
    )
    
    # Get package paths
    image_tools_share = FindPackageShare('image_tools')
    perception_share = FindPackageShare('perception')
    line_follower_share = FindPackageShare('line_follower')
    visualizations_share = FindPackageShare('visualizations')
    
    # Include launch files
    image_subscriber_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([image_tools_share, 'launch', 'image_subscriber.launch.py'])
        )
    )
    
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([perception_share, 'launch', 'image_detection.launch.py'])
        ),
        launch_arguments={
            'detector_type': LaunchConfiguration('detector_type'),
        }.items()
    )
    
    line_follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([line_follower_share, 'launch', 'line_follower.launch.py'])
        ),
        launch_arguments={
            'speed_control': LaunchConfiguration('speed_control'),
            'selector': LaunchConfiguration('selector'),
            'smoothing_factor': LaunchConfiguration('smoothing_factor'),
            'forward_speed': LaunchConfiguration('forward_speed'),
        }.items()
    )
    
    # Visualization node (optional)
    visualization_node = Node(
        package='visualizations',
        executable='visualization_node',
        name='visualization',
        output='screen',
        condition=LaunchConfiguration('enable_visualization'),
    )
    
    return LaunchDescription([
        detector_type_arg,
        speed_control_arg,
        selector_arg,
        smoothing_factor_arg,
        forward_speed_arg,
        enable_visualization_arg,
        image_subscriber_launch,
        perception_launch,
        line_follower_launch,
        visualization_node,
    ])

