from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the image subscriber node."""
    return LaunchDescription([
        Node(
            package='image_tools',
            executable='img_subscriber_uni',
            name='image_subscriber',
            output='screen',
        ),
    ])

