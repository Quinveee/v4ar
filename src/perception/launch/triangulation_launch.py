from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('perception'),
        'config',
        'markers.yaml'
    )

    return LaunchDescription([
        Node(
            package='perception',
            executable='triangulation_node',
            name='triangulation_node',
            output='screen',
            parameters=[
                {'marker_config': config_path},
                {'solver_type': 'least_squares'}, 
            ],
        ),
    ])
