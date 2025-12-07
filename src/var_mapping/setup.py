from setuptools import setup
from glob import glob
import os

package_name = 'var_mapping'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, f'{package_name}.utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=[
        'setuptools',
        'numpy<2.0',  # Required for cv_bridge compatibility
        'opencv-python>=4.5.0',  # For image processing
        'open3d>=0.17.0',  # For point cloud visualization
        'matplotlib>=3.5.0',  # For plotting camera trajectories and frames
    ],
    zip_safe=True,
    maintainer='VAR Team',
    maintainer_email='your.email@example.com',
    description='VAR Lab 3 - Mapping Package (Sessions 1 & 2)',
    license='MIT',
    entry_points={
        'console_scripts': [
            # Scripts that can be run with 'ros2 run'
            'session1_rtabmap = var_mapping.session1_rtabmap:main',
            'map_g = var_mapping.session1_rtabmap:main',
            'display_cam = var_mapping.utils.display_cam:main',
            'visualize_point_cloud = var_mapping.utils.visualization:main',
            'odom_to_tf = var_mapping.utils.odom_to_tf:main',
            'depth_filter = var_mapping.depth_filter_node:main',
            'display_depth_comparison = var_mapping.utils.display_depth_comparison:main',
        ],
    },
)