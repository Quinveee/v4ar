from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        where='.',  # search in current dir
        include=['marker_detector', "*marker_detector*",'line_detector', '*detector*', 'odometry', '*odometry*', 'launch', 'config',
                 "launch", "*launch*", "perception", "perception.*"]
    ),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*_launch.py'))),        
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'launch', 'odometry'), glob(os.path.join('launch', 'odometry_*.py'))),
        (os.path.join('share', package_name, 'config', 'odometry'), glob(os.path.join('config', 'odometry_*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='quintenengelen@gmail.com',
    description='Perception module with line and marker detectors for the UGV.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'line_detector = line_detector.line_detector_node:main',
            'apriltag_vis_node = marker_detector.apriltag_vis_node:main',
            'apriltag_vis_node_2 = marker_detector.apriltag_vis_node_2:main',
            'camera_calibration = marker_detector.camera_calibration:main',
            'triangulation = marker_detector.triangulation_node:main',
            'marker_buffer = marker_detector.marker_buffer_node:main',
            'field_visualization = visualizations.field_visualization_node:main',
            'robust_localization = marker_detector.robust_localization_node:main',
            'apriltag = marker_detector.apriltag_detector:main',
            'oak_apriltag = marker_detector.oak_apriltag_detector:main',
            'localization = marker_detector.localization:main',
            'odometry_node = odometry.odometry:main',
            'oak_mono_triangulation = marker_detector.oak_mono_triangulation_node:main',
            'triangulation_uni = marker_detector.triangulation_uni:main',
            'rover_detector_with_pose = obstacle_detector.detector:main',
        ],
    },
)
