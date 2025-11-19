from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        where='.',  # search in current dir
        include=['marker_detector', 'line_detector', '*detector*']
    ),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
    ],
    install_requires=['setuptools', "pupil-apriltags"],
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
        ],
    },
)
