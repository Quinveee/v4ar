from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['marker_gaze', "*marker_gaze*", 'line_follower', 'launch', 'config',
                                    "curling_control", "*curling_control*", "marker_gaze.gaze_strategies",
                                    "curling_control.curling_strategies", "curling_control.navigation_strategies",
                                    "telecontrol", "*telecontrol*", "control_strategies", "*control_strategies*",
                                    "control_node", "*control_node*"]
                           ),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*_launch.py'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools', "apriltag"],
    zip_safe=True,
    maintainer='root',
    maintainer_email='bierling.lukas@gmail.com',
    description='Control module containing controllers',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'line_follower = line_follower.line_follower:main',
            'marker_gaze = marker_gaze.marker_gaze:main',
            'teleop = telecontrol.teleop_gui:main',
            'follow = curling_control.follow:main',
            "curling_gaze = curling_control.follow:main",
            'dwa_strategy = curling_control.navigation_strategies.dwa_strategy:main',
            'potential_field_strategy = curling_control.navigation_strategies.potential_field_strategy:main',
            'direct_goal_strategy = curling_control.navigation_strategies.direct_goal_strategy:main',
            'control_node = control_node.control_node:main',
        ],
    },
)
