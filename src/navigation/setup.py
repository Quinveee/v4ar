from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'navigation'

setup(
    name=package_name,
    version='0.0.1',
    packages=['navigation', 'navigation.planning', 'navigation.planning.planner'],
    include=["navigation", "*navigation*", "*launch*", "planning", "*planning*"],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='V4AR Team',
    maintainer_email='your@email.com',
    description='Nav2 navigation package for UGV rover',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner = navigation.planning.path_planner:main',
        ],
    },
)
