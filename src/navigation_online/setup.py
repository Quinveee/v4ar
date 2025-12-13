from setuptools import setup
import os
from glob import glob

package_name = 'navigation_online'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='V4AR Team',
    maintainer_email='your@email.com',
    description='Online navigation system with RTAB-Map and Dijkstra planning',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rtabmap_bridge = navigation_online.rtabmap_bridge:main',
            'online_navigator = navigation_online.online_navigator:main',
            'command_executor = navigation_online.command_executor:main',
        ],
    },
)

