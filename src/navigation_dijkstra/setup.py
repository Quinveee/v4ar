from setuptools import setup
import os
from glob import glob

package_name = 'navigation_dijkstra'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Simple Dijkstra-based navigation for UGV Rover',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dijkstra_planner = navigation_dijkstra.dijkstra_planner:main',
            'path_follower = navigation_dijkstra.path_follower:main',
        ],
    },
)
