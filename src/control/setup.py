from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['control', 'control.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Control module containing line_follower and related controllers',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'line_follower = line_follower.line_follower:main',
        ],
    },
)

