from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'ugv_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        where='.',  # search in current dir
        include=['ugv_launch', "*ugv_launch*",]
    ),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),        
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='quintenengelen@gmail.com',
    description='Launch files and configurations for UGV vision system',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={},
)
