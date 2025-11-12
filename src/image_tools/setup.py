from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'image_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'line_msgs']),
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
    description='Image processing tools for camera feed handling and visualization',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'img_publisher = image_tools.basic_image_publisher:main',
        'img_subscriber_uni = image_tools.image_subscriber:main'
   ],
},

)
