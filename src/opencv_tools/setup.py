from setuptools import find_packages, setup

package_name = 'opencv_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'img_publisher = opencv_tools.basic_image_publisher:main',
        'img_subscriber = opencv_tools.basic_image_subscriber:main',
        'img_subscriber_canny = opencv_tools.basic_image_subscriber_detector:main',
        'img_subscriber_custom = opencv_tools.basic_image_subscriber_custom:main',
        'line_follower = opencv_tools.line_follower_node:main',
        'img_subscriber_notcanny = opencv_tools.basic_image_subscriber_notcanny:main',
        'img_subscriber_rect = opencv_tools.basic_image_subscriber_rect:main',
    ],
},

)
