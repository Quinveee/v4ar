from setuptools import setup, find_packages

package_name = 'line_detector'

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
    maintainer_email='quintenengelen@gmail.com',
    description='Modular line detection package with multiple detection algorithms (Canny, Brightness, Gradient).',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # Entry point for running the node
            'line_detector_node = line_detector.line_detector_node:main',
        ],
    },
)
