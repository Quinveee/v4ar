from setuptools import find_packages, setup

package_name = 'visualizations'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
       'visualization_node = visualizations.line_visualization_node:main',
       'angle_plot_node = visualizations.angle_plot_node:main',     
        'field_visualization_node = visualizations.field_visualization_node:main',  

        ],
    },
)
