from setuptools import setup

package_name = 'line_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Line follower node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'line_follower = line_follower.line_follower:main',
        ],
    },
)
