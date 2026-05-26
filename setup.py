from glob import glob
from setuptools import setup

package_name = 'save_frames_gui'

setup(
    name=package_name,
    version='0.1.2',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Samuele Sandrini',
    maintainer_email='samuelesandrini@cnr.it',
    description='ROS 2 multi-camera RGB/depth frame preview and dataset saver GUI.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'save_frames_gui_node = save_frames_gui.save_frames_gui_node:main',
        ],
    },
)
