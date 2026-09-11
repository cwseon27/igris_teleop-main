from setuptools import setup
from glob import glob
import os

package_name = 'mediapipe_hand_pose_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    scripts=['scripts/mediapipe_hand_pose_publisher'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seon',
    maintainer_email='seon@example.com',
    description='Publish MediaPipe wrist and fingertip 3D landmarks as VR-compatible hand PoseArray topics.',
    license='MIT',
)
