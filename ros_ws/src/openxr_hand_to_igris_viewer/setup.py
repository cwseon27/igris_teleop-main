import os
from glob import glob

from setuptools import setup

package_name = 'openxr_hand_to_igris_viewer'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (
            os.path.join('share', package_name, 'meshes', 'igris_c_end_effector', 'hand'),
            glob('meshes/igris_c_end_effector/hand/*.stl'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seon',
    maintainer_email='seon@example.com',
    description='OpenXR hand fingertip norm normalizer and IGRIS hand JointState viewer node.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vr_hand_to_joint_state = openxr_hand_to_igris_viewer.vr_hand_to_joint_state:main',
            'hybrid_openxr_mediapipe_pose_bridge = openxr_hand_to_igris_viewer.hybrid_openxr_mediapipe_pose_bridge:main',
        ],
    },
)
