from setuptools import setup
import os
from glob import glob

package_name = 'stereo_sbs_cam_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.npz') + glob('config/*.yml.gz')),

        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seon',
    maintainer_email='seon@example.com',
    description='Open SBS stereo camera (2560x720), split L/R, undistort with given calibration, publish left/right images and camera_info.',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'sbs_cam_pub = stereo_sbs_cam_pub.sbs_cam_split_undistort_pub:main',
            'realsense_multi_comp_pub = stereo_sbs_cam_pub.realsense_multi_compressed_pub:main',
            'zed_pub = stereo_sbs_cam_pub.zed_pub:main',
        ],
    },
)
