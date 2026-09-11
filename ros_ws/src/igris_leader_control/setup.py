"""Set up the igris_leader_control ROS package."""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'igris_leader_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'docs'),
            glob('docs/*.md')),
        (os.path.join('share', package_name, 'udev'),
            glob('udev/*.rules')),
        # 만약 launch 파일을 만드신다면 아래 주석을 해제하세요
        # (os.path.join('share', package_name, 'launch'),
        #     glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'dynamixel_sdk',  # pip 기반 설치 확인용
    ],
    zip_safe=True,
    maintainer='seon',
    maintainer_email='seon@todo.todo',
    description=(
        'Igris Humanoid Leader Arm Control Package using Dynamixel SDK'
    ),
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 실행파일명 = 패키지명.파일명:메인함수이름
            'leader_node = igris_leader_control.leader_node:main'
        ],
    },
)
