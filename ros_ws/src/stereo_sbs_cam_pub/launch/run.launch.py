from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('stereo_sbs_cam_pub')
    params = os.path.join(pkg_share, 'config', 'params.yaml')
    rviz_cfg = os.path.join(pkg_share, 'rviz', 'stereo_lr_image.rviz')

    use_rviz = LaunchConfiguration('use_rviz')

    cam_node = Node(
        package='stereo_sbs_cam_pub',
        executable='sbs_cam_pub',
        name='sbs_cam_split_undistort_pub',
        output='screen',
        parameters=[params]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg],
        condition=None  # 아래에서 use_rviz 조건 걸고 싶으면 고급 구성 가능
    )


    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        cam_node,
        # rviz_node,
    ])
