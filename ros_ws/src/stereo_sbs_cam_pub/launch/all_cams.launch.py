from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1) RealSense compressed publisher
        Node(
            package="stereo_sbs_cam_pub",
            executable="realsense_multi_comp_pub",
            name="realsense_multi_comp_pub",
            output="screen",
            parameters=[{
                "base_ns": "/rs_comp",
                "enable_color": True,
                "enable_infra": False,
                "jpeg_quality": 80,
            }],
        ),

        # 2) SBS stereo (OpenCV) publisher  ← device를 by-path로 고정!
        Node(
            package="stereo_sbs_cam_pub",
            executable="sbs_cam_pub",
            name="sbs_cam_pub",
            output="screen",
            parameters=[{
                "device": "/dev/video0",
                # 기존 run.launch.py에서 쓰던 파라미터들도 필요하면 그대로 추가
                # "capture_width": 2560, "capture_height": 720, ...
            }],
        ),

        # RViz는 필요하면 추가(또는 기존 run.launch.py 방식 유지)
        # Node(package='rviz2', executable='rviz2', arguments=['-d', <cfg>], output='screen'),
    ])
