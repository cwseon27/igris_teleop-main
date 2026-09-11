from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="stereo_sbs_cam_pub",
            executable="realsense_multi_comp_pub",
            name="realsense_multi_comp_pub",
            output="screen",
            parameters=[{
                "base_ns": "/rs_comp",
                "enable_color": True,
                "enable_infra": True,
                "color_profile": "640x480x30",
                "infra_profile": "640x480x30",
                "jpeg_quality": 80,
                "log_throttle_sec": 5.0,
                "reconnect_wait_sec": 2.0,
            }],
        )
    ])
