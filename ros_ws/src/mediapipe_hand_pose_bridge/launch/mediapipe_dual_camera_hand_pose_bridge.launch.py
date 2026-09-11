from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def make_camera_node(side: str, device_arg: str, mirror_arg: str, swap_lr_arg: str, window_name: str) -> Node:
    label = "Left" if side == "left" else "Right"
    return Node(
        package="mediapipe_hand_pose_bridge",
        executable="mediapipe_hand_pose_publisher",
        name=f"{side}_mediapipe_hand_pose_publisher",
        output="screen",
        parameters=[
            {
                "device": ParameterValue(LaunchConfiguration(device_arg), value_type=str),
                "camera_retry_interval_s": ParameterValue(
                    LaunchConfiguration("camera_retry_interval_s"), value_type=float,
                ),
                "target_hand": label,
                "hand_assignment_mode": "target_best",
                "model_complexity": ParameterValue(LaunchConfiguration("model_complexity"), value_type=int),
                "max_hands": ParameterValue(LaunchConfiguration("max_hands"), value_type=int),
                "publish_inactive_tracked": False,
                "mirror": ParameterValue(LaunchConfiguration(mirror_arg), value_type=bool),
                "swap_lr": ParameterValue(LaunchConfiguration(swap_lr_arg), value_type=bool),
                "trapezoid_preprocess": ParameterValue(
                    LaunchConfiguration("trapezoid_preprocess"),
                    value_type=bool,
                ),
                "trapezoid_bottom_width": ParameterValue(
                    LaunchConfiguration("trapezoid_bottom_width"),
                    value_type=int,
                ),
                "trapezoid_canvas_width": ParameterValue(
                    LaunchConfiguration("trapezoid_canvas_width"),
                    value_type=int,
                ),
                "trapezoid_canvas_height": ParameterValue(
                    LaunchConfiguration("trapezoid_canvas_height"),
                    value_type=int,
                ),
                "trapezoid_display_scale": ParameterValue(
                    LaunchConfiguration("trapezoid_display_scale"),
                    value_type=float,
                ),
                "show_image": ParameterValue(LaunchConfiguration("show_image"), value_type=bool),
                "draw_landmarks": ParameterValue(LaunchConfiguration("draw_landmarks"), value_type=bool),
                "show_assignment_debug": ParameterValue(
                    LaunchConfiguration("show_assignment_debug"),
                    value_type=bool,
                ),
                "window_name": window_name,
                "publish_rate_hz": ParameterValue(LaunchConfiguration("publish_rate_hz"), value_type=float),
                "coordinate_scale": ParameterValue(LaunchConfiguration("coordinate_scale"), value_type=float),
                "publish_all_landmarks": ParameterValue(
                    LaunchConfiguration("publish_all_landmarks"),
                    value_type=bool,
                ),
                "preview_output_dir": LaunchConfiguration("preview_output_dir"),
                "preview_side": side,
                "preview_write_hz": ParameterValue(
                    LaunchConfiguration("preview_write_hz"),
                    value_type=float,
                ),
                "right_pose_topic": "/right_mediapipe_hand/poses",
                "right_all_pose_topic": "/right_mediapipe_hand/all_poses",
                "right_tracked_topic": "/right_mediapipe_hand/is_tracked",
                "left_pose_topic": "/left_mediapipe_hand/poses",
                "left_all_pose_topic": "/left_mediapipe_hand/all_poses",
                "left_tracked_topic": "/left_mediapipe_hand/is_tracked",
            }
        ],
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("left_device", default_value="1"),
            DeclareLaunchArgument("right_device", default_value="0"),
            DeclareLaunchArgument("camera_retry_interval_s", default_value="1.0"),
            DeclareLaunchArgument("model_complexity", default_value="1"),
            DeclareLaunchArgument("max_hands", default_value="1"),
            DeclareLaunchArgument("left_mirror", default_value="false"),
            DeclareLaunchArgument("right_mirror", default_value="false"),
            DeclareLaunchArgument("left_swap_lr", default_value="true"),
            DeclareLaunchArgument("right_swap_lr", default_value="true"),
            DeclareLaunchArgument("trapezoid_preprocess", default_value="true"),
            DeclareLaunchArgument("trapezoid_bottom_width", default_value="320"),
            DeclareLaunchArgument("trapezoid_canvas_width", default_value="1280"),
            DeclareLaunchArgument("trapezoid_canvas_height", default_value="720"),
            DeclareLaunchArgument("trapezoid_display_scale", default_value="1.5"),
            DeclareLaunchArgument("show_image", default_value="true"),
            DeclareLaunchArgument("draw_landmarks", default_value="true"),
            DeclareLaunchArgument("show_assignment_debug", default_value="true"),
            DeclareLaunchArgument("publish_rate_hz", default_value="30.0"),
            DeclareLaunchArgument("coordinate_scale", default_value="1.0"),
            DeclareLaunchArgument("publish_all_landmarks", default_value="true"),
            DeclareLaunchArgument("preview_output_dir", default_value=""),
            DeclareLaunchArgument("preview_write_hz", default_value="10.0"),
            make_camera_node("left", "left_device", "left_mirror", "left_swap_lr", "MediaPipe LEFT camera"),
            make_camera_node("right", "right_device", "right_mirror", "right_swap_lr", "MediaPipe RIGHT camera"),
        ]
    )
