from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mediapipe_share = Path(get_package_share_directory("mediapipe_hand_pose_bridge"))

    hand_inference = Node(
        package="igris_reliability_runtime",
        executable="hand_confidence_inference",
        name="hand_tracking_confidence_inference",
        output="screen",
        parameters=[
            {
                "model_variant": LaunchConfiguration("hand_model_variant"),
                "fallback_variant": "histgb",
                "default_confidence": 0.0,
                "publish_when_not_ready": True,
                "rate_hz": 30.0,
            }
        ],
    )

    controller_inference = Node(
        package="igris_reliability_runtime",
        executable="controller_confidence_inference",
        name="controller_tracking_confidence_inference",
        output="screen",
        parameters=[
            {
                "model_variant": LaunchConfiguration("controller_model_variant"),
                "fallback_variant": "histgb",
                "left_pose_topic": LaunchConfiguration("left_controller_pose_topic"),
                "right_pose_topic": LaunchConfiguration("right_controller_pose_topic"),
                "default_confidence": 0.0,
                "publish_when_not_ready": True,
                "rate_hz": 30.0,
            }
        ],
    )

    mediapipe = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(mediapipe_share / "launch" / "mediapipe_dual_camera_hand_pose_bridge.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("start_mediapipe")),
        launch_arguments={
            "left_device": LaunchConfiguration("left_camera_device"),
            "right_device": LaunchConfiguration("right_camera_device"),
            "model_complexity": "1",
            "max_hands": "1",
            "left_mirror": LaunchConfiguration("left_camera_mirror"),
            "right_mirror": LaunchConfiguration("right_camera_mirror"),
            "left_swap_lr": LaunchConfiguration("left_camera_swap_lr"),
            "right_swap_lr": LaunchConfiguration("right_camera_swap_lr"),
            "trapezoid_preprocess": LaunchConfiguration("trapezoid_preprocess"),
            "trapezoid_bottom_width": LaunchConfiguration("trapezoid_bottom_width"),
            "preview_output_dir": LaunchConfiguration("mediapipe_preview_output_dir"),
            "show_image": LaunchConfiguration("show_mediapipe_image"),
            "show_assignment_debug": "true",
            "publish_rate_hz": "30.0",
            "publish_all_landmarks": "true",
        }.items(),
    )

    # Both observations now use the same six-joint DexRetargeting framework.
    # Never run the legacy distance/5-finger bridge alongside this backend.
    hand_fusion = Node(
        package="igris_reliability_runtime",
        executable="hand_retarget_fusion",
        name="reliability_hand_retarget",
        output="screen",
        parameters=[{
            "openxr_only": ParameterValue(EqualsSubstitution(
                LaunchConfiguration("hand_model_variant"), "always_1"
            ), value_type=bool),
            "publish_rate_hz": 30.0,
            **{parameter: ParameterValue(LaunchConfiguration(argument), value_type=float)
               for parameter, argument in {
                   "confidence_gamma": "hand_confidence_gamma",
                   "previous_command_weight": "hand_previous_command_weight",
                   "close_rate_per_sec": "hand_close_rate_per_sec",
                   "open_rate_per_sec": "hand_open_rate_per_sec",
                   "openxr_only_confidence_threshold": "openxr_only_confidence_threshold",
                   "source_dropout_grace_sec": "hand_source_dropout_grace_sec",
               }.items()},
        }],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("hand_model_variant", default_value="rnn"),
            DeclareLaunchArgument("controller_model_variant", default_value="rnn"),
            DeclareLaunchArgument("left_controller_pose_topic", default_value="/left_controller/poses"),
            DeclareLaunchArgument("right_controller_pose_topic", default_value="/right_controller/poses"),
            DeclareLaunchArgument("start_mediapipe", default_value="true"),
            DeclareLaunchArgument("left_camera_device", default_value="1"),
            DeclareLaunchArgument("right_camera_device", default_value="0"),
            DeclareLaunchArgument("left_camera_mirror", default_value="false"),
            DeclareLaunchArgument("right_camera_mirror", default_value="false"),
            DeclareLaunchArgument("left_camera_swap_lr", default_value="true"),
            DeclareLaunchArgument("right_camera_swap_lr", default_value="true"),
            DeclareLaunchArgument("trapezoid_preprocess", default_value="true"),
            DeclareLaunchArgument("trapezoid_bottom_width", default_value="320"),
            DeclareLaunchArgument("mediapipe_preview_output_dir", default_value=""),
            DeclareLaunchArgument("show_mediapipe_image", default_value="false"),
            DeclareLaunchArgument("hand_confidence_gamma", default_value="1.0"),
            DeclareLaunchArgument("hand_previous_command_weight", default_value="0.0"),
            DeclareLaunchArgument("hand_close_rate_per_sec", default_value="15.0"),
            DeclareLaunchArgument("hand_open_rate_per_sec", default_value="20.0"),
            DeclareLaunchArgument("hand_source_dropout_grace_sec", default_value="0.15"),
            DeclareLaunchArgument("openxr_only_confidence_threshold", default_value="0.6"),
            DeclareLaunchArgument("debug_log", default_value="false"),
            hand_inference,
            controller_inference,
            mediapipe,
            hand_fusion,
        ]
    )
