from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_float(context, name: str) -> float:
    return float(LaunchConfiguration(name).perform(context))


def launch_bool(context, name: str) -> bool:
    value = LaunchConfiguration(name).perform(context).strip().lower()
    return value in ("1", "true", "yes", "on")


def make_bridge_node(side: str, context):
    return Node(
        package="openxr_hand_to_igris_viewer",
        executable="hybrid_openxr_mediapipe_pose_bridge",
        name=f"{side}_hybrid_openxr_mediapipe_pose_bridge",
        output="screen",
        parameters=[
            {
                "hand_side": side,
                "openxr_only": launch_bool(context, "openxr_only"),
                "apply_calibration_remap": launch_bool(context, "apply_calibration_remap"),
                "openxr_calibration_json": LaunchConfiguration("openxr_calibration_json").perform(context),
                "mediapipe_calibration_json": LaunchConfiguration("mediapipe_calibration_json").perform(context),
                "output_calibration_json": LaunchConfiguration("output_calibration_json").perform(context),
                "publish_rate_hz": launch_float(context, "publish_rate_hz"),
                "pose_stale_sec": launch_float(context, "pose_stale_sec"),
                "confidence_stale_sec": launch_float(context, "confidence_stale_sec"),
                "tracked_stale_sec": launch_float(context, "tracked_stale_sec"),
                "confidence_gamma": launch_float(context, "confidence_gamma"),
                "default_confidence": launch_float(context, "default_confidence"),
                "previous_command_weight": launch_float(context, "previous_command_weight"),
                "close_rate_per_sec": launch_float(context, "close_rate_per_sec"),
                "open_rate_per_sec": launch_float(context, "open_rate_per_sec"),
                "openxr_only_confidence_threshold": launch_float(
                    context, "openxr_only_confidence_threshold"
                ),
                "source_dropout_grace_sec": launch_float(
                    context, "source_dropout_grace_sec"
                ),
                "debug_log": launch_bool(context, "debug_log"),
            }
        ],
    )


def launch_setup(context, *args, **kwargs):
    hand_side = LaunchConfiguration("hand_side").perform(context).lower()
    if hand_side == "both":
        return [make_bridge_node("right", context), make_bridge_node("left", context)]
    if hand_side in ("right", "left"):
        return [make_bridge_node(hand_side, context)]
    raise ValueError("hand_side must be one of: right, left, both")


def generate_launch_description():
    calibration_dir = Path(get_package_share_directory("igris_reliability_runtime")) / "config" / "hand"
    default_openxr_calib = str(calibration_dir / "openxr_hand_calibration.json")
    default_mediapipe_calib = str(calibration_dir / "mediapipe_hand_calibration.json")
    return LaunchDescription(
        [
            DeclareLaunchArgument("hand_side", default_value="right"),
            DeclareLaunchArgument("openxr_only", default_value="false"),
            DeclareLaunchArgument("apply_calibration_remap", default_value="true"),
            DeclareLaunchArgument("openxr_calibration_json", default_value=default_openxr_calib),
            DeclareLaunchArgument("mediapipe_calibration_json", default_value=default_mediapipe_calib),
            DeclareLaunchArgument("output_calibration_json", default_value=default_openxr_calib),
            DeclareLaunchArgument("publish_rate_hz", default_value="30.0"),
            DeclareLaunchArgument("pose_stale_sec", default_value="0.2"),
            DeclareLaunchArgument("confidence_stale_sec", default_value="0.5"),
            DeclareLaunchArgument("tracked_stale_sec", default_value="0.5"),
            DeclareLaunchArgument("confidence_gamma", default_value="1.0"),
            DeclareLaunchArgument("default_confidence", default_value="0.0"),
            DeclareLaunchArgument("previous_command_weight", default_value="0.0"),
            DeclareLaunchArgument("close_rate_per_sec", default_value="15.0"),
            DeclareLaunchArgument("open_rate_per_sec", default_value="20.0"),
            DeclareLaunchArgument("openxr_only_confidence_threshold", default_value="0.6"),
            DeclareLaunchArgument("source_dropout_grace_sec", default_value="0.15"),
            DeclareLaunchArgument("debug_log", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )
