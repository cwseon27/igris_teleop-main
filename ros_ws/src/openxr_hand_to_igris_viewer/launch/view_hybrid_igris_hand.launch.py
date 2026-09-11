import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_float(context, name: str) -> float:
    return float(LaunchConfiguration(name).perform(context))


def launch_bool(context, name: str) -> bool:
    value = LaunchConfiguration(name).perform(context).strip().lower()
    return value in ("1", "true", "yes", "on")


def launch_setup(context, *args, **kwargs):
    hand_side = LaunchConfiguration("hand_side").perform(context).lower()
    if hand_side not in ("right", "left"):
        raise ValueError("view_hybrid_igris_hand.launch.py supports hand_side:=right or hand_side:=left. Use hybrid_hand_pose_bridge.launch.py for hand_side:=both bridge-only output.")

    robot_hand_side = LaunchConfiguration("robot_hand_side").perform(context)
    urdf_path = LaunchConfiguration("urdf_path").perform(context)
    calibration_json = LaunchConfiguration("calibration_json").perform(context)
    mediapipe_calibration_json = LaunchConfiguration("mediapipe_calibration_json").perform(context)
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config").perform(context)

    hybrid_pose_topic = f"/{hand_side}_hybrid_hand/poses"
    hybrid_tracked_topic = f"/{hand_side}_hybrid_hand/is_tracked"

    with open(urdf_path, "r", encoding="utf-8") as f:
        robot_description = f.read()

    return [
        Node(
            package="openxr_hand_to_igris_viewer",
            executable="hybrid_openxr_mediapipe_pose_bridge",
            name=f"{hand_side}_hybrid_openxr_mediapipe_pose_bridge",
            output="screen",
            parameters=[
                {
                    "hand_side": hand_side,
                    "apply_calibration_remap": launch_bool(context, "apply_calibration_remap"),
                    "openxr_calibration_json": calibration_json,
                    "mediapipe_calibration_json": mediapipe_calibration_json,
                    "output_calibration_json": calibration_json,
                    "publish_rate_hz": launch_float(context, "publish_rate_hz"),
                    "pose_stale_sec": launch_float(context, "pose_stale_sec"),
                    "confidence_stale_sec": launch_float(context, "confidence_stale_sec"),
                    "tracked_stale_sec": launch_float(context, "tracked_stale_sec"),
                    "confidence_gamma": launch_float(context, "confidence_gamma"),
                    "debug_log": launch_bool(context, "debug_log"),
                    "output_pose_topic": hybrid_pose_topic,
                    "output_tracked_topic": hybrid_tracked_topic,
                }
            ],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[
                {
                    "robot_description": robot_description,
                    "use_sim_time": False,
                }
            ],
        ),
        Node(
            package="openxr_hand_to_igris_viewer",
            executable="vr_hand_to_joint_state",
            name="hybrid_vr_hand_to_igris_joint_state",
            output="screen",
            parameters=[
                {
                    "hand_side": hand_side,
                    "robot_hand_side": robot_hand_side,
                    "pose_topic": hybrid_pose_topic,
                    "tracked_topic": hybrid_tracked_topic,
                    "fallback_pose_topic": "",
                    "calibration_json": calibration_json,
                    "urdf_path": urdf_path,
                    "joint_states_topic": "/joint_states",
                    "require_tracked": True,
                    "smoothing_alpha": launch_float(context, "smoothing_alpha"),
                    "publish_all_urdf_joints": True,
                }
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(rviz),
        ),
    ]


def generate_launch_description():
    package_share = get_package_share_directory("openxr_hand_to_igris_viewer")
    reliability_share = get_package_share_directory("igris_reliability_runtime")
    calibration_dir = os.path.join(reliability_share, "config", "hand")
    default_urdf = os.path.join(package_share, "urdf", "igris_c_v2_parallel_hand.urdf")
    default_rviz = os.path.join(package_share, "rviz", "igris_hand.rviz")
    default_calib = os.path.join(calibration_dir, "openxr_hand_calibration.json")
    default_mediapipe_calib = os.path.join(calibration_dir, "mediapipe_hand_calibration.json")

    return LaunchDescription(
        [
            DeclareLaunchArgument("hand_side", default_value="right"),
            DeclareLaunchArgument("robot_hand_side", default_value="right"),
            DeclareLaunchArgument("urdf_path", default_value=default_urdf),
            DeclareLaunchArgument("calibration_json", default_value=default_calib),
            DeclareLaunchArgument("mediapipe_calibration_json", default_value=default_mediapipe_calib),
            DeclareLaunchArgument("apply_calibration_remap", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            DeclareLaunchArgument("publish_rate_hz", default_value="30.0"),
            DeclareLaunchArgument("pose_stale_sec", default_value="0.2"),
            DeclareLaunchArgument("confidence_stale_sec", default_value="0.5"),
            DeclareLaunchArgument("tracked_stale_sec", default_value="0.5"),
            DeclareLaunchArgument("confidence_gamma", default_value="1.0"),
            DeclareLaunchArgument("smoothing_alpha", default_value="0.35"),
            DeclareLaunchArgument("debug_log", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )
