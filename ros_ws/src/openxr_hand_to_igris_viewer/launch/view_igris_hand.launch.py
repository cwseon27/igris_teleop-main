import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_bool(context, name: str) -> bool:
    value = LaunchConfiguration(name).perform(context).strip().lower()
    return value in ("1", "true", "yes", "on")


def launch_setup(context, *args, **kwargs):
    urdf_path = LaunchConfiguration("urdf_path").perform(context)
    calibration_json = LaunchConfiguration("calibration_json").perform(context)
    hand_side = LaunchConfiguration("hand_side").perform(context)
    robot_hand_side = LaunchConfiguration("robot_hand_side").perform(context)
    pose_topic = LaunchConfiguration("pose_topic").perform(context)
    tracked_topic = LaunchConfiguration("tracked_topic").perform(context)
    fallback_pose_topic = LaunchConfiguration("fallback_pose_topic").perform(context)
    require_tracked = launch_bool(context, "require_tracked")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config").perform(context)

    with open(urdf_path, "r", encoding="utf-8") as f:
        robot_description = f.read()

    nodes = [
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
            name="vr_hand_to_igris_joint_state",
            output="screen",
            parameters=[
                {
                    "hand_side": hand_side,
                    "robot_hand_side": robot_hand_side,
                    "pose_topic": pose_topic,
                    "tracked_topic": tracked_topic,
                    "fallback_pose_topic": fallback_pose_topic,
                    "calibration_json": calibration_json,
                    "urdf_path": urdf_path,
                    "joint_states_topic": "/joint_states",
                    "require_tracked": require_tracked,
                    "smoothing_alpha": 0.35,
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

    return nodes


def generate_launch_description():
    package_share = get_package_share_directory("openxr_hand_to_igris_viewer")
    reliability_share = get_package_share_directory("igris_reliability_runtime")
    default_urdf = os.path.join(package_share, "urdf", "igris_c_v2_parallel_hand.urdf")
    default_rviz = os.path.join(package_share, "rviz", "igris_hand.rviz")
    default_calib = os.path.join(
        reliability_share,
        "config",
        "hand",
        "openxr_hand_calibration.json",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("urdf_path", default_value=default_urdf),
            DeclareLaunchArgument("calibration_json", default_value=default_calib),
            DeclareLaunchArgument("hand_side", default_value="right"),
            DeclareLaunchArgument("robot_hand_side", default_value="right"),
            DeclareLaunchArgument("pose_topic", default_value=""),
            DeclareLaunchArgument("tracked_topic", default_value=""),
            DeclareLaunchArgument("fallback_pose_topic", default_value=""),
            DeclareLaunchArgument("require_tracked", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            OpaqueFunction(function=launch_setup),
        ]
    )
