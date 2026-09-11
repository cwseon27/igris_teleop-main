#!/usr/bin/env python3
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
PACKAGE_NAME = "openxr_hand_to_igris_viewer"
DEFAULT_URDF_NAME = "igris_c_v2_parallel_hand.urdf"

# Unity XROriginToROSPublisher.cs 기준:
# poses[0] = wrist
# poses[1] = thumb_tip
# poses[2] = index_tip
# poses[3] = middle_tip
# poses[4] = ring_tip
# poses[5] = little_tip
FINGER_TIP_INDEX = {
    "thumb": 1,
    "index": 2,
    "middle": 3,
    "ring": 4,
    "little": 5,
}


def default_urdf_path() -> str:
    try:
        package_share = Path(get_package_share_directory(PACKAGE_NAME))
        return str(package_share / "urdf" / DEFAULT_URDF_NAME)
    except Exception:
        return str(Path(__file__).resolve().parents[1] / "urdf" / DEFAULT_URDF_NAME)


def default_calibration_path() -> str:
    try:
        package_share = Path(get_package_share_directory("igris_reliability_runtime"))
        return str(package_share / "config" / "hand" / "openxr_hand_calibration.json")
    except Exception:
        for parent in Path(__file__).resolve().parents:
            candidate = (
                parent
                / "ros_ws"
                / "src"
                / "igris_reliability_runtime"
                / "config"
                / "hand"
                / "openxr_hand_calibration.json"
            )
            if candidate.is_file():
                return str(candidate)
        return ""


class VRHandToIgrisJointState(Node):
    def __init__(self):
        super().__init__("vr_hand_to_igris_joint_state")

        self.declare_parameter("hand_side", "right")          # VR topic side: right or left
        self.declare_parameter("robot_hand_side", "right")    # URDF hand side: right or left
        self.declare_parameter("pose_topic", "")
        self.declare_parameter("fallback_pose_topic", "")
        self.declare_parameter("tracked_topic", "")
        self.declare_parameter(
            "calibration_json",
            default_calibration_path(),
        )
        self.declare_parameter(
            "urdf_path",
            default_urdf_path(),
        )
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("normalized_topic", "")
        self.declare_parameter("require_tracked", True)
        self.declare_parameter("smoothing_alpha", 0.35)
        self.declare_parameter("publish_all_urdf_joints", True)
        self.declare_parameter("debug_log", False)

        self.hand_side = self.get_parameter("hand_side").value.lower()
        self.robot_hand_side = self.get_parameter("robot_hand_side").value.lower()

        pose_topic_param = self.get_parameter("pose_topic").value
        fallback_pose_topic_param = self.get_parameter("fallback_pose_topic").value
        tracked_topic_param = self.get_parameter("tracked_topic").value
        normalized_topic_param = self.get_parameter("normalized_topic").value

        self.pose_topic = pose_topic_param or f"/{self.hand_side}_hand/poses"
        self.fallback_pose_topic = (
            fallback_pose_topic_param or f"/{self.hand_side}_mediapipe_hand/poses"
        )
        self.tracked_topic = tracked_topic_param or f"/{self.hand_side}_hand/is_tracked"
        self.normalized_topic = normalized_topic_param or f"/{self.hand_side}_hand/finger_normalized"
        self.joint_states_topic = self.get_parameter("joint_states_topic").value

        self.calibration_path = Path(self.get_parameter("calibration_json").value).expanduser()
        self.urdf_path = Path(self.get_parameter("urdf_path").value).expanduser()

        self.require_tracked = bool(self.get_parameter("require_tracked").value)
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self.smoothing_alpha = max(0.0, min(1.0, self.smoothing_alpha))
        self.publish_all_urdf_joints = bool(self.get_parameter("publish_all_urdf_joints").value)
        self.debug_log = bool(self.get_parameter("debug_log").value)

        self.is_tracked = None
        self.last_normalized = None
        self.active_pose_source = None

        self.calib = self.load_calibration(self.calibration_path)
        self.all_joint_names, self.joint_limits = self.parse_urdf(self.urdf_path)

        self.robot_joint_map = self.make_robot_joint_map(self.robot_hand_side)

        self.pose_sub = self.create_subscription(
            PoseArray,
            self.pose_topic,
            self.on_primary_pose_array,
            10,
        )

        self.fallback_pose_sub = self.create_subscription(
            PoseArray,
            self.fallback_pose_topic,
            self.on_fallback_pose_array,
            10,
        )

        self.tracked_sub = self.create_subscription(
            Bool,
            self.tracked_topic,
            self.on_tracked,
            10,
        )

        self.normalized_pub = self.create_publisher(
            Float32MultiArray,
            self.normalized_topic,
            10,
        )

        self.joint_state_pub = self.create_publisher(
            JointState,
            self.joint_states_topic,
            10,
        )
        self.open_state_timer = self.create_timer(0.1, self.publish_open_state_until_input)

        self.get_logger().info("VR hand to IGRIS hand viewer node started.")
        self.get_logger().info(f"pose_topic         : {self.pose_topic}")
        self.get_logger().info(f"fallback_pose_topic: {self.fallback_pose_topic}")
        self.get_logger().info(f"tracked_topic      : {self.tracked_topic}")
        self.get_logger().info(f"normalized_topic   : {self.normalized_topic}")
        self.get_logger().info(f"joint_states_topic : {self.joint_states_topic}")
        self.get_logger().info(f"calibration_json   : {self.calibration_path}")
        self.get_logger().info(f"urdf_path          : {self.urdf_path}")
        self.get_logger().info(f"robot_hand_side    : {self.robot_hand_side}")
        self.get_logger().info(f"require_tracked    : {self.require_tracked}")

    def load_calibration(self, path: Path) -> Dict[str, float]:
        if not path.exists():
            raise FileNotFoundError(f"Calibration JSON not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        required = []
        for finger in FINGER_NAMES:
            required.append(f"{finger}_max")
            required.append(f"{finger}_min")

        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Calibration JSON missing keys: {missing}")

        calib = {k: float(data[k]) for k in required}

        self.get_logger().info("Loaded calibration values:")
        for finger in FINGER_NAMES:
            self.get_logger().info(
                f"  {finger:6s}: max={calib[f'{finger}_max']:.6f}, min={calib[f'{finger}_min']:.6f}"
            )

        return calib

    def parse_urdf(self, path: Path) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
        if not path.exists():
            raise FileNotFoundError(f"URDF not found: {path}")

        root = ET.parse(str(path)).getroot()

        joint_names = []
        joint_limits = {}

        for joint in root.findall("joint"):
            name = joint.attrib.get("name", "")
            joint_type = joint.attrib.get("type", "")

            if joint_type == "fixed":
                continue

            if not name:
                continue

            joint_names.append(name)

            limit = joint.find("limit")
            if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])
                joint_limits[name] = (lower, upper)

        return joint_names, joint_limits

    def make_robot_joint_map(self, side: str) -> Dict[str, List[str]]:
        prefix = "Right" if side == "right" else "Left"

        return {
            "thumb": [
                f"{prefix}_0_Joint_Thumb_Proximal",
                f"{prefix}_1_Joint_Thumb_Middle",
                f"{prefix}_2_Joint_Thumb_Distal",
            ],
            "index": [
                f"{prefix}_3_Joint_Index_Middle",
                f"{prefix}_4_Joint_Index_Distal",
            ],
            "middle": [
                f"{prefix}_5_Joint_Middle_Middle",
                f"{prefix}_6_Joint_Middle_Distal",
            ],
            "ring": [
                f"{prefix}_7_Joint_Ring_Middle",
                f"{prefix}_8_Joint_Ring_Distal",
            ],
            "little": [
                f"{prefix}_9_Joint_Little_Middle",
                f"{prefix}_10_Joint_Little_Distal",
            ],
        }

    def on_tracked(self, msg: Bool):
        self.is_tracked = bool(msg.data)

    def on_primary_pose_array(self, msg: PoseArray):
        if self.require_tracked and self.is_tracked is False:
            return

        self.process_pose_array(msg, "openxr")

    def on_fallback_pose_array(self, msg: PoseArray):
        if self.is_tracked is not False:
            return

        self.process_pose_array(msg, "mediapipe")

    def process_pose_array(self, msg: PoseArray, source: str):
        if len(msg.poses) < 6:
            self.get_logger().warn(f"Expected 6 hand poses, got {len(msg.poses)}")
            return

        normalized = self.compute_normalized_from_pose_array(msg)

        if source != self.active_pose_source:
            self.active_pose_source = source
            self.last_normalized = None
            self.get_logger().info(f"pose_source        : {source}")

        if self.last_normalized is None or self.smoothing_alpha <= 0.0:
            filtered = normalized
        else:
            a = self.smoothing_alpha
            filtered = [
                (a * normalized[i]) + ((1.0 - a) * self.last_normalized[i])
                for i in range(len(normalized))
            ]

        self.last_normalized = filtered

        self.publish_normalized(filtered)
        self.publish_joint_state(filtered)

        if self.debug_log:
            self.get_logger().info(
                "normalized: " + ", ".join(
                    f"{FINGER_NAMES[i]}={filtered[i]:.3f}" for i in range(5)
                )
            )

    def publish_open_state_until_input(self):
        if self.last_normalized is not None:
            self.open_state_timer.cancel()
            return

        self.publish_joint_state([0.0] * len(FINGER_NAMES))

    def compute_normalized_from_pose_array(self, msg: PoseArray) -> List[float]:
        result = []

        for finger in FINGER_NAMES:
            idx = FINGER_TIP_INDEX[finger]
            p = msg.poses[idx].position

            # Unity에서 fingertip이 이미 wrist 기준 상대좌표라면,
            # 현재 fingertip vector의 norm이 wrist-to-tip 거리와 같다.
            distance = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)

            max_dist = self.calib[f"{finger}_max"]  # open hand
            min_dist = self.calib[f"{finger}_min"]  # closed hand
            denom = max_dist - min_dist

            if abs(denom) < 1e-9:
                s = 0.0
            else:
                # distance가 max에 가까우면 0, min에 가까우면 1
                s = (max_dist - distance) / denom

            s = max(0.0, min(1.0, s))
            result.append(s)

        return result

    def publish_normalized(self, normalized: List[float]):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in normalized]
        self.normalized_pub.publish(msg)

    def publish_joint_state(self, normalized: List[float]):
        positions_by_name = {}

        # 전체 URDF movable joints를 같이 publish하면 robot_state_publisher가
        # 손까지 이어지는 전체 TF chain을 안정적으로 만들 수 있다.
        if self.publish_all_urdf_joints:
            for name in self.all_joint_names:
                positions_by_name[name] = 0.0

        for finger_idx, finger in enumerate(FINGER_NAMES):
            s = float(normalized[finger_idx])

            for joint_name in self.robot_joint_map[finger]:
                lower, upper = self.joint_limits.get(joint_name, (0.0, 0.0))

                q_open = 0.0
                q_open = max(lower, min(upper, q_open))

                q_closed = self.get_closed_position(joint_name, lower, upper)

                q = q_open + s * (q_closed - q_open)
                q = max(lower, min(upper, q))

                positions_by_name[joint_name] = q

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()

        # stable order
        if self.publish_all_urdf_joints:
            js.name = list(self.all_joint_names)
        else:
            js.name = []
            for finger in FINGER_NAMES:
                js.name.extend(self.robot_joint_map[finger])

        js.position = [float(positions_by_name.get(name, 0.0)) for name in js.name]

        self.joint_state_pub.publish(js)

    def get_closed_position(self, joint_name: str, lower: float, upper: float) -> float:
        # IGRIS URDF 기준:
        # 대부분 hand joint는 0 -> open, upper -> closed.
        # 단, Right_0_Joint_Thumb_Proximal은 lower가 음수이고 upper가 0이라서
        # 0 -> open, lower -> closed로 둔다.
        if joint_name == "Right_0_Joint_Thumb_Proximal":
            return lower

        return upper


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = VRHandToIgrisJointState()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
