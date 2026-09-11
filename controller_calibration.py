#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OPENXR_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = Path("local_state/controller_to_chest_calibration")


@dataclass(frozen=True)
class ControllerPairSample:
    left: np.ndarray
    right: np.ndarray
    stamp: float


@dataclass(frozen=True)
class CalibrationResult:
    left_transform: np.ndarray
    right_transform: np.ndarray
    sample_count: int
    separation_mean_m: float
    separation_std_m: float
    chest_position_std_m: float


def _normalize(vector: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("cannot normalize near-zero vector")
    return np.asarray(vector, dtype=np.float64) / norm


def _project_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = (quat / norm).tolist()
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat(rotation: np.ndarray) -> np.ndarray:
    r = _project_rotation(rotation)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
            quat = np.array(
                [
                    0.25 * scale,
                    (r[0, 1] + r[1, 0]) / scale,
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[2, 1] - r[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
            quat = np.array(
                [
                    (r[0, 1] + r[1, 0]) / scale,
                    0.25 * scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    (r[0, 2] - r[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
            quat = np.array(
                [
                    (r[0, 2] + r[2, 0]) / scale,
                    (r[1, 2] + r[2, 1]) / scale,
                    0.25 * scale,
                    (r[1, 0] - r[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def _average_quaternions(quaternions: list[np.ndarray]) -> np.ndarray:
    if not quaternions:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    accum = np.zeros((4, 4), dtype=np.float64)
    for quat in quaternions:
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        accum += np.outer(q, q)
    _, vectors = np.linalg.eigh(accum)
    averaged = vectors[:, -1]
    if averaged[3] < 0.0:
        averaged *= -1.0
    return averaged / max(float(np.linalg.norm(averaged)), 1e-12)


def _mat_from_xyz_quat(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_rot(*quat_xyzw.tolist())
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    inverse = np.eye(4, dtype=np.float64)
    rotation_t = transform[:3, :3].T
    inverse[:3, :3] = rotation_t
    inverse[:3, 3] = -rotation_t @ transform[:3, 3]
    return inverse


def _pose_msg_to_mat(pose: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_rot(
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )
    transform[:3, 3] = np.array(
        [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        dtype=np.float64,
    )
    return transform


def build_horizontal_chest_pose(
    left_controller: np.ndarray,
    right_controller: np.ndarray,
    *,
    min_separation_m: float,
) -> np.ndarray:
    left_pos = np.asarray(left_controller, dtype=np.float64).reshape(4, 4)[:3, 3]
    right_pos = np.asarray(right_controller, dtype=np.float64).reshape(4, 4)[:3, 3]
    midpoint = 0.5 * (left_pos + right_pos)

    up = OPENXR_UP.copy()
    x_right = right_pos - left_pos
    x_right = x_right - up * float(np.dot(x_right, up))
    separation = float(np.linalg.norm(x_right))
    if separation < min_separation_m:
        raise ValueError(
            f"controller horizontal separation is too small ({separation:.4f} m)"
        )

    x_right = _normalize(x_right)
    z_back = _normalize(np.cross(x_right, up))
    y_up = _normalize(np.cross(z_back, x_right))

    chest = np.eye(4, dtype=np.float64)
    chest[:3, :3] = np.column_stack((x_right, y_up, z_back))
    chest[:3, 3] = midpoint
    return chest


def _average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    positions = np.stack([transform[:3, 3] for transform in transforms], axis=0)
    quats = [_rot_to_quat(transform[:3, :3]) for transform in transforms]
    return _mat_from_xyz_quat(np.median(positions, axis=0), _average_quaternions(quats))


def calibrate_controller_to_chest(
    samples: list[ControllerPairSample],
    *,
    min_separation_m: float,
) -> CalibrationResult:
    left_relatives: list[np.ndarray] = []
    right_relatives: list[np.ndarray] = []
    chest_positions: list[np.ndarray] = []
    separations: list[float] = []

    for sample in samples:
        chest = build_horizontal_chest_pose(
            sample.left,
            sample.right,
            min_separation_m=min_separation_m,
        )
        left_relatives.append(_invert_transform(sample.left) @ chest)
        right_relatives.append(_invert_transform(sample.right) @ chest)
        chest_positions.append(chest[:3, 3].copy())
        separations.append(float(np.linalg.norm(sample.right[:3, 3] - sample.left[:3, 3])))

    if not left_relatives:
        raise ValueError("no valid controller samples were collected")

    chest_position_arr = np.stack(chest_positions, axis=0)
    return CalibrationResult(
        left_transform=_average_transforms(left_relatives),
        right_transform=_average_transforms(right_relatives),
        sample_count=len(left_relatives),
        separation_mean_m=float(np.mean(separations)),
        separation_std_m=float(np.std(separations)),
        chest_position_std_m=float(np.mean(np.std(chest_position_arr, axis=0))),
    )


def transform_to_xyz_xyzw(transform: np.ndarray) -> list[float]:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    quat = _rot_to_quat(transform[:3, :3])
    return [
        float(transform[0, 3]),
        float(transform[1, 3]),
        float(transform[2, 3]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]


def _format_float(value: float) -> str:
    value = 0.0 if abs(value) < 5e-10 else float(value)
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def format_transform_for_env(transform: np.ndarray) -> str:
    return ",".join(_format_float(value) for value in transform_to_xyz_xyzw(transform))


def _write_outputs(result: CalibrationResult, output_prefix: Path) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    env_path = output_prefix.with_suffix(".env")
    json_path = output_prefix.with_suffix(".json")

    left_env = format_transform_for_env(result.left_transform)
    right_env = format_transform_for_env(result.right_transform)
    env_path.write_text(
        "\n".join(
            [
                "# Generated by controller_calibration.py",
                f'export IGRIS_LEFT_CONTROLLER_TO_CHEST="{left_env}"',
                f'export IGRIS_RIGHT_CONTROLLER_TO_CHEST="{right_env}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "generated_by": "controller_calibration.py",
                "format": "x,y,z,qx,qy,qz,qw",
                "left_controller_to_chest": transform_to_xyz_xyzw(result.left_transform),
                "right_controller_to_chest": transform_to_xyz_xyzw(result.right_transform),
                "sample_count": result.sample_count,
                "separation_mean_m": result.separation_mean_m,
                "separation_std_m": result.separation_std_m,
                "chest_position_std_m": result.chest_position_std_m,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return env_path, json_path


def collect_samples(args: argparse.Namespace) -> list[ControllerPairSample]:
    try:
        import rclpy
        from geometry_msgs.msg import Pose, PoseStamped
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        ros_setup = PROJECT_ROOT / "ros_ws" / "install" / "setup.bash"
        runtime_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        raise SystemExit(
            "ROS2 Python modules are not available. Run after sourcing ROS, e.g.\n"
            "  source /opt/ros/jazzy/setup.bash\n"
            f"  source {ros_setup}\n"
            f"  {runtime_python} {PROJECT_ROOT / 'controller_calibration.py'}"
        ) from exc

    class ControllerCalibrationNode(Node):
        def __init__(self) -> None:
            super().__init__("igris_controller_to_chest_calibration")
            self.left_pose: np.ndarray | None = None
            self.right_pose: np.ndarray | None = None
            self.left_stamp = 0.0
            self.right_stamp = 0.0
            self.samples: list[ControllerPairSample] = []
            self.start_mono = time.monotonic()
            self.collect_start_mono = self.start_mono + float(args.prep_delay_sec)
            self.collect_end_mono = self.collect_start_mono + float(args.sample_duration_sec)
            self.done = False
            self.last_status_second: int | None = None
            self.left_count = 0
            self.right_count = 0
            self.left_msg_type = self._resolve_pose_message_type(args.left_topic)
            self.right_msg_type = self._resolve_pose_message_type(args.right_topic)
            self.qos_profile = self._make_qos_profile()
            self.create_subscription(
                self.left_msg_type,
                args.left_topic,
                self._on_left_pose,
                self.qos_profile,
            )
            self.create_subscription(
                self.right_msg_type,
                args.right_topic,
                self._on_right_pose,
                self.qos_profile,
            )
            self.create_timer(1.0 / max(1.0, float(args.sample_rate_hz)), self._tick)

        def _resolve_pose_message_type(self, topic: str):
            topic_types = dict(self.get_topic_names_and_types())
            types = topic_types.get(topic, [])
            if "geometry_msgs/msg/PoseStamped" in types:
                return PoseStamped
            if "geometry_msgs/msg/Pose" in types:
                return Pose
            if types:
                print(
                    f"[calibration] WARNING: {topic} type={types}; "
                    "PoseStamped로 구독을 시도합니다."
                )
            return PoseStamped

        def _make_qos_profile(self) -> QoSProfile:
            reliability = (
                ReliabilityPolicy.RELIABLE
                if args.qos == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            )
            return QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=max(1, int(args.qos_depth)),
                reliability=reliability,
                durability=DurabilityPolicy.VOLATILE,
            )

        @staticmethod
        def _pose_from_message(msg):
            return msg.pose if hasattr(msg, "pose") else msg

        def _on_left_pose(self, msg) -> None:
            self.left_pose = _pose_msg_to_mat(self._pose_from_message(msg))
            self.left_stamp = time.monotonic()
            self.left_count += 1

        def _on_right_pose(self, msg) -> None:
            self.right_pose = _pose_msg_to_mat(self._pose_from_message(msg))
            self.right_stamp = time.monotonic()
            self.right_count += 1

        @staticmethod
        def _age_text(now: float, stamp: float, count: int) -> str:
            if count <= 0:
                return "never"
            return f"{now - stamp:0.2f}s"

        def _print_status_once_per_second(self, now: float) -> None:
            second = int(now - self.start_mono)
            if self.last_status_second == second:
                return
            self.last_status_second = second
            rx_status = (
                f"left_rx={self.left_count} age={self._age_text(now, self.left_stamp, self.left_count)}, "
                f"right_rx={self.right_count} age={self._age_text(now, self.right_stamp, self.right_count)}"
            )
            if now < self.collect_start_mono:
                remain = max(0.0, self.collect_start_mono - now)
                print(
                    f"[calibration] 준비 시간: {remain:0.1f}s 후 샘플링 시작 "
                    f"({rx_status})"
                )
            else:
                remain = max(0.0, self.collect_end_mono - now)
                print(
                    f"[calibration] 샘플링 중: 남은 시간 {remain:0.1f}s, "
                    f"samples={len(self.samples)} ({rx_status})"
                )

        def _tick(self) -> None:
            now = time.monotonic()
            self._print_status_once_per_second(now)
            if now < self.collect_start_mono:
                return
            if now >= self.collect_end_mono:
                self.done = True
                return
            if self.left_pose is None or self.right_pose is None:
                return
            if now - self.left_stamp > args.max_stale_sec:
                return
            if now - self.right_stamp > args.max_stale_sec:
                return
            self.samples.append(
                ControllerPairSample(
                    left=self.left_pose.copy(),
                    right=self.right_pose.copy(),
                    stamp=now,
                )
            )

    rclpy.init(args=None)
    node = ControllerCalibrationNode()
    print("[calibration] left topic :", args.left_topic)
    print("[calibration] right topic:", args.right_topic)
    print("[calibration] left type  :", node.left_msg_type.__name__)
    print("[calibration] right type :", node.right_msg_type.__name__)
    print("[calibration] qos        :", args.qos)
    print(
        "[calibration] 시작 후 "
        f"{args.prep_delay_sec:g}s 대기, 이후 {args.sample_duration_sec:g}s 동안 샘플링합니다."
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        samples = list(node.samples)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return samples


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect left/right OpenXR controller PoseStamped topics and compute "
            "IGRIS controller-to-chest calibration transforms."
        )
    )
    parser.add_argument("--left-topic", default="/left_controller/poses")
    parser.add_argument("--right-topic", default="/right_controller/poses")
    parser.add_argument("--prep-delay-sec", type=float, default=5.0)
    parser.add_argument("--sample-duration-sec", type=float, default=10.0)
    parser.add_argument("--sample-rate-hz", type=float, default=60.0)
    parser.add_argument("--max-stale-sec", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-separation-m", type=float, default=0.05)
    parser.add_argument("--qos-depth", type=int, default=10)
    parser.add_argument(
        "--qos",
        choices=("best_effort", "reliable"),
        default="best_effort",
        help="Subscriber QoS reliability. Streaming controller topics commonly need best_effort.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Path without suffix. Writes .env and .json. Default: local_state/controller_to_chest_calibration",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    samples = collect_samples(args)
    if len(samples) < args.min_samples:
        print(
            f"[calibration] ERROR: valid paired samples {len(samples)} < "
            f"required {args.min_samples}",
            file=sys.stderr,
        )
        print(
            "[calibration] controller 토픽이 publish 중인지, 양쪽 pose가 동시에 들어오는지 확인하세요.",
            file=sys.stderr,
        )
        return 2

    try:
        result = calibrate_controller_to_chest(
            samples,
            min_separation_m=float(args.min_separation_m),
        )
    except Exception as exc:
        print(f"[calibration] ERROR: {exc}", file=sys.stderr)
        return 3

    env_path, json_path = _write_outputs(result, Path(args.output_prefix))
    left_env = format_transform_for_env(result.left_transform)
    right_env = format_transform_for_env(result.right_transform)

    print()
    print("[calibration] 완료")
    print(f"[calibration] samples: {result.sample_count}")
    print(f"[calibration] controller separation mean/std: {result.separation_mean_m:.4f} / {result.separation_std_m:.4f} m")
    print(f"[calibration] chest position std: {result.chest_position_std_m:.5f} m")
    print()
    print('export IGRIS_LEFT_CONTROLLER_TO_CHEST="' + left_env + '"')
    print('export IGRIS_RIGHT_CONTROLLER_TO_CHEST="' + right_env + '"')
    print()
    print(f"[calibration] saved env : {env_path}")
    print(f"[calibration] saved json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
