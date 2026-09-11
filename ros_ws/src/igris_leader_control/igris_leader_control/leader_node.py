import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from dynamixel_sdk import *
import os
import time

from .timing import LeaderTimingWindow


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except ValueError:
        return float(default)


class IgrisLeaderNode(Node):
    def __init__(self):
        super().__init__('igris_leader_node')
        
        # --- 설정 ---
        default_device_name = os.getenv("IGRIS_LEADER_DEVICE", "/dev/ttyUSB0")
        self.declare_parameter("device_name", default_device_name)
        self.DEVICE_NAME = str(self.get_parameter("device_name").value)
        self.BAUDRATE = 1000000
        self.ADDR_PRESENT_POSITION = 132
        self.LEN_PRESENT_POSITION = 4
        
        # 1. [Static Joints] 고정 관절
        self.static_joints = [
            "0_Joint_Waist_Yaw", "1_Joint_Waist_Roll", "2_Joint_Waist_Pitch",
            "3_Joint_Hip_Pitch_Left", "4_Joint_Hip_Roll_Left", "5_Joint_Hip_Yaw_Left",
            "6_Joint_Knee_Pitch_Left", "7_Joint_Ankle_Pitch_Left", "8_Joint_Ankle_Roll_Left",
            "9_Joint_Hip_Pitch_Right", "10_Joint_Hip_Roll_Right", "11_Joint_Hip_Yaw_Right",
            "12_Joint_Knee_Pitch_Right", "13_Joint_Ankle_Pitch_Right", "14_Joint_Ankle_Roll_Right",
            "29_Joint_Neck_Yaw", "30_Joint_Neck_Pitch"
        ]

        env_hand_enabled = _env_flag("IGRIS_LEADER_HAND", default=False)
        env_hand_autodetect = _env_flag("IGRIS_LEADER_HAND_AUTODETECT", default=True)
        self.declare_parameter("hand_enabled", env_hand_enabled)
        self.declare_parameter("hand_autodetect", env_hand_autodetect)
        self.hand_enabled = bool(self.get_parameter("hand_enabled").value)
        self.hand_autodetect = bool(self.get_parameter("hand_autodetect").value)
        self.dashboard_enabled = _env_flag("IGRIS_LEADER_DASHBOARD", default=False)
        default_publish_hz = max(20.0, min(200.0, _env_float("IGRIS_LEADER_PUBLISH_HZ", 100.0)))
        self.declare_parameter("publish_hz", default_publish_hz)
        self.publish_hz = max(20.0, min(200.0, float(self.get_parameter("publish_hz").value)))
        default_timing_interval_s = _env_float(
            "IGRIS_LEADER_TIMING_INTERVAL_S", 10.0
        )
        if default_timing_interval_s > 0.0:
            default_timing_interval_s = max(
                1.0, min(300.0, default_timing_interval_s)
            )
        self.declare_parameter(
            "timing_report_interval_s", default_timing_interval_s
        )
        self.timing_report_interval_s = float(
            self.get_parameter("timing_report_interval_s").value
        )
        if self.timing_report_interval_s > 0.0:
            self.timing_report_interval_s = max(
                1.0, min(300.0, self.timing_report_interval_s)
            )

        # 2. [Arm Motors] 실제 연결된 팔 모터
        self.arm_joint_map = {
            # Right Arm (ID 11~17)
            11: "22_Joint_Shoulder_Pitch_Right", 12: "23_Joint_Shoulder_Roll_Right",
            13: "24_Joint_Shoulder_Yaw_Right",   14: "25_Joint_Elbow_Pitch_Right",
            15: "26_Joint_Wrist_Yaw_Right",      
            16: "27_Joint_Wrist_Roll_Right",     # [Coupled A]
            17: "28_Joint_Wrist_Pitch_Right",    # [Coupled B]
            
            # Left Arm (ID 21~27)
            21: "15_Joint_Shoulder_Pitch_Left",  22: "16_Joint_Shoulder_Roll_Left",
            23: "17_Joint_Shoulder_Yaw_Left",    24: "18_Joint_Elbow_Pitch_Left",
            25: "19_Joint_Wrist_Yaw_Left",       
            26: "20_Joint_Wrist_Roll_Left",      # [Coupled A]
            27: "21_Joint_Wrist_Pitch_Left",     # [Coupled B]
        }

        self.hand_joint_map = {
            18: "Finger_T_R", 19: "Finger_F_R",
            28: "Finger_T_L", 29: "Finger_F_L"
        }
        self.hand_candidates = {
            "right": [18, 19],
            "left": [28, 29],
        }
        self.hand_joint_side = {
            18: "right", 19: "right",
            28: "left", 29: "left",
        }
        self.hand_role_names = {
            "right": {"thumb": "Finger_T_R", "fingers": "Finger_F_R"},
            "left": {"thumb": "Finger_T_L", "fingers": "Finger_F_L"},
        }
        self.hand_role_sources = {
            "right": {"thumb": None, "fingers": None},
            "left": {"thumb": None, "fingers": None},
        }
        self.hand_calibration = {
            "right": {"open": -1.54, "close": -3.14},
            "left": {"open": -1.59, "close": 0.0},
        }

        self.joint_map = dict(self.arm_joint_map)
        self.dummy_joints = {}
        self.dummy_joint_sources = {}
        self.hand_status = "Dummy Values: 0.00"
        self.hand_mode = "DISABLED (publishing dummy hand joints)"

        # SDK가 통신할 ID 리스트 (여기에는 실제 존재하는 것만 들어감)
        self.ids = list(self.arm_joint_map.keys())

        # Direction
        self.directions = {
            11: 1, 12: -1, 13: 1, 14: 1, 15: 1, 16: 1, 17: -1,
            18: 1, 19: 1,
            21: -1, 22: -1, 23: 1, 24: -1, 25: 1, 26: 1, 27: 1,
            28: 1, 29: 1,
        }

        if not os.path.exists(self.DEVICE_NAME):
            raise RuntimeError(
                f"Leader serial device does not exist: {self.DEVICE_NAME}. "
                "Set IGRIS_LEADER_DEVICE or the ROS device_name parameter."
            )
        if not os.access(self.DEVICE_NAME, os.R_OK | os.W_OK):
            current_user = os.getenv("USER", "<user>")
            raise PermissionError(
                f"No read/write permission for leader serial device {self.DEVICE_NAME}. "
                f"Immediate fix: sudo setfacl -m u:{current_user}:rw {self.DEVICE_NAME}. "
                f"Permanent fix: sudo usermod -aG dialout {current_user}, then log out and back in."
            )

        # SDK Init
        self.port_handler = PortHandler(self.DEVICE_NAME)
        self.packet_handler = PacketHandler(2.0)
        self.group_sync_read = GroupSyncRead(self.port_handler, self.packet_handler, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)

        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open leader serial port {self.DEVICE_NAME}")
        if not self.port_handler.setBaudRate(self.BAUDRATE):
            raise RuntimeError(
                f"Failed to set leader baudrate {self.BAUDRATE} on {self.DEVICE_NAME}"
            )

        if self.hand_enabled:
            self._configure_hand_joints()
        else:
            # hand off 시에는 미연결 모터를 통신 대상에서 제외하고, 토픽에만 더미로 유지
            self.dummy_joints = dict(self.hand_joint_map)
            self.dummy_joint_sources = {dxl_id: None for dxl_id in self.dummy_joints}

        # 값 저장소는 손 전체 ID를 항상 포함해서 대시보드/더미 출력을 단순화한다.
        self.current_vals = {
            dxl_id: 0.0 for dxl_id in (*self.arm_joint_map.keys(), *self.hand_joint_map.keys())
        }
        self._comm_connected = False
        self._last_comm_error_code = None
        self._last_missing_optional_hand_ids = None
        self._last_drop_detail = None
        timing_sample_capacity = min(
            10000,
            max(
                512,
                int(
                    self.publish_hz
                    * max(self.timing_report_interval_s, 10.0)
                    * 1.25
                ),
            ),
        )
        self._timing = LeaderTimingWindow(max_samples=timing_sample_capacity)
        self._last_callback_started_ns = None
        self._last_timing_report_ns = time.perf_counter_ns()

        # [중요] 실제 존재하는 ID만 SyncRead 파라미터로 추가
        for dxl_id in self.ids:
            self.group_sync_read.addParam(dxl_id)
        self._monitored_ids = ", ".join(str(dxl_id) for dxl_id in self.ids)

        # Publisher
        # Keep only the newest sample throughout the leader path. At 1 Mbps,
        # 100 Hz leaves generous serial time while halving command age.
        self.joint_publisher_ = self.create_publisher(JointState, 'joint_states', 1)
        self.timer = self.create_timer(1.0 / self.publish_hz, self.publish_data)
        
        if self.dashboard_enabled:
            print("\033[H\033[J")
            self.get_logger().info("Leader dashboard enabled via IGRIS_LEADER_DASHBOARD")
        self.get_logger().info(f"Leader hand mode: {self.hand_mode}")
        self.get_logger().info(f"Leader arm publish rate: {self.publish_hz:.1f} Hz")
        if self.timing_report_interval_s > 0.0:
            self.get_logger().info(
                "Leader timing diagnostics enabled: "
                f"{self.timing_report_interval_s:.1f} s windows"
            )

    def _is_motor_available(self, dxl_id):
        _, comm_result, _ = self.packet_handler.ping(self.port_handler, dxl_id)
        return comm_result == COMM_SUCCESS

    def _configure_hand_joints(self):
        live_hand_joint_map = {}
        dashboard_tokens = []
        mode_tokens = []

        for side, candidate_ids in self.hand_candidates.items():
            # Even in fixed-ID mode, verify optional hand motors before adding
            # them to the sync-read group. A missing hand ID can make the whole
            # Dynamixel broadcast read time out and would otherwise block arm
            # joint publishing. Arm IDs remain mandatory and are handled in
            # publish_data().
            detected_ids = [dxl_id for dxl_id in candidate_ids if self._is_motor_available(dxl_id)]
            thumb_name = self.hand_role_names[side]["thumb"]
            finger_name = self.hand_role_names[side]["fingers"]

            if len(detected_ids) >= 2:
                thumb_id, finger_id = candidate_ids
                live_hand_joint_map[thumb_id] = thumb_name
                live_hand_joint_map[finger_id] = finger_name
                self.hand_role_sources[side]["thumb"] = thumb_id
                self.hand_role_sources[side]["fingers"] = finger_id
                dashboard_tokens.append(f"{side[0].upper()}: T={thumb_id}, F={finger_id}")
                mode_tokens.append(f"{side}=thumb:{thumb_id}, fingers:{finger_id}")
            elif len(detected_ids) == 1:
                only_id = detected_ids[0]
                missing_id = candidate_ids[0] if only_id == candidate_ids[1] else candidate_ids[1]
                detected_role = "thumb" if only_id == candidate_ids[0] else "fingers"
                missing_role = "fingers" if detected_role == "thumb" else "thumb"
                live_hand_joint_map[only_id] = self.hand_role_names[side][detected_role]
                self.dummy_joints[missing_id] = self.hand_role_names[side][missing_role]
                self.dummy_joint_sources[missing_id] = only_id
                self.hand_role_sources[side][detected_role] = only_id
                self.hand_role_sources[side][missing_role] = only_id
                dashboard_tokens.append(f"{side[0].upper()}: T/F={only_id}")
                mode_tokens.append(f"{side}=only {only_id} detected -> thumb/fingers shared")
            else:
                self.dummy_joints[candidate_ids[0]] = thumb_name
                self.dummy_joints[candidate_ids[1]] = finger_name
                self.dummy_joint_sources[candidate_ids[0]] = None
                self.dummy_joint_sources[candidate_ids[1]] = None
                dashboard_tokens.append(f"{side[0].upper()}: T=dummy, F=dummy")
                mode_tokens.append(f"{side}=no hand motors")

        self.joint_map.update(live_hand_joint_map)
        self.ids.extend(live_hand_joint_map.keys())
        self.hand_status = " | ".join(dashboard_tokens)
        mode_label = "auto-detect" if self.hand_autodetect else "fixed IDs verified"
        self.hand_mode = f"ENABLED ({mode_label}: {'; '.join(mode_tokens)})"

    def _hand_value(self, side, role):
        source_id = self.hand_role_sources[side][role]
        if source_id is None:
            return 0.0
        return float(self.current_vals.get(source_id, 0.0))

    def _normalize_open1_close0(self, value, open_val, close_val):
        if open_val == close_val:
            return 1.0
        normalized = (float(value) - float(close_val)) / (float(open_val) - float(close_val))
        return max(0.0, min(1.0, normalized))

    def _normalize_hand_value(self, dxl_id, value):
        side = self.hand_joint_side.get(dxl_id)
        if side is None:
            return float(value)
        calibration = self.hand_calibration[side]
        return self._normalize_open1_close0(value, calibration["open"], calibration["close"])

    def _report_comm_status(self, is_error, err_code):
        if is_error:
            if self._comm_connected or self._last_comm_error_code != err_code:
                self.get_logger().warning(
                    f"Leader arm communication lost (code: {err_code}). Monitoring IDs: {self._monitored_ids}"
                )
            self._comm_connected = False
            self._last_comm_error_code = err_code
            return

        if not self._comm_connected:
            self.get_logger().info(f"Leader arm connected. Monitoring IDs: {self._monitored_ids}")
        self._comm_connected = True
        self._last_comm_error_code = None

    @staticmethod
    def _format_timing_metric(name, summary):
        if summary is None:
            return f"{name}_ms[n=0]"
        return (
            f"{name}_ms[n={summary['count']} "
            f"p50={summary['p50_ms']:.3f} "
            f"p95={summary['p95_ms']:.3f} "
            f"p99={summary['p99_ms']:.3f} "
            f"max={summary['max_ms']:.3f}]"
        )

    def _finish_timing(self, callback_started_ns, published):
        callback_finished_ns = time.perf_counter_ns()
        self._timing.observe_ns(
            "callback", callback_finished_ns - callback_started_ns
        )
        self._timing.record_attempt(published)

        report_interval_ns = int(self.timing_report_interval_s * 1.0e9)
        if report_interval_ns <= 0:
            return
        elapsed_ns = callback_finished_ns - self._last_timing_report_ns
        if elapsed_ns < report_interval_ns:
            return

        snapshot = self._timing.snapshot(reset=True)
        self._last_timing_report_ns = callback_finished_ns
        metric_text = " ".join(
            self._format_timing_metric(name, snapshot["metrics"][name])
            for name in ("loop", "txrx", "callback")
        )
        drop_text = ""
        if snapshot["dropped"] > 0 and self._last_drop_detail is not None:
            drop_text = f" last_drop={self._last_drop_detail!r}"
        self.get_logger().info(
            "leader_timing "
            f"window_s={elapsed_ns * 1.0e-9:.3f} "
            f"attempts={snapshot['attempts']} "
            f"published={snapshot['published']} "
            f"dropped={snapshot['dropped']} "
            f"{metric_text}"
            f"{drop_text}"
        )

    def _drop_sample(self, callback_started_ns, error_detail):
        """Report a failed acquisition without refreshing downstream data."""
        self._last_drop_detail = str(error_detail)
        self._report_comm_status(True, error_detail)
        if self.dashboard_enabled:
            v = self.current_vals
            rw_roll = (v[16] + v[17]) * 0.75
            rw_pitch = (v[16] - v[17]) * 0.75
            self.print_dashboard(rw_roll, rw_pitch, True, error_detail)
        self._finish_timing(callback_started_ns, published=False)

    def publish_data(self):
        callback_started_ns = time.perf_counter_ns()
        if self._last_callback_started_ns is not None:
            self._timing.observe_ns(
                "loop", callback_started_ns - self._last_callback_started_ns
            )
        self._last_callback_started_ns = callback_started_ns

        # 1. 통신 시도 (hand off 시에는 팔만 읽음)
        txrx_started_ns = time.perf_counter_ns()
        try:
            dxl_comm_result = self.group_sync_read.txRxPacket()
        except Exception as exc:
            txrx_finished_ns = time.perf_counter_ns()
            self._timing.observe_ns(
                "txrx", txrx_finished_ns - txrx_started_ns
            )
            self._drop_sample(
                callback_started_ns,
                f"txRxPacket exception: {type(exc).__name__}: {exc}",
            )
            return
        txrx_finished_ns = time.perf_counter_ns()
        self._timing.observe_ns("txrx", txrx_finished_ns - txrx_started_ns)

        if dxl_comm_result != COMM_SUCCESS:
            self._drop_sample(callback_started_ns, dxl_comm_result)
            return

        missing_ids = [
            dxl_id for dxl_id in self.ids
            if not self.group_sync_read.isAvailable(
                dxl_id,
                self.ADDR_PRESENT_POSITION,
                self.LEN_PRESENT_POSITION,
            )
        ]
        missing_arm_ids = [
            dxl_id for dxl_id in missing_ids if dxl_id in self.arm_joint_map
        ]
        if missing_arm_ids:
            missing_text = ",".join(str(dxl_id) for dxl_id in missing_arm_ids)
            self._drop_sample(
                callback_started_ns,
                f"sync-read arm data unavailable for IDs: {missing_text}",
            )
            return
        missing_hand_ids = [
            dxl_id for dxl_id in missing_ids if dxl_id in self.hand_joint_map
        ]
        missing_hand_tuple = tuple(missing_hand_ids)
        if missing_hand_tuple and missing_hand_tuple != self._last_missing_optional_hand_ids:
            missing_text = ",".join(str(dxl_id) for dxl_id in missing_hand_ids)
            self.get_logger().warning(
                "Optional leader hand sync-read data unavailable for IDs: "
                f"{missing_text}. Continuing to publish arm joints; missing hand "
                "joints keep their last/dummy values. Enable hand_autodetect or "
                "check hand wiring if hand teleop is required."
            )
        self._last_missing_optional_hand_ids = missing_hand_tuple or None

        next_values = {}
        try:
            for dxl_id in self.ids:
                if dxl_id in missing_ids:
                    continue
                raw_pos = self.group_sync_read.getData(
                    dxl_id,
                    self.ADDR_PRESENT_POSITION,
                    self.LEN_PRESENT_POSITION,
                )
                direction = self.directions.get(dxl_id, 1)
                next_values[dxl_id] = (
                    (2048 - raw_pos) * 0.001533981 * direction
                )
        except Exception as exc:
            self._drop_sample(
                callback_started_ns,
                f"sync-read decode exception: {type(exc).__name__}: {exc}",
            )
            return

        # Commit atomically only after every monitored motor supplied fresh data.
        self.current_vals.update(next_values)
        acquisition_stamp = self.get_clock().now().to_msg()
        
        # hand off 또는 미검출 손가락 관절은 초기값 0.0으로 유지됩니다.
        v = self.current_vals

        # 2. 손목 계산
        rw_roll  = (v[16] + v[17]) * 0.75
        rw_pitch = (v[16] - v[17]) * 0.75
        lw_roll  = (v[26] - v[27]) * 0.75
        lw_pitch = (v[26] + v[27]) * -0.75

        # 3. 메시지 생성
        joint_msg = JointState()
        joint_msg.header.stamp = acquisition_stamp

        # (A) Static Joints
        for s_name in self.static_joints:
            joint_msg.name.append(s_name)
            joint_msg.position.append(0.0)

        # (B) Real Joints
        for dxl_id in self.ids:
            name = self.joint_map[dxl_id]
            val = v[dxl_id]
            # Coupling Apply
            if dxl_id == 16: val = rw_roll
            elif dxl_id == 17: val = rw_pitch
            elif dxl_id == 26: val = lw_roll
            elif dxl_id == 27: val = lw_pitch
            
            joint_msg.name.append(name)
            joint_msg.position.append(float(val))
            
        # (C) Dummy Finger Joints (hand off 시 값은 0.0으로 전송)
        for dxl_id, name in self.dummy_joints.items():
            source_id = self.dummy_joint_sources.get(dxl_id)
            if source_id is None:
                val = 0.0
            else:
                val = v.get(source_id, 0.0)
            joint_msg.name.append(name)
            joint_msg.position.append(float(val))

        self.joint_publisher_.publish(joint_msg)
        self._report_comm_status(False, dxl_comm_result)
        if self.dashboard_enabled:
            self.print_dashboard(rw_roll, rw_pitch, False, dxl_comm_result)
        self._finish_timing(callback_started_ns, published=True)

    def print_dashboard(self, rw_roll, rw_pitch, is_error, err_code):
        print("\033[H", end="")
        v = self.current_vals
        status_msg = "Running [OK]" if not is_error else f"COMM ERROR (Code: {err_code})"
        right_thumb = self._hand_value("right", "thumb")
        right_fingers = self._hand_value("right", "fingers")
        left_thumb = self._hand_value("left", "thumb")
        left_fingers = self._hand_value("left", "fingers")
        
        dashboard = f"""
================================================================================
     IGRIS Leader Arm - {status_msg}
================================================================================
 [RIGHT ARM] (Wrist Coupled)        |  [LEFT ARM] (Wrist Coupled)
 Shoulder P : {v[11]:5.2f}              |  Shoulder P : {v[21]:5.2f}
 Shoulder R : {v[12]:5.2f}              |  Shoulder R : {v[22]:5.2f}
 Shoulder Y : {v[13]:5.2f}              |  Shoulder Y : {v[23]:5.2f}
 Elbow    P : {v[14]:5.2f}              |  Elbow    P : {v[24]:5.2f}
 Wrist    Y : {v[15]:5.2f}              |  Wrist    Y : {v[25]:5.2f}
 Wrist    R : {rw_roll:5.2f} (Calc)      |  Wrist    R : {v[26]:5.2f} (Calc)
 Wrist    P : {rw_pitch:5.2f} (Calc)      |  Wrist    P : {v[27]:5.2f} (Calc)
 -----------------------------------+---------------------------------------
 [HANDS] ({self.hand_status})
 Finger_T_R : {right_thumb:5.2f}              |  Finger_T_L : {left_thumb:5.2f}
 Finger_F_R : {right_fingers:5.2f}              |  Finger_F_L : {left_fingers:5.2f}
================================================================================
 * hand_enabled = {self.hand_enabled}
 * Checking IDs: {self._monitored_ids}
"""
        print(dashboard, end="", flush=True)

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IgrisLeaderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.port_handler.closePort()
            node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
