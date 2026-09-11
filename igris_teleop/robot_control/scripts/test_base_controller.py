"""
실제 구동용 단독 스크립트:
1) BaseController 초기화
2) zero_torque_state → damping_state → default_pos 이동
3) while 루프에서 neck pitch만 sin 파로 왕복
"""
import math
import signal
import sys
import time
import threading
import pathlib
import argparse
import numpy as np

# Ensure repository root is on sys.path so `igris_teleop.*` imports work in script mode.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import igris_c_sdk as igc_sdk

from igris_teleop.core.project_paths import DATASETS_ROOT
from igris_teleop.robot_control.controller.igris_controller import BaseController
from igris_teleop.robot_control.kinematics.joints import JointIndex, ARM_INDICES
from collections import deque
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id")
    parser.add_argument("--hz", type=float, default=300.0, help="control loop frequency")
    parser.add_argument("--state-timeout", type=float, default=5.0, help="wait for first LowState (s)")
    parser.add_argument("--svc-timeout", type=int, default=30000, help="service timeout (ms)")
    parser.add_argument("--amp", type=float, default=0.6, help="neck pitch sine amplitude (rad)")
    parser.add_argument("--freq", type=float, default=0.5, help="neck pitch sine frequency (Hz)")
    parser.add_argument("--damping-kd", type=float, default=3.0, help="kd for damping_state")
    parser.add_argument("--move-duration", type=float, default=5.0, help="seconds to move to default pose")
    parser.add_argument("--mode", choices=["pjs", "ms"], default="pjs", help="kinematic mode for LowCmd")
    args = parser.parse_args()

    domain_id = args.domain
    control_hz = args.hz
    state_timeout = args.state_timeout
    damping_kd = args.damping_kd
    move_duration = args.move_duration
    amp = args.amp
    freq = args.freq
    svc_timeout_ms = args.svc_timeout
    kinematic_mode = igc_sdk.KinematicMode.PJS if args.mode == "pjs" else igc_sdk.KinematicMode.MS
    use_motor_state = args.mode == "ms"

    stop_requested = False
    reverse_requested = False
    
    episode_idx = 0  # 재생할 episode
    root = str((DATASETS_ROOT / "IGRIS_C_20251215_231430").resolve())
    dataset = LeRobotDataset("IGRIS_C", root=root, episodes=[episode_idx])

    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
    actions_only = episode_frames.select_columns(ACTION)   # action 컬럼만 남김
        
        
        
    def _keyboard_listener():
        nonlocal stop_requested, reverse_requested
        while not stop_requested and not reverse_requested:
            ch = sys.stdin.read(1)
            if not ch:
                continue
            if ch.lower() == "q":
                print("[input] 'q' pressed → reverse sequence")
                reverse_requested = True
                stop_requested = True
                break

    # Start non-blocking keyboard listener for 'q'
    kb_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    kb_thread.start()

    def _handle_signal(signum, _frame):
        nonlocal stop_requested
        print(f"[signal] {signum} received, stopping...")
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass


    ctrl = BaseController(
        domain_id=domain_id,
        control_hz=control_hz,
        kinematic_mode=kinematic_mode,
        use_motor_state=use_motor_state,
        auto_service_init=True,
    )
    try:
        if not ctrl.wait_for_state(timeout=state_timeout):
            print(f"[error] No LowState within {state_timeout}s")
            return 1


        print(f"[step] move_to__pose ({move_duration}s) + default_pos_state")
        ctrl.move_to_pose("default_pos", duration=move_duration)
        ctrl.default_pos_state()
        time.sleep(0.5)
        if stop_requested:
            return 0

        q_now = ctrl.get_joint_q()
        if q_now is None:
            print("[error] Missing joint state after init")
            return 1
        
        # yaw_base = float(q_now[JointIndex.NECK_YAW])
        # pitch_base = float(q_now[JointIndex.NECK_PITCH])

        arm_indices = list(ARM_INDICES)  # len=14 (L arm 7 + R arm 7)
        # ctrl_arm에 넣을 기본(hold) 벡터: 현재 팔 관절값을 그대로 사용
        arm_base = [float(q_now[i]) for i in arm_indices]

        # 왼쪽 손목 롤의 arm 벡터 내 위치
        pos_l_wrist_roll  = arm_indices.index(JointIndex.R_WRIST_ROLL)
        pos_l_wrist_pitch = arm_indices.index(JointIndex.R_WRIST_PITCH)

        roll_base  = arm_base[pos_l_wrist_roll]
        pitch_base = arm_base[pos_l_wrist_pitch]


        # ---- 실시간 플롯 버퍼/초기화 ----
        win_sec = 10.0
        maxlen = int(win_sec / 0.01) + 200

        t_buf = deque(maxlen=maxlen)

        roll_t_buf  = deque(maxlen=maxlen)
        roll_m_buf  = deque(maxlen=maxlen)
        pitch_t_buf = deque(maxlen=maxlen)
        pitch_m_buf = deque(maxlen=maxlen)

        roll_mtr_buf  = deque(maxlen=maxlen)
        pitch_mtr_buf = deque(maxlen=maxlen)
        roll_tau_buf  = deque(maxlen=maxlen)
        pitch_tau_buf = deque(maxlen=maxlen)

        plt.ion()
        fig, (ax_j, ax_m) = plt.subplots(2, 1, sharex=True)

        # (추가) 아래쪽 텍스트 영역 확보 (하단 여백 늘리기)
        fig.subplots_adjust(bottom=0.22)

        # (추가) figure 하단에 표시할 텍스트 아티스트
        info_text = fig.text(
            0.01, 0.02, "",  # (x,y) figure 좌표(0~1)
            ha="left", va="bottom",
            fontsize=10,
            family="monospace"
        )

        # --- Joint subplot (target vs joint current) ---
        (line_r_t_j,) = ax_j.plot([], [], label="roll target(cmd_q)")
        (line_r_j,)   = ax_j.plot([], [], label="roll current(joint)")
        (line_p_t_j,) = ax_j.plot([], [], label="pitch target(cmd_q)")
        (line_p_j,)   = ax_j.plot([], [], label="pitch current(joint)")

        ax_j.set_title("RIGHT WRIST: target vs JOINT current")
        ax_j.set_ylabel("q [rad]")
        ax_j.grid(True)
        ax_j.legend(loc="upper right")

        # --- Motor subplot (target vs motor current) ---
        (line_r_t_m,) = ax_m.plot([], [], label="roll tau_est(cmd_q)")
        (line_r_m,)   = ax_m.plot([], [], label="Wrist_Front_R(motor)")
        (line_p_t_m,) = ax_m.plot([], [], label="pitch tau_est(cmd_q)")
        (line_p_m,)   = ax_m.plot([], [], label="Wrist_Back_R(motor)")

        ax_m.set_title("RIGHT WRIST: MOTOR current")
        ax_m.set_xlabel("t [s]")
        ax_m.set_ylabel("q [rad]")
        ax_m.grid(True)
        ax_m.legend(loc="upper right")

        last_plot_t = 0.0
        plot_dt = 0.05

        print("[step] neck pitch sin-wave start (Ctrl+C to stop)")
        start = time.perf_counter()
        
        IDX_ROLL  = int(JointIndex.R_WRIST_ROLL)
        IDX_PITCH = int(JointIndex.R_WRIST_PITCH)

        L_IDX_ROLL = int(JointIndex.L_WRIST_ROLL)
        L_IDX_PITCH = int(JointIndex.L_WRIST_PITCH)

        def map_left_to_right(l):
            Lmin, Lmax = -1.22, 0.87
            Rmin, Rmax = -0.87, 1.22
            
            # 범위 밖 값 방지(선택)
            l = max(Lmin, min(Lmax, l))
            
            # 반전 선형 매핑
            r = Rmin + (Lmax - l) * (Rmax - Rmin) / (Lmax - Lmin)
            return r

        step = 0  # dataset frame index
        
        while not stop_requested:
            t = time.perf_counter() - start

            # # 같은 주파수로 동시에 구동 (원하면 위상차도 줄 수 있음)
            # roll  = 0.0 #roll_base  + amp * math.sin(2.0 * math.pi * freq * t)
            # # pitch = pitch_base + amp * math.sin(2.0 * math.pi * freq * t)
            # pitch = 0.0 #pitch_base + amp * math.sin(2.0 * math.pi * freq * t + math.pi/2)
            
            action_array = actions_only[step][ACTION]  # list/np/torch 형태일 수 있음

            roll = float(action_array[IDX_ROLL])
            pitch = float(action_array[IDX_PITCH])

            roll = np.clip(roll, -0.60, 1.10)
            pitch = np.clip(pitch, -0.60, 0.60)

            # roll  = map_left_to_right(float(action_array[L_IDX_ROLL]))
            # pitch = float(action_array[L_IDX_PITCH])

            arm_cmd = list(arm_base)
            arm_cmd[pos_l_wrist_roll]  = float(roll)
            arm_cmd[pos_l_wrist_pitch] = float(pitch)

            ctrl.ctrl_arm(arm_cmd, apply_clip=False)

            # ---- 현재값/타겟값 읽기 ----
            q_target = ctrl._target_q.copy()
            q_joint = ctrl.get_joint_q()
            q_motor = ctrl.get_motor_q()
            q_tau = ctrl.get_joint_tau()

            if (q_joint is not None) and (q_motor is not None):
                cmd_roll   = float(q_target[JointIndex.R_WRIST_ROLL])
                cmd_pitch  = float(q_target[JointIndex.R_WRIST_PITCH])

                current_roll_j  = float(q_joint[JointIndex.R_WRIST_ROLL])
                current_pitch_j = float(q_joint[JointIndex.R_WRIST_PITCH])

                current_roll_m  = float(q_motor[JointIndex.R_WRIST_ROLL])
                current_pitch_m = float(q_motor[JointIndex.R_WRIST_PITCH])
                
                current_rool_tau = float(q_tau[JointIndex.R_WRIST_ROLL])
                current_pitch_tau = float(q_tau[JointIndex.R_WRIST_PITCH])

                print(f"[{t:.2f}s] cmd_roll: {cmd_roll:.3f}, current_roll_j: {current_roll_j:.3f}, cmd_pitch: {cmd_pitch:.3f} current_pitch_j: {current_pitch_j:.3f} | ")

                t_buf.append(t)
                roll_t_buf.append(cmd_roll)
                pitch_t_buf.append(cmd_pitch)

                roll_m_buf.append(current_roll_j)
                pitch_m_buf.append(current_pitch_j)

                roll_mtr_buf.append(current_roll_m)
                pitch_mtr_buf.append(current_pitch_m)
                
                roll_tau_buf.append(current_rool_tau)
                pitch_tau_buf.append(current_pitch_tau)
                
                info = (
                    f"t={t:6.2f}s\n"
                    f"ROLL : cmd={cmd_roll:+.3f} | joint={current_roll_j:+.3f} | motor={current_roll_m:+.3f}\n"
                    f"PITCH: cmd={cmd_pitch:+.3f} | joint={current_pitch_j:+.3f} | motor={current_pitch_m:+.3f}"
                )
                # 너무 자주 갱신하면 깜빡일 수 있어서 plot_dt 주기에 맞춰 갱신 추천
                info_text.set_text(info)

                
            # ---- plot 업데이트(저주기) ----
            now = time.perf_counter()
            if now - last_plot_t >= plot_dt and len(t_buf) > 2:
                last_plot_t = now

                tt = list(t_buf)
                yr_t = list(roll_t_buf)
                yp_t = list(pitch_t_buf)

                yr_j = list(roll_m_buf)
                yp_j = list(pitch_m_buf)

                yr_m = list(roll_mtr_buf)
                yp_m = list(pitch_mtr_buf)
                
                yr_tau = list(roll_tau_buf)
                yp_tau = list(pitch_tau_buf)

                # Joint subplot
                line_r_t_j.set_data(tt, yr_t)
                line_p_t_j.set_data(tt, yp_t)
                line_r_j.set_data(tt, yr_j)
                line_p_j.set_data(tt, yp_j)

                # Motor subplot
                line_r_t_m.set_data(tt, yr_tau)
                line_p_t_m.set_data(tt, yp_tau)
                line_r_m.set_data(tt, yr_m)
                line_p_m.set_data(tt, yp_m)

                t_max = tt[-1]
                ax_m.set_xlim(max(0.0, t_max - win_sec), t_max)





                # y-limit은 축별로 따로 잡는 편이 보기 좋습니다.
                def _set_ylim(ax, ys):
                    y_min, y_max = min(ys), max(ys)
                    margin = max(0.02, 0.1 * (y_max - y_min))
                    ax.set_ylim(y_min - margin, y_max + margin)

                _set_ylim(ax_j, yr_t + yp_t + yr_j + yp_j)
                # _set_ylim(ax_m, yr_t + yp_t + yr_m + yp_m)
                _set_ylim(ax_m, yr_tau + yp_tau + yr_m + yp_m)

                info_text.set_text(info)   # (여기에)

                fig.canvas.draw_idle()
                plt.pause(0.001)

                if not plt.fignum_exists(fig.number):
                    stop_requested = True

            step += 1

            time.sleep(0.033)



        return 0
    finally:
        print("[demo] Stopping controller thread")
        ctrl.stop()


if __name__ == "__main__":
    sys.exit(main())
