#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import sys

import cv2
import numpy as np

def open_stereo_capture(device: str | None, index: int, width: int, height: int, fps: int, mjpg: bool):
    src = device if device else index
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)

    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    if not cap.isOpened():
        raise RuntimeError(f"[Stereo] VideoCapture open failed: {src}")
    return cap

def open_realsense_pipeline(serial: str | None, width: int, height: int, fps: int):
    try:
        import pyrealsense2 as rs
    except Exception as e:
        raise RuntimeError(
            "pyrealsense2 import 실패. RealSense SDK/파이썬 바인딩 설치가 필요합니다.\n"
            f"원인: {e}"
        )

    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("[RealSense] No device found.")

    if serial is None:
        serial = devs[0].get_info(rs.camera_info.serial_number)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)

    # color: RGB8로 받고 OpenCV용 BGR로 변환
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    try:
        pipeline.start(cfg)
    except Exception as e:
        raise RuntimeError(
            "[RealSense] pipeline.start() 실패. (USB 대역폭/전원/허브 문제일 가능성 큼)\n"
            f"원인: {e}"
        )

    return rs, pipeline, serial

def split_sbs(frame_bgr: np.ndarray, rotate180: bool, swap_lr: bool):
    if rotate180:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)

    h, w = frame_bgr.shape[:2]
    w_half = w // 2
    left  = frame_bgr[:, :w_half]
    right = frame_bgr[:, w_half: w_half*2]

    if swap_lr:
        left, right = right, left

    return left, right

def pad_or_resize_to_height(img: np.ndarray, target_h: int):
    if img is None:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    if h == target_h:
        return img

    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

def main():
    ap = argparse.ArgumentParser()
    # Stereo(SBS)
    ap.add_argument("--stereo_device", default=None, help="예: /dev/v4l/by-path/...  (없으면 index 사용)")
    ap.add_argument("--stereo_index", type=int, default=0, help="stereo_device 미지정 시 사용")
    ap.add_argument("--st_w", type=int, default=2560)
    ap.add_argument("--st_h", type=int, default=720)
    ap.add_argument("--st_fps", type=int, default=30)
    ap.add_argument("--mjpg", action="store_true", help="MJPG 강제 (권장)")
    ap.add_argument("--rotate180", action="store_true", help="SBS 프레임 180도 회전")
    ap.add_argument("--swap_lr", action="store_true", help="좌/우 스왑")

    # RealSense
    ap.add_argument("--rs_serial", default=None, help="특정 RealSense serial 선택")
    ap.add_argument("--rs_w", type=int, default=640)
    ap.add_argument("--rs_h", type=int, default=480)
    ap.add_argument("--rs_fps", type=int, default=30)

    # Display
    ap.add_argument("--display_scale", type=float, default=0.7, help="전체 표시 스케일")
    ap.add_argument("--separate", action="store_true", help="창 3개로 분리")
    ap.add_argument("--rs_timeout_ms", type=int, default=0,
                    help="0이면 poll(논블로킹). >0이면 wait_for_frames timeout(ms) 사용")

    args = ap.parse_args()

    cap = None
    pipeline = None

    last_left = None
    last_right = None
    last_rs_bgr = None

    try:
        cap = open_stereo_capture(
            args.stereo_device, args.stereo_index,
            args.st_w, args.st_h, args.st_fps,
            mjpg=args.mjpg
        )
        rs, pipeline, serial = open_realsense_pipeline(args.rs_serial, args.rs_w, args.rs_h, args.rs_fps)
        print(f"[OK] Stereo opened. source={args.stereo_device or args.stereo_index}")
        print(f"[OK] RealSense opened. serial={serial}")

        while True:
            # ---- Stereo ----
            ok, frame = cap.read()
            if ok and frame is not None:
                left, right = split_sbs(frame, rotate180=args.rotate180, swap_lr=args.swap_lr)
                last_left, last_right = left, right

            # ---- RealSense ----
            frames = None
            if args.rs_timeout_ms and args.rs_timeout_ms > 0:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=args.rs_timeout_ms)
                except Exception:
                    frames = None
            else:
                frames = pipeline.poll_for_frames()

            if frames is not None:
                color = frames.get_color_frame()
                if color:
                    rgb = np.asanyarray(color.get_data())
                    last_rs_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ---- Display ----
            if last_left is None or last_right is None:
                # 아직 스테레오 프레임이 없으면 대기
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            target_h = last_left.shape[0]
            rs_disp = pad_or_resize_to_height(last_rs_bgr, target_h)

            # if args.separate:
            #     cv2.imshow("Stereo Left",  cv2.resize(last_left,  None, fx=args.display_scale, fy=args.display_scale))
            #     cv2.imshow("Stereo Right", cv2.resize(last_right, None, fx=args.display_scale, fy=args.display_scale))
            #     cv2.imshow("RealSense Color", cv2.resize(rs_disp, None, fx=args.display_scale, fy=args.display_scale))
            # else:
            #     grid = np.hstack([last_left, last_right, rs_disp])
            #     grid = cv2.resize(grid, None, fx=args.display_scale, fy=args.display_scale)
            #     cv2.imshow("Stereo(L/R) + RealSense(Color)", grid)

            cv2.imwrite("/tmp/stereo_left.jpg", left)
            cv2.imwrite("/tmp/stereo_right.jpg", right)
            cv2.imwrite("/tmp/realsense.jpg", rs_disp)

            # key = cv2.waitKey(1) & 0xFF
            # if key == ord("q"):
            #     break

            # 너무 CPU 먹으면 약간 쉬기
            # time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e, file=sys.stderr)
    finally:
        if cap is not None:
            cap.release()
        if pipeline is not None:
            try:
                pipeline.stop()
                
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        
if __name__ == "__main__":
    main()
