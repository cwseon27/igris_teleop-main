#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse

def main():
    # -------------------------------------------------------------
    # 설정 (필요시 수정)
    # -------------------------------------------------------------
    # 튜닝된 맵 파일 경로 (사용자 경로에 맞춤)
    npz_path = os.path.expanduser("~/stereo_cam_ws/src/stereo_sbs_cam_pub/config/stereo_rectify_maps_tuned.npz")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_index", type=int, default=2, help="Camera Device Index")
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    # -------------------------------------------------------------
    # 1. 데이터 로드
    # -------------------------------------------------------------
    if not os.path.exists(npz_path):
        # 현재 폴더에서라도 찾아봄
        if os.path.exists("stereo_rectify_maps_tuned.npz"):
            npz_path = "stereo_rectify_maps_tuned.npz"
        else:
            print(f"[ERR] NPZ file not found: {npz_path}")
            return

    print(f"[INFO] Loading maps from: {npz_path}")
    data = np.load(npz_path)
    
    # (3) Stereo Rectification Maps (Stereo 단계용)
    # 키 이름 호환성 체크
    if 'mapL1' in data:
        mapL1, mapL2 = data['mapL1'], data['mapL2']
        mapR1, mapR2 = data['mapR1'], data['mapR2']
    else:
        mapL1, mapL2 = data['map1x'], data['map1y']
        mapR1, mapR2 = data['map2x'], data['map2y']

    # (2) Monocular Intrinsics (Calibration 단계용 - 단순 렌즈 왜곡 제거)
    K1, D1 = data['K1'], data['D1']
    K2, D2 = data['K2'], data['D2']

    # -------------------------------------------------------------
    # 2. 카메라 연결
    # -------------------------------------------------------------
    cap = cv2.VideoCapture(args.device_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERR] Cannot open camera {args.device_index}")
        return

    print("========================================================")
    print(" [Visualizer Started] Mode 4 Fixed (Flip + Swap)")
    print(" ROW 1: Raw Input (After Flip/Swap)")
    print(" ROW 2: Monocular Undistort (Lens Only)")
    print(" ROW 3: Stereo Rectify (Lens + Alignment + Zoom)")
    print(" Press 'q' to exit")
    print("========================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No Frame")
            break

        # =========================================================
        # 1. 전처리 (Mode 4 고정: Flip 180 -> Split -> Swap)
        # =========================================================
        # A. Flip 180
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # B. Split
        h, w = frame.shape[:2]
        w_half = w // 2
        img_l_raw = frame[:, :w_half]
        img_r_raw = frame[:, w_half:]

        # C. Swap (물리적 좌우 반전 대응)
        img_l_raw, img_r_raw = img_r_raw, img_l_raw

        # --> 여기까지가 "Raw Input" (물리적으로 올바른 좌/우)

        # =========================================================
        # 2. 처리 단계별 생성
        # =========================================================
        
        # [ROW 1] Raw Image
        # 시각화를 위해 복사
        vis_raw_l = img_l_raw.copy()
        vis_raw_r = img_r_raw.copy()
        cv2.putText(vis_raw_l, "Raw Left", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(vis_raw_r, "Raw Right", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # [ROW 2] Monocular Undistort (일반 캘리브레이션)
        # K, D 만을 이용해서 렌즈 왜곡만 폄 (정렬/줌 없음)
        vis_mono_l = cv2.undistort(img_l_raw, K1, D1)
        vis_mono_r = cv2.undistort(img_r_raw, K2, D2)
        cv2.putText(vis_mono_l, "Mono Undistort Left", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(vis_mono_r, "Mono Undistort Right", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # [ROW 3] Stereo Rectify (최종 튜닝 결과)
        # Map을 이용해서 렌즈왜곡 + 평행정렬 + 줌/알파 적용
        vis_rect_l = cv2.remap(img_l_raw, mapL1, mapL2, cv2.INTER_LINEAR)
        vis_rect_r = cv2.remap(img_r_raw, mapR1, mapR2, cv2.INTER_LINEAR)
        
        # 가이드라인 그리기 (정렬 확인용)
        for y in range(0, h, 40):
            cv2.line(vis_rect_l, (0, y), (w_half, y), (0, 255, 0), 1)
            cv2.line(vis_rect_r, (0, y), (w_half, y), (0, 255, 0), 1)
            
        cv2.putText(vis_rect_l, "Stereo Rectified Left", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(vis_rect_r, "Stereo Rectified Right", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # =========================================================
        # 3. 통합 출력 (2x3 Grid)
        # =========================================================
        row1 = np.hstack([vis_raw_l, vis_raw_r])
        row2 = np.hstack([vis_mono_l, vis_mono_r])
        row3 = np.hstack([vis_rect_l, vis_rect_r])

        full_grid = np.vstack([row1, row2, row3])

        # 화면이 너무 크니 절반으로 줄여서 출력
        display_img = cv2.resize(full_grid, None, fx=0.5, fy=0.5)

        cv2.imshow("Calibration Stages Comparison", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()