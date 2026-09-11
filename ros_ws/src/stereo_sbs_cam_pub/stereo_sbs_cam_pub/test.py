#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge

class SBSCamRectifiedPub(Node):
    def __init__(self):
        super().__init__('sbs_cam_split_undistort_pub')

        # --- 파라미터 ---
        self.declare_parameter('npz_filename', 'stereo_rectify_maps_tuned.npz')
        self.declare_parameter('device_index', 0)
        self.declare_parameter('capture_width', 2560)
        self.declare_parameter('capture_height', 720)
        self.declare_parameter('capture_fps', 30.0)
        self.declare_parameter('publish_fps', 30.0)
        self.declare_parameter('resize_scale', 1.0)
        self.declare_parameter('jpeg_quality', 80)
        
        # --- 변수 초기화 ---
        self.npz_filename = self.get_parameter('npz_filename').value
        self.dev_idx = self.get_parameter('device_index').value
        self.cap_w = self.get_parameter('capture_width').value
        self.cap_h = self.get_parameter('capture_height').value
        self.resize_scale = self.get_parameter('resize_scale').value
        self.jpeg_qual = self.get_parameter('jpeg_quality').value
        
        # --- 디버그 모드 변수 ---
        self.swap_lr = False # 좌우 반전 여부
        self.flip_180 = False # 180도 회전 여부 (상하좌우 반전)

        # --- 맵 로드 ---
        # (경로 하드코딩: config 폴더에 파일이 있다고 가정)
        # 개발 환경 경로 or 현재 경로
        path1 = os.path.expanduser(f"~/stereo_cam_ws/src/stereo_sbs_cam_pub/config/{self.npz_filename}")
        path2 = self.npz_filename # 실행 위치 기준
        
        if os.path.exists(path1): self.npz_path = path1
        else: self.npz_path = path2
            
        self.load_maps()

        # --- Publisher ---
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_l_comp = self.create_publisher(CompressedImage, '/stereo_left/image_rect/compressed', qos)
        self.pub_r_comp = self.create_publisher(CompressedImage, '/stereo_right/image_rect/compressed', qos)

        self.bridge = CvBridge()

        # --- Camera ---
        print(f"[INFO] Opening Camera Index: {self.dev_idx}")
        self.cap = cv2.VideoCapture(self.dev_idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera Open Failed: Index {self.dev_idx}")

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 스레드 시작
        threading.Thread(target=self._capture_loop, daemon=True).start()
        self.create_timer(1.0/30.0, self._publish_callback)
        
        print("\n" + "="*50)
        print(" [DIAGNOSTIC MODE STARTED] ")
        print(" Press keys in the OpenCV window to toggle modes:")
        print(" '1': Normal (No Swap, No Flip)")
        print(" '2': Swap L/R Only")
        print(" '3': Rotate 180 (Flip Both)")
        print(" '4': Rotate 180 + Swap L/R")
        print(" 'q': Exit")
        print("="*50 + "\n")

    def load_maps(self):
        if not os.path.exists(self.npz_path):
            self.get_logger().error(f"NPZ not found: {self.npz_path}")
            return
        data = np.load(self.npz_path)
        # 키 호환성 체크
        if 'mapL1' in data:
            self.map1x, self.map1y = data['mapL1'], data['mapL2']
            self.map2x, self.map2y = data['mapR1'], data['mapR2']
        else:
            self.map1x, self.map1y = data['map1x'], data['map1y']
            self.map2x, self.map2y = data['map2x'], data['map2y']
        
        # **검증 코드**: P1[0,0] (Zoom Factor) 출력
        P1 = data['P1'] if 'P1' in data else np.eye(3,4)
        print(f"[VERIFY] Loaded Map Zoom Factor (fx): {P1[0,0]:.2f} (If > 500, Zoom applied!)")
        self.maps_loaded = True

    def _capture_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.1)
            time.sleep(0.005)

    def _publish_callback(self):
        frame = None
        with self.frame_lock:
            if self.latest_frame is not None: frame = self.latest_frame.copy()
        if frame is None: return

        # --- [1] 전처리: 전체 이미지 회전 (물리적 설치 방향 대응) ---
        if self.flip_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # --- [2] 분할 ---
        h, w, _ = frame.shape
        w_half = w // 2
        img_l = frame[:, :w_half]
        img_r = frame[:, w_half:]

        # --- [3] 스왑 (전송 순서 대응) ---
        if self.swap_lr:
            img_l, img_r = img_r, img_l

        # --- [4] 리맵 (캘리브레이션) ---
        if self.maps_loaded:
            rect_l = cv2.remap(img_l, self.map1x, self.map1y, cv2.INTER_LINEAR)
            rect_r = cv2.remap(img_r, self.map2x, self.map2y, cv2.INTER_LINEAR)
        else:
            rect_l, rect_r = img_l, img_r

        # --- [5] 화면 출력 (진단용) ---
        vis = np.hstack([rect_l, rect_r])
        # 가로선 그리기 (정렬 확인)
        for y in range(0, h, 50):
            cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
        
        # 상태 텍스트
        status = f"Mode: {'Normal' if not self.swap_lr and not self.flip_180 else ''}"
        status += f"{'SwapLR ' if self.swap_lr else ''}"
        status += f"{'Flip180 ' if self.flip_180 else ''}"
        
        cv2.putText(vis, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.imshow("Diagnostic (Press 1/2/3/4)", cv2.resize(vis, (1280, 360)))
        key = cv2.waitKey(1) & 0xFF
        
        # 키 입력 처리
        if key == ord('1'):
            self.swap_lr = False; self.flip_180 = False
        elif key == ord('2'):
            self.swap_lr = True; self.flip_180 = False
        elif key == ord('3'):
            self.swap_lr = False; self.flip_180 = True
        elif key == ord('4'):
            self.swap_lr = True; self.flip_180 = True
        elif key == ord('q'):
            rclpy.shutdown()

        # --- [6] ROS Publish ---
        stamp = self.get_clock().now().to_msg()
        self.pub_l_comp.publish(self.create_compressed_msg(rect_l, stamp))
        self.pub_r_comp.publish(self.create_compressed_msg(rect_r, stamp))

    def create_compressed_msg(self, cv_img, stamp):
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.format = "jpeg"
        _, encoded = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_qual])
        msg.data = encoded.tobytes()
        return msg

# --- 메인 함수 수정 (버그 해결됨) ---
def main(args=None):
    rclpy.init(args=args)
    node = SBSCamRectifiedPub() # 한 번만 생성!
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
