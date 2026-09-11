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
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge

class SBSCamRectifiedPub(Node):
    def __init__(self):
        super().__init__('sbs_cam_split_undistort_pub')

        # ---------------------------------------------------------------------
        # 1. 파라미터 설정
        # ---------------------------------------------------------------------
        self.declare_parameter('device_index', 0)
        self.declare_parameter('capture_width', 2560)
        self.declare_parameter('capture_height', 720)
        self.declare_parameter('capture_fps', 30.0)

        self.declare_parameter('publish_fps', 30.0)
        self.declare_parameter('camera_info_hz', 0.5)

        self.declare_parameter('use_v4l2', True)
        self.declare_parameter('force_mjpg', True)

        # 이 값이 True면 'Mode 4' (뒤집기 + 좌우바꾸기)로 동작합니다.
        self.declare_parameter('rotate_180', True) 
        
        self.declare_parameter('resize_scale', 1.0)
        self.declare_parameter('jpeg_quality', 85)
        
        self.declare_parameter('use_stereo_rectify', True)
        self.declare_parameter('stereo_npz_path', 'stereo_rectify_maps_tuned.npz')

        self.declare_parameter('left_image_topic', '/stereo_left/image_rect')
        self.declare_parameter('right_image_topic', '/stereo_right/image_rect')
        self.declare_parameter('left_image_topic_compressed', '/stereo_left/image_rect/compressed')
        self.declare_parameter('right_image_topic_compressed', '/stereo_right/image_rect/compressed')
        self.declare_parameter('left_info_topic', '/stereo_left/camera_info')
        self.declare_parameter('right_info_topic', '/stereo_right/camera_info')

        self.declare_parameter('left_frame_id', 'stereo_left_camera')
        self.declare_parameter('right_frame_id', 'stereo_right_camera')

        # ---------------------------------------------------------------------
        # 2. 파라미터 읽기
        # ---------------------------------------------------------------------
        self.dev_idx = self.get_parameter('device_index').value
        self.cap_w = self.get_parameter('capture_width').value
        self.cap_h = self.get_parameter('capture_height').value
        self.cap_fps = self.get_parameter('capture_fps').value
        
        self.pub_fps = self.get_parameter('publish_fps').value
        self.use_v4l2 = self.get_parameter('use_v4l2').value
        self.force_mjpg = self.get_parameter('force_mjpg').value
        
        # rotate_180=True means each split eye is rotated 180 degrees before rectification.
        self.mode4_active = self.get_parameter('rotate_180').value
        
        self.resize_scale = self.get_parameter('resize_scale').value
        self.jpeg_qual = self.get_parameter('jpeg_quality').value
        
        self.use_rectify = self.get_parameter('use_stereo_rectify').value
        self.npz_path_param = self.get_parameter('stereo_npz_path').value

        self.topic_l_raw = self.get_parameter('left_image_topic').value
        self.topic_r_raw = self.get_parameter('right_image_topic').value
        self.topic_l_comp = self.get_parameter('left_image_topic_compressed').value
        self.topic_r_comp = self.get_parameter('right_image_topic_compressed').value
        self.topic_l_info = self.get_parameter('left_info_topic').value
        self.topic_r_info = self.get_parameter('right_info_topic').value
        
        self.fid_l = self.get_parameter('left_frame_id').value
        self.fid_r = self.get_parameter('right_frame_id').value

        # ---------------------------------------------------------------------
        # 3. 맵 로드
        # ---------------------------------------------------------------------
        self.maps_loaded = False
        if self.use_rectify:
            self.load_rectification_maps(self.npz_path_param)

        # ---------------------------------------------------------------------
        # 4. Publisher 설정
        # ---------------------------------------------------------------------
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.pub_l_comp = self.create_publisher(CompressedImage, self.topic_l_comp, qos)
        self.pub_r_comp = self.create_publisher(CompressedImage, self.topic_r_comp, qos)
        self.pub_l_raw = self.create_publisher(Image, self.topic_l_raw, qos)
        self.pub_r_raw = self.create_publisher(Image, self.topic_r_raw, qos)
        self.pub_l_info = self.create_publisher(CameraInfo, self.topic_l_info, qos)
        self.pub_r_info = self.create_publisher(CameraInfo, self.topic_r_info, qos)

        self.bridge = CvBridge()

        # ---------------------------------------------------------------------
        # 5. 카메라 시작 (V4L2)
        # ---------------------------------------------------------------------
        backend = cv2.CAP_V4L2 if self.use_v4l2 else cv2.CAP_ANY
        # self.cap = cv2.VideoCapture(self.dev_idx, backend)
        self.declare_parameter('device', '/dev/v4l/by-path/pci-0000:71:00.4-usb-0:2.1.2:1.0-video-index0')
        dev_path = self.get_parameter('device').value
        self.cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        
        if self.force_mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_h)
        self.cap.set(cv2.CAP_PROP_FPS, self.cap_fps)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera index {self.dev_idx}")
            raise RuntimeError("Camera Open Failed")

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.stop_signal = False

        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

        if self.pub_fps > 0:
            self.timer = self.create_timer(1.0 / self.pub_fps, self._publish_callback)
        self.info_timer = self.create_timer(1.0, self._publish_info_callback)
        
        self.get_logger().info(f"Node Started. Mode 4 (Flip+Swap) = {self.mode4_active}")

    def load_rectification_maps(self, path):
        # 경로 찾기
        if not os.path.exists(path):
            try:
                pkg_share = get_package_share_directory('stereo_sbs_cam_pub')
                cand = os.path.join(pkg_share, 'config', os.path.basename(path))
                if os.path.exists(cand): path = cand
            except: pass
        
        if not os.path.exists(path):
            cand = os.path.expanduser(f"~/stereo_cam_ws/src/stereo_sbs_cam_pub/config/{os.path.basename(path)}")
            if os.path.exists(cand): path = cand

        if not os.path.exists(path):
            self.get_logger().error(f"NPZ file missing: {path}")
            return

        try:
            data = np.load(path)
            self.get_logger().info(f"NPZ Keys: {list(data.keys())}")
            
            # 키 호환성
            if 'mapL1' in data:
                self.map1x = data['mapL1']; self.map1y = data['mapL2']
                self.map2x = data['mapR1']; self.map2y = data['mapR2']
            elif 'map1x' in data:
                self.map1x = data['map1x']; self.map1y = data['map1y']
                self.map2x = data['map2x']; self.map2y = data['map2y']
            else:
                self.get_logger().error("Unknown map keys in NPZ.")
                return

            self.P1 = data['P1'] if 'P1' in data else np.eye(3,4)
            self.P2 = data['P2'] if 'P2' in data else np.eye(3,4)
            
            # 줌 확인용 로그
            fx = self.P1[0,0]
            self.get_logger().info(f"Map Loaded. FX={fx:.2f} (Zoom Check)")

            if 'image_size' in data:
                 sz = data['image_size']
                 if len(sz) == 2: self.rect_size = tuple(sz)
                 else: self.rect_size = (self.cap_w//2, self.cap_h)
            else:
                self.rect_size = (self.cap_w//2, self.cap_h)
            
            self.maps_loaded = True

        except Exception as e:
            self.get_logger().error(f"Failed to load NPZ: {e}")

    def _capture_loop(self):
        while not self.stop_signal and rclpy.ok():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def _publish_callback(self):
        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
        if frame is None: return

        # =========================================================
        # SBS 파이프라인: Split -> per-eye Rotate 180 -> Remap
        # =========================================================

        # 1. Split SBS
        h, w, _ = frame.shape
        w_half = w // 2
        img_l = frame[:, :w_half].copy()
        img_r = frame[:, w_half:].copy()

        # 2. Rotate each eye by 180 degrees before applying calibration maps.
        #    This is a 180-degree rotation, not a horizontal/vertical flip.
        if self.mode4_active:
            img_l = cv2.rotate(img_l, cv2.ROTATE_180)
            img_r = cv2.rotate(img_r, cv2.ROTATE_180)

        # 3. Rectification (보정)
        if self.maps_loaded:
            img_l = cv2.remap(img_l, self.map1x, self.map1y, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # 4. Resize (Optional)
        if self.resize_scale != 1.0:
            new_w = int(w_half * self.resize_scale)
            new_h = int(h * self.resize_scale)
            img_l = cv2.resize(img_l, (new_w, new_h))
            img_r = cv2.resize(img_r, (new_w, new_h))

        # 5. Publish
        stamp = self.get_clock().now().to_msg()

        msg_lc = self.create_compressed_msg(img_l, stamp, self.fid_l)
        msg_rc = self.create_compressed_msg(img_r, stamp, self.fid_r)
        
        self.pub_l_comp.publish(msg_lc)
        self.pub_r_comp.publish(msg_rc)

    def _publish_info_callback(self):
        if not self.maps_loaded: return
        
        curr_w = int(self.rect_size[0] * self.resize_scale)
        curr_h = int(self.rect_size[1] * self.resize_scale)
        stamp = self.get_clock().now().to_msg()
        
        P1_scaled = self.P1.copy()
        P2_scaled = self.P2.copy()
        if self.resize_scale != 1.0:
            P1_scaled[:2, :] *= self.resize_scale
            P2_scaled[:2, :] *= self.resize_scale
            
        inf_l = self.create_camera_info(stamp, self.fid_l, P1_scaled, (curr_w, curr_h))
        inf_r = self.create_camera_info(stamp, self.fid_r, P2_scaled, (curr_w, curr_h))
        
        self.pub_l_info.publish(inf_l)
        self.pub_r_info.publish(inf_r)

    def create_compressed_msg(self, cv_img, stamp, frame_id):
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.format = "jpeg"
        success, encoded = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_qual])
        if success:
            msg.data = encoded.tobytes()
        return msg

    def create_camera_info(self, stamp, frame_id, P, size):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.width = int(size[0])
        msg.height = int(size[1])
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0]*5
        msg.k = [P[0,0], P[0,1], P[0,2], P[1,0], P[1,1], P[1,2], P[2,0], P[2,1], P[2,2]]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = P.flatten().tolist()
        return msg

    def destroy_node(self):
        self.stop_signal = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        if hasattr(self, 'cap'):
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SBSCamRectifiedPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
