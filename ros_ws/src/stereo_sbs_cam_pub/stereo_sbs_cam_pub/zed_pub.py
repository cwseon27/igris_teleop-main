#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def center_crop_width(img, new_w: int):
    h, w = img.shape[:2]
    if new_w >= w:
        return img
    x0 = (w - new_w) // 2
    return img[:, x0:x0 + new_w]


def letterbox_or_crop_height(img, target_h: int):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    if h < target_h:
        pad = target_h - h
        top = pad // 2
        bottom = pad - top
        return cv2.copyMakeBorder(
            img, top, bottom, 0, 0,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
    # h > target_h: center crop
    y0 = (h - target_h) // 2
    return img[y0:y0 + target_h, :]


def shift_x(img, dx: int):
    if dx == 0:
        return img
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


class SbsCamPublisher(Node):
    def __init__(self):
        super().__init__('sbs_cam_publisher')

        # -------- Parameters --------
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 2560)     # SBS면 보통 2560x720(=1280x720*2)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('fourcc', 'MJPG')  # MJPG / YUYV 등
        self.declare_parameter('sbs', True)       # side-by-side 분리
        self.declare_parameter('publish_right', True)

        # VR용 전처리(튜토리얼의 "FOV 맞춤 + 스케일/크롭"을 OpenCV로 구현)
        self.declare_parameter('vr_preprocess', True)
        self.declare_parameter('target_eye_width', 1024)
        self.declare_parameter('target_eye_height', 1024)
        self.declare_parameter('zed_hfov_deg', 90.0)      # 입력 카메라 HFOV(대략값 가능)
        self.declare_parameter('target_hfov_deg', 90.0)   # VR에서 공통으로 쓸 HFOV
        self.declare_parameter('hit_px', 0)               # 0이면 비활성(가상 수렴용 수평 이동)

        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.fourcc = self.get_parameter('fourcc').get_parameter_value().string_value
        self.sbs = bool(self.get_parameter('sbs').value)
        self.publish_right = bool(self.get_parameter('publish_right').value)

        self.vr_preprocess = bool(self.get_parameter('vr_preprocess').value)
        self.tgt_w = int(self.get_parameter('target_eye_width').value)
        self.tgt_h = int(self.get_parameter('target_eye_height').value)
        self.zed_hfov = float(self.get_parameter('zed_hfov_deg').value)
        self.tgt_hfov = float(self.get_parameter('target_hfov_deg').value)
        self.hit_px = int(self.get_parameter('hit_px').value)

        # -------- QoS (camera default) --------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.pub_left = self.create_publisher(Image, '/left/image_raw', qos)
        self.pub_right = self.create_publisher(Image, '/right/image_raw', qos) if self.publish_right else None

        self.bridge = CvBridge()

        # -------- OpenCV capture --------
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera device: {self.device}')

        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.get_logger().info(
            f'Opened {self.device}: {real_w}x{real_h} @ {real_fps:.2f}fps, '
            f'FOURCC={self.fourcc}, SBS={self.sbs}, VR_PRE={self.vr_preprocess}, '
            f'TGT={self.tgt_w}x{self.tgt_h}, HFOV={self.zed_hfov}->{self.tgt_hfov}, HIT={self.hit_px}px'
        )

        period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(period, self.tick)

    def vr_process_eye(self, eye_img: np.ndarray) -> np.ndarray:
        """
        튜토리얼의 아이디어를 OpenCV로 구현:
        1) target_hfov/zed_hfov 비율로 가로(center) 크롭(usefulWidth)
        2) target_eye_width에 맞춰 리사이즈
        3) 세로는 target_eye_height에 맞춰 letterbox(패딩) 또는 center-crop
        """
        h, w = eye_img.shape[:2]
        if w <= 0 or h <= 0:
            return eye_img

        # useful_w = w * target_hfov / zed_hfov (클램프)
        zed_hfov = max(self.zed_hfov, 1e-6)
        useful_w = int(w * (self.tgt_hfov / zed_hfov))
        useful_w = max(1, min(useful_w, w))

        cropped = center_crop_width(eye_img, useful_w)

        # 가로 기준 스케일
        scale = self.tgt_w / float(useful_w)
        out_h = max(1, int(h * scale))
        resized = cv2.resize(cropped, (self.tgt_w, out_h), interpolation=cv2.INTER_LINEAR)

        final_img = letterbox_or_crop_height(resized, self.tgt_h)
        return final_img

    def tick(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn('Frame read failed')
            return

        stamp = self.get_clock().now().to_msg()

        # ---- Split SBS ----
        if self.sbs:
            h, w = frame.shape[:2]
            if w % 2 == 0:
                left = frame[:, :w // 2]
                right = frame[:, w // 2:]
            else:
                left = frame
                right = None
        else:
            left = frame
            right = None

        # ---- VR preprocess (optional) ----
        if self.vr_preprocess:
            left_p = self.vr_process_eye(left)
            right_p = self.vr_process_eye(right) if right is not None else None

            # HIT(가상 수렴) - 좌/우를 서로 반대 방향으로 약간 이동
            if self.hit_px != 0 and right_p is not None:
                left_p = shift_x(left_p, +self.hit_px // 2)
                right_p = shift_x(right_p, -self.hit_px // 2)

            left = left_p
            right = right_p

        # ---- Publish ----
        msg_left = self.bridge.cv2_to_imgmsg(left, encoding='bgr8')
        msg_left.header.stamp = stamp
        msg_left.header.frame_id = 'left_camera'
        self.pub_left.publish(msg_left)

        if self.pub_right is not None and right is not None:
            msg_right = self.bridge.cv2_to_imgmsg(right, encoding='bgr8')
            msg_right.header.stamp = stamp
            msg_right.header.frame_id = 'right_camera'
            self.pub_right.publish(msg_right)

    def destroy_node(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = SbsCamPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
