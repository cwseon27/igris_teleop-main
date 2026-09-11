#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import mediapipe as mp

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseArray

from .camera_capture import RecoveringCameraCapture, normalize_camera_device


# MediaPipe Hands landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
LITTLE_TIP = 20
TIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, LITTLE_TIP]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
BGR_COLOR = {"Right": (0, 0, 255), "Left": (255, 0, 0)}
TRAPEZOID_WIDTH = 640
TRAPEZOID_HEIGHT = 480
TRAPEZOID_BOTTOM_WIDTH = 320
TRAPEZOID_CANVAS_WIDTH = 1280
TRAPEZOID_CANVAS_HEIGHT = 720
TRAPEZOID_DISPLAY_SCALE = 1.5


def swap_lr(label: str) -> str:
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def normalize_hand_label(value: str) -> str:
    lowered = str(value).strip().lower()
    if lowered == "right":
        return "Right"
    if lowered == "left":
        return "Left"
    if lowered == "any":
        return "Any"
    raise ValueError("target_hand must be one of: Right, Left, Any")


def img_landmarks_to_px(hand_landmarks, w: int, h: int) -> np.ndarray:
    pts = np.zeros((21, 2), dtype=np.int32)
    for i, lm in enumerate(hand_landmarks.landmark):
        pts[i, 0] = int(lm.x * w)
        pts[i, 1] = int(lm.y * h)
    return pts


def world_landmarks_to_np(world_landmarks) -> np.ndarray:
    xyz = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(world_landmarks.landmark):
        xyz[i] = (lm.x, lm.y, lm.z)
    return xyz


def trapezoid_on_black_canvas(
    frame: np.ndarray,
    bottom_width: int = TRAPEZOID_BOTTOM_WIDTH,
    canvas_width: int = TRAPEZOID_CANVAS_WIDTH,
    canvas_height: int = TRAPEZOID_CANVAS_HEIGHT,
    display_scale: float = TRAPEZOID_DISPLAY_SCALE,
) -> np.ndarray:
    """Match mediapipe_webcam_test.py: trapezoid warp, scale, centered black canvas."""
    bottom_width = int(np.clip(bottom_width, 1, TRAPEZOID_WIDTH))
    canvas_width = max(1, int(canvas_width))
    canvas_height = max(1, int(canvas_height))
    display_scale = max(0.01, float(display_scale))

    resized = cv2.resize(frame, (TRAPEZOID_WIDTH, TRAPEZOID_HEIGHT), interpolation=cv2.INTER_LINEAR)
    bottom_inset = (TRAPEZOID_WIDTH - bottom_width) / 2.0
    src = np.float32([
        [0, 0],
        [TRAPEZOID_WIDTH - 1, 0],
        [TRAPEZOID_WIDTH - 1, TRAPEZOID_HEIGHT - 1],
        [0, TRAPEZOID_HEIGHT - 1],
    ])
    dst = np.float32([
        [0, 0],
        [TRAPEZOID_WIDTH - 1, 0],
        [TRAPEZOID_WIDTH - bottom_inset - 1, TRAPEZOID_HEIGHT - 1],
        [bottom_inset, TRAPEZOID_HEIGHT - 1],
    ])
    transform = cv2.getPerspectiveTransform(src, dst)
    trapezoid = cv2.warpPerspective(
        resized,
        transform,
        (TRAPEZOID_WIDTH, TRAPEZOID_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    output_width = max(1, int(TRAPEZOID_WIDTH * display_scale))
    output_height = max(1, int(TRAPEZOID_HEIGHT * display_scale))
    trapezoid = cv2.resize(trapezoid, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    paste_width = min(output_width, canvas_width)
    paste_height = min(output_height, canvas_height)
    src_x = max(0, (output_width - paste_width) // 2)
    src_y = max(0, (output_height - paste_height) // 2)
    dst_x = max(0, (canvas_width - paste_width) // 2)
    dst_y = max(0, (canvas_height - paste_height) // 2)
    canvas[dst_y:dst_y + paste_height, dst_x:dst_x + paste_width] = trapezoid[
        src_y:src_y + paste_height,
        src_x:src_x + paste_width,
    ]
    return canvas


def make_pose(x: float, y: float, z: float) -> Pose:
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = float(z)
    p.orientation.x = 0.0
    p.orientation.y = 0.0
    p.orientation.z = 0.0
    p.orientation.w = 1.0
    return p


class MediaPipeHandPosePublisher(Node):
    """Publish MediaPipe wrist + fingertips in the same PoseArray layout as the VR/OpenXR hand topic.

    Output PoseArray layout:
      poses[0] = wrist, placed at (0, 0, 0)
      poses[1] = thumb_tip relative to wrist
      poses[2] = index_tip relative to wrist
      poses[3] = middle_tip relative to wrist
      poses[4] = ring_tip relative to wrist
      poses[5] = little_tip relative to wrist

    This intentionally matches the Unity/OpenXR publisher assumption used by the hand calibration
    and IGRIS hand viewer pipeline, where fingertip positions are already wrist-relative.
    """

    def __init__(self):
        super().__init__('mediapipe_hand_pose_publisher')

        # Camera / input params
        self.declare_parameter('device', '0', ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('camera_retry_interval_s', 1.0)
        self.declare_parameter('video', '')
        self.declare_parameter('loop_video', False)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('max_hands', 1)
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('model_complexity', 1)
        self.declare_parameter('mirror', False)
        self.declare_parameter('swap_lr', False)
        self.declare_parameter('trapezoid_preprocess', True)
        self.declare_parameter('trapezoid_bottom_width', TRAPEZOID_BOTTOM_WIDTH)
        self.declare_parameter('trapezoid_canvas_width', TRAPEZOID_CANVAS_WIDTH)
        self.declare_parameter('trapezoid_canvas_height', TRAPEZOID_CANVAS_HEIGHT)
        self.declare_parameter('trapezoid_display_scale', TRAPEZOID_DISPLAY_SCALE)
        self.declare_parameter('target_hand', 'Right')  # Right, Left, Any after swap_lr
        self.declare_parameter('hand_assignment_mode', 'target_best')  # target_best or handedness
        self.declare_parameter('publish_inactive_tracked', False)
        self.declare_parameter('show_image', True)
        self.declare_parameter('draw_landmarks', True)
        self.declare_parameter('show_assignment_debug', True)
        self.declare_parameter('window_name', '')
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('preview_output_dir', '')
        self.declare_parameter('preview_side', '')
        self.declare_parameter('preview_write_hz', 10.0)

        # Topic / frame params
        self.declare_parameter('right_pose_topic', '/right_mediapipe_hand/poses')
        self.declare_parameter('left_pose_topic', '/left_mediapipe_hand/poses')
        self.declare_parameter('right_all_pose_topic', '/right_mediapipe_hand/all_poses')
        self.declare_parameter('left_all_pose_topic', '/left_mediapipe_hand/all_poses')
        self.declare_parameter('right_tracked_topic', '/right_mediapipe_hand/is_tracked')
        self.declare_parameter('left_tracked_topic', '/left_mediapipe_hand/is_tracked')
        self.declare_parameter('right_frame_id', 'mediapipe_right_hand_wrist')
        self.declare_parameter('left_frame_id', 'mediapipe_left_hand_wrist')
        self.declare_parameter('publish_all_landmarks', True)

        # Coordinate params
        self.declare_parameter('coordinate_scale', 1.0)
        self.declare_parameter('invert_x', False)
        self.declare_parameter('invert_y', False)
        self.declare_parameter('invert_z', False)

        self.device = normalize_camera_device(self.get_parameter('device').value)
        self.camera_retry_interval_s = float(self.get_parameter('camera_retry_interval_s').value)
        self.video = str(self.get_parameter('video').value)
        self.loop_video = bool(self.get_parameter('loop_video').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.max_hands = int(self.get_parameter('max_hands').value)
        self.min_det = float(self.get_parameter('min_detection_confidence').value)
        self.min_trk = float(self.get_parameter('min_tracking_confidence').value)
        self.model_complexity = int(self.get_parameter('model_complexity').value)
        self.mirror = bool(self.get_parameter('mirror').value)
        self.swap_lr = bool(self.get_parameter('swap_lr').value)
        self.trapezoid_preprocess = bool(self.get_parameter('trapezoid_preprocess').value)
        self.trapezoid_bottom_width = int(self.get_parameter('trapezoid_bottom_width').value)
        self.trapezoid_canvas_width = int(self.get_parameter('trapezoid_canvas_width').value)
        self.trapezoid_canvas_height = int(self.get_parameter('trapezoid_canvas_height').value)
        self.trapezoid_display_scale = float(self.get_parameter('trapezoid_display_scale').value)
        self.target_hand = normalize_hand_label(str(self.get_parameter('target_hand').value))
        self.hand_assignment_mode = str(self.get_parameter('hand_assignment_mode').value).strip().lower()
        if self.hand_assignment_mode not in ('target_best', 'handedness'):
            raise ValueError("hand_assignment_mode must be one of: target_best, handedness")
        self.publish_inactive_tracked = bool(self.get_parameter('publish_inactive_tracked').value)
        self.show_image = bool(self.get_parameter('show_image').value)
        self.draw_landmarks = bool(self.get_parameter('draw_landmarks').value)
        self.show_assignment_debug = bool(self.get_parameter('show_assignment_debug').value)
        self.window_name = str(self.get_parameter('window_name').value) or f'MediaPipe hand pose bridge - {self.target_hand}'
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        preview_output_dir = str(self.get_parameter('preview_output_dir').value).strip()
        self.preview_output_dir = Path(preview_output_dir).expanduser().resolve() if preview_output_dir else None
        preview_side = str(self.get_parameter('preview_side').value).strip().lower()
        self.preview_side = preview_side if preview_side in ('left', 'right') else self.target_hand.lower()
        self.preview_write_hz = max(0.1, float(self.get_parameter('preview_write_hz').value))
        self.preview_write_period = 1.0 / self.preview_write_hz
        self.preview_last_write_t = 0.0
        if self.preview_output_dir is not None:
            self.preview_output_dir.mkdir(parents=True, exist_ok=True)

        self.scale = float(self.get_parameter('coordinate_scale').value)
        self.invert_x = bool(self.get_parameter('invert_x').value)
        self.invert_y = bool(self.get_parameter('invert_y').value)
        self.invert_z = bool(self.get_parameter('invert_z').value)

        self.right_pose_topic = str(self.get_parameter('right_pose_topic').value)
        self.left_pose_topic = str(self.get_parameter('left_pose_topic').value)
        self.right_all_pose_topic = str(self.get_parameter('right_all_pose_topic').value)
        self.left_all_pose_topic = str(self.get_parameter('left_all_pose_topic').value)
        self.right_tracked_topic = str(self.get_parameter('right_tracked_topic').value)
        self.left_tracked_topic = str(self.get_parameter('left_tracked_topic').value)
        self.right_frame_id = str(self.get_parameter('right_frame_id').value)
        self.left_frame_id = str(self.get_parameter('left_frame_id').value)
        self.publish_all_landmarks = bool(self.get_parameter('publish_all_landmarks').value)

        self.pose_pubs = {
            'Right': self.create_publisher(PoseArray, self.right_pose_topic, 10),
            'Left': self.create_publisher(PoseArray, self.left_pose_topic, 10),
        }
        self.all_pose_pubs = {
            'Right': self.create_publisher(PoseArray, self.right_all_pose_topic, 10),
            'Left': self.create_publisher(PoseArray, self.left_all_pose_topic, 10),
        }
        self.tracked_pubs = {
            'Right': self.create_publisher(Bool, self.right_tracked_topic, 10),
            'Left': self.create_publisher(Bool, self.left_tracked_topic, 10),
        }
        self.frame_ids = {
            'Right': self.right_frame_id,
            'Left': self.left_frame_id,
        }

        self.frame_i = 0
        try:
            self.cap = self.open_capture()
        except Exception:
            self._write_preview_status(camera_ok=False, detections=[], pose_by_label={})
            raise
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_det,
            min_tracking_confidence=self.min_trk,
        )

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self.step)
        self.last_log_t = time.time()

        self.get_logger().info('MediaPipe hand pose publisher started.')
        self.get_logger().info(f'target_hand: {self.target_hand}')
        self.get_logger().info(f'hand_assignment_mode: {self.hand_assignment_mode}')
        self.get_logger().info(f'publish_inactive_tracked: {self.publish_inactive_tracked}')
        self.get_logger().info(f'mirror={self.mirror}, swap_lr={self.swap_lr}')
        self.get_logger().info(
            f'trapezoid_preprocess={self.trapezoid_preprocess}, '
            f'bottom_width={self.trapezoid_bottom_width}, '
            f'canvas={self.trapezoid_canvas_width}x{self.trapezoid_canvas_height}, '
            f'scale={self.trapezoid_display_scale}'
        )
        self.get_logger().info(f'Right output: {self.right_pose_topic}, {self.right_tracked_topic}')
        self.get_logger().info(f'Left output : {self.left_pose_topic}, {self.left_tracked_topic}')
        if self.publish_all_landmarks:
            self.get_logger().info(f'Right all landmarks: {self.right_all_pose_topic}')
            self.get_logger().info(f'Left all landmarks : {self.left_all_pose_topic}')
        self.get_logger().info('PoseArray order: [wrist, thumb_tip, index_tip, middle_tip, ring_tip, little_tip]')
        self.get_logger().info('All PoseArray order: MediaPipe landmarks[0..20], wrist-relative.')
        if self.preview_output_dir is not None:
            self.get_logger().info(
                f'GUI preview: side={self.preview_side}, dir={self.preview_output_dir}, '
                f'rate={self.preview_write_hz:.1f}Hz'
            )

    def open_capture(self):
        if self.video:
            cap = cv2.VideoCapture(self.video)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open video file: {self.video}')
            return cap

        return RecoveringCameraCapture(
            self.device,
            self.width,
            self.height,
            retry_interval_s=self.camera_retry_interval_s,
            logger=self.get_logger(),
        )

    def step(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            now = time.monotonic()
            if now - self.preview_last_write_t >= self.preview_write_period:
                self.preview_last_write_t = now
                self._write_preview_status(camera_ok=False, detections=[], pose_by_label={})
            self.publish_untracked_for_active_labels()
            if self.video and self.loop_video:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            return

        self.frame_i += 1
        raw_frame = frame.copy()
        if self.mirror:
            frame = cv2.flip(frame, 1)
        if self.trapezoid_preprocess:
            frame = trapezoid_on_black_canvas(
                frame,
                bottom_width=self.trapezoid_bottom_width,
                canvas_width=self.trapezoid_canvas_width,
                canvas_height=self.trapezoid_canvas_height,
                display_scale=self.trapezoid_display_scale,
            )
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        pose_by_label: Dict[str, Optional[np.ndarray]] = {'Right': None, 'Left': None}
        all_pose_by_label: Dict[str, Optional[np.ndarray]] = {'Right': None, 'Left': None}
        best_score = {'Right': -1.0, 'Left': -1.0}
        detections: List[Dict[str, object]] = []
        display_detections: List[Dict[str, object]] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            world_list = getattr(results, 'multi_hand_world_landmarks', None)
            if world_list is not None:
                for i, (hand_landmarks, handed) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    if i >= len(world_list) or world_list[i] is None:
                        continue

                    raw_label = handed.classification[0].label  # Right / Left
                    score = float(handed.classification[0].score)
                    label = raw_label
                    if self.swap_lr:
                        label = swap_lr(label)
                    if label not in pose_by_label:
                        continue

                    xyz = world_landmarks_to_np(world_list[i])
                    rel6 = self.extract_wrist_relative_six_points(xyz)
                    rel21 = self.extract_wrist_relative_all_points(xyz)
                    pts_px = img_landmarks_to_px(hand_landmarks, w, h)
                    detections.append(
                        {
                            'raw_label': raw_label,
                            'label': label,
                            'score': score,
                            'rel6': rel6,
                            'rel21': rel21,
                            'pts_px': pts_px,
                        }
                    )

        if self.target_hand in ('Right', 'Left') and self.hand_assignment_mode == 'target_best':
            best_detection = None
            best_detection_score = -1.0
            for det in detections:
                score = float(det['score'])
                if score > best_detection_score:
                    best_detection_score = score
                    best_detection = det
            if best_detection is not None:
                pose_by_label[self.target_hand] = best_detection['rel6']
                all_pose_by_label[self.target_hand] = best_detection['rel21']
                best_score[self.target_hand] = best_detection_score
                display_det = dict(best_detection)
                display_det['publish_label'] = self.target_hand
                display_det['display_text'] = self.display_text(self.target_hand, display_det)
                display_detections.append(display_det)
        else:
            selected_by_label: Dict[str, Dict[str, object]] = {}
            for det in detections:
                label = str(det['label'])
                score = float(det['score'])
                if self.target_hand != 'Any' and label != self.target_hand:
                    continue
                if score > best_score[label]:
                    best_score[label] = score
                    pose_by_label[label] = det['rel6']
                    all_pose_by_label[label] = det['rel21']
                    selected_by_label[label] = det
            for label in ('Right', 'Left'):
                if label not in selected_by_label:
                    continue
                det = dict(selected_by_label[label])
                det['publish_label'] = label
                det['display_text'] = self.display_text(label, det)
                display_detections.append(det)

        # Publish both tracked flags every tick.
        for label in ('Right', 'Left'):
            should_publish_label = (self.target_hand == 'Any' or self.target_hand == label)
            if not should_publish_label and not self.publish_inactive_tracked:
                continue

            tracked = pose_by_label[label] is not None if should_publish_label else False
            self.publish_tracked(label, tracked)
            if tracked and pose_by_label[label] is not None:
                self.publish_pose_array(label, pose_by_label[label])
                if self.publish_all_landmarks and all_pose_by_label[label] is not None:
                    self.publish_all_pose_array(label, all_pose_by_label[label])

        annotated_frame = frame.copy()
        if self.show_image or self.preview_output_dir is not None:
            self.draw_status(annotated_frame, pose_by_label, display_detections)
        self._write_preview_frames(
            raw_frame=raw_frame,
            preprocessed_frame=frame,
            annotated_frame=annotated_frame,
            detections=display_detections,
            pose_by_label=pose_by_label,
        )

        if self.show_image:
            cv2.imshow(f'{self.window_name} - q/ESC', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                rclpy.shutdown()

        now = time.time()
        if now - self.last_log_t > 2.0:
            self.last_log_t = now
            published = ', '.join([f'{k}={pose_by_label[k] is not None}' for k in ('Right', 'Left')])
            detected = ', '.join(
                [
                    f"{det['raw_label']}->{det['label']}({float(det['score']):.2f})"
                    for det in detections
                ]
            ) or 'none'
            displayed = ', '.join([str(det['display_text']) for det in display_detections]) or 'none'
            self.get_logger().info(f'detected: {detected}; selected: {displayed}; published: {published}')

    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        tmp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, path)

    def _write_preview_jpeg(self, stage: str, frame: np.ndarray) -> None:
        if self.preview_output_dir is None:
            return
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        self._atomic_write_bytes(
            self.preview_output_dir / f'{self.preview_side}_{stage}.jpg',
            encoded.tobytes(),
        )

    def _write_preview_status(self, *, camera_ok: bool, detections, pose_by_label) -> None:
        if self.preview_output_dir is None:
            return
        detected = [
            {
                'raw_label': str(det.get('raw_label', '')),
                'label': str(det.get('label', '')),
                'publish_label': str(det.get('publish_label', '')),
                'score': float(det.get('score', 0.0)),
            }
            for det in detections
        ]
        tracked = {
            label.lower(): bool(pose_by_label.get(label) is not None)
            for label in ('Left', 'Right')
        }
        payload = {
            'updated_at': time.time(),
            'camera_ok': bool(camera_ok),
            'side': self.preview_side,
            'device': self.device,
            'frame': self.frame_i,
            'mirror': self.mirror,
            'swap_lr': self.swap_lr,
            'trapezoid_preprocess': self.trapezoid_preprocess,
            'trapezoid_bottom_width': self.trapezoid_bottom_width,
            'detections': detected,
            'tracked': tracked,
        }
        self._atomic_write_bytes(
            self.preview_output_dir / f'{self.preview_side}_status.json',
            json.dumps(payload, ensure_ascii=True, separators=(',', ':')).encode('utf-8'),
        )

    def _write_preview_frames(
        self,
        *,
        raw_frame: np.ndarray,
        preprocessed_frame: np.ndarray,
        annotated_frame: np.ndarray,
        detections,
        pose_by_label,
    ) -> None:
        if self.preview_output_dir is None:
            return
        now = time.monotonic()
        if now - self.preview_last_write_t < self.preview_write_period:
            return
        self.preview_last_write_t = now
        try:
            self._write_preview_jpeg('raw', raw_frame)
            self._write_preview_jpeg('trapezoid', preprocessed_frame)
            self._write_preview_jpeg('mediapipe', annotated_frame)
            self._write_preview_status(
                camera_ok=True,
                detections=detections,
                pose_by_label=pose_by_label,
            )
        except Exception as exc:
            self.get_logger().warning(f'GUI preview write failed: {exc}')

    def extract_wrist_relative_six_points(self, world_xyz: np.ndarray) -> np.ndarray:
        """Return 6x3 array matching VR topic order.

        MediaPipe world landmarks may use a hand-centered origin. This function removes
        the origin by subtracting wrist position, so final fingertip points are wrist-relative.
        """
        wrist = world_xyz[WRIST].astype(np.float32)
        points = np.zeros((6, 3), dtype=np.float32)
        points[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for out_i, mp_i in enumerate(TIP_INDICES, start=1):
            points[out_i] = world_xyz[mp_i].astype(np.float32) - wrist

        points *= self.scale
        if self.invert_x:
            points[:, 0] *= -1.0
        if self.invert_y:
            points[:, 1] *= -1.0
        if self.invert_z:
            points[:, 2] *= -1.0
        return points

    def extract_wrist_relative_all_points(self, world_xyz: np.ndarray) -> np.ndarray:
        wrist = world_xyz[WRIST].astype(np.float32)
        points = world_xyz.astype(np.float32, copy=True) - wrist.reshape(1, 3)
        points *= self.scale
        if self.invert_x:
            points[:, 0] *= -1.0
        if self.invert_y:
            points[:, 1] *= -1.0
        if self.invert_z:
            points[:, 2] *= -1.0
        return points.astype(np.float32)

    def display_text(self, publish_label: str, det: Dict[str, object]) -> str:
        if not self.show_assignment_debug:
            return publish_label
        raw_label = str(det.get('raw_label', publish_label))
        adjusted_label = str(det.get('label', raw_label))
        score = float(det.get('score', 0.0))
        return f'{raw_label}->{adjusted_label}->{publish_label} {score:.2f}'

    def publish_tracked(self, label: str, tracked: bool):
        msg = Bool()
        msg.data = bool(tracked)
        self.tracked_pubs[label].publish(msg)

    def publish_untracked_for_active_labels(self):
        for label in ('Right', 'Left'):
            if self.target_hand == 'Any' or self.target_hand == label or self.publish_inactive_tracked:
                self.publish_tracked(label, False)

    def publish_pose_array(self, label: str, points: np.ndarray):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_ids[label]
        msg.poses = [make_pose(x, y, z) for x, y, z in points]
        self.pose_pubs[label].publish(msg)

    def publish_all_pose_array(self, label: str, points: np.ndarray):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_ids[label]
        msg.poses = [make_pose(x, y, z) for x, y, z in points]
        self.all_pose_pubs[label].publish(msg)

    def draw_2d_skeleton(self, frame: np.ndarray, pts_px: np.ndarray, label: str):
        color = BGR_COLOR.get(label, (0, 255, 0))
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts_px[a]), tuple(pts_px[b]), color, 2, cv2.LINE_AA)
        for x, y in pts_px:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

    def draw_status(self, frame, pose_by_label, detections):
        y = 30
        cv2.putText(
            frame,
            f'target={self.target_hand} mode={self.hand_assignment_mode} mirror={self.mirror} swap_lr={self.swap_lr}',
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        y += 30

        if self.draw_landmarks:
            for det in detections:
                label = str(det['publish_label'])
                display_text = str(det['display_text'])
                pts_px = det['pts_px']
                self.draw_2d_skeleton(frame, pts_px, label)
                wx, wy = pts_px[WRIST]
                cv2.putText(
                    frame,
                    display_text,
                    (int(wx), int(max(0, wy - 10))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    BGR_COLOR.get(label, (0, 255, 0)),
                    2,
                    cv2.LINE_AA,
                )

        for label in ('Right', 'Left'):
            ok = pose_by_label[label] is not None
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.putText(frame, f'{label}: {"tracked" if ok else "none"}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30

    def destroy_node(self):
        try:
            if hasattr(self, 'hands') and self.hands is not None:
                self.hands.close()
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MediaPipeHandPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
