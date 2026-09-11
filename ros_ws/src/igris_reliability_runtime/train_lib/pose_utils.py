from __future__ import annotations

import numpy as np


POSE_FLAT_DIM = 7
FEATURE_MODE_HMD_RELATIVE_MOTION = "hmd_relative_motion_tracking_v2"
EPS = 1e-8
HAND_JOINT_COUNT = 21
COMPACT_HAND_POINT_COUNT = 6
COMPACT_FROM_FULL_INDICES = (0, 4, 8, 12, 16, 20)
COMPACT_FINGER_JOINT_GROUPS = {
    "thumb": (1,),
    "index": (2,),
    "middle": (3,),
    "ring": (4,),
    "little": (5,),
}
COMPACT_FINGER_SEGMENT_GROUPS = {
    "thumb": ((0, 1),),
    "index": ((0, 2),),
    "middle": ((0, 3),),
    "ring": ((0, 4),),
    "little": ((0, 5),),
}
PALM_ALIGNMENT_JOINTS = (0, 5, 9, 13, 17)
FINGER_JOINT_GROUPS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}
FINGER_SEGMENT_GROUPS = {
    "thumb": ((0, 1), (1, 2), (2, 3), (3, 4)),
    "index": ((0, 5), (5, 6), (6, 7), (7, 8)),
    "middle": ((0, 9), (9, 10), (10, 11), (11, 12)),
    "ring": ((0, 13), (13, 14), (14, 15), (15, 16)),
    "little": ((0, 17), (17, 18), (18, 19), (19, 20)),
}
FINGER_BASE_ANGLE_TRIPLES = {
    "thumb": (0, 1, 4),
    "index": (0, 5, 8),
    "middle": (0, 9, 12),
    "ring": (0, 13, 16),
    "little": (0, 17, 20),
}


def pose_to_flat_from_fields(position, orientation) -> np.ndarray:
    """Flatten position/orientation fields in the shared collector/inference order."""
    return np.asarray(
        [
            position.x,
            position.y,
            position.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ],
        dtype=np.float32,
    )


def validate_vector(vec) -> np.ndarray:
    """Return a finite 1D float32 vector or raise ValueError."""
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1D vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector contains NaN or inf")
    return arr


def hand_joint_points_to_array(points, expected_count: int | None = HAND_JOINT_COUNT) -> np.ndarray:
    """Convert point-like fields with x/y/z attributes into a finite (N, 3) array."""
    joints = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    if expected_count is not None and joints.shape != (int(expected_count), 3):
        raise ValueError(f"expected hand joints shape ({expected_count}, 3), got {joints.shape}")
    if expected_count is None and (joints.ndim != 2 or joints.shape[1] != 3):
        raise ValueError(f"expected hand joints shape (N, 3), got {joints.shape}")
    if not np.all(np.isfinite(joints)):
        raise ValueError("hand joints contain NaN or inf")
    return joints


def _hand_joint_scale(joints: np.ndarray) -> float:
    arr = np.asarray(joints, dtype=np.float32)
    wrist = arr[0]
    if arr.shape[0] >= HAND_JOINT_COUNT:
        scale_indices = list(PALM_ALIGNMENT_JOINTS)[1:]
    elif arr.shape[0] == COMPACT_HAND_POINT_COUNT:
        scale_indices = list(range(1, COMPACT_HAND_POINT_COUNT))
    else:
        scale_indices = list(range(1, arr.shape[0]))
    palm_distances = np.linalg.norm(arr[scale_indices] - wrist.reshape(1, 3), axis=1)
    valid = palm_distances[palm_distances > EPS]
    if valid.size:
        return float(np.median(valid))
    all_distances = np.linalg.norm(arr - wrist.reshape(1, 3), axis=1)
    valid = all_distances[all_distances > EPS]
    if valid.size:
        return float(np.median(valid))
    return 1.0


def _align_hand_joints_to_reference(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate source wrist-relative joints to best match reference palm joints."""
    src = np.asarray(source, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    indices = list(PALM_ALIGNMENT_JOINTS)
    src_basis = src[indices]
    ref_basis = ref[indices]
    if np.linalg.norm(src_basis) <= EPS or np.linalg.norm(ref_basis) <= EPS:
        return src.copy()

    try:
        covariance = src_basis.T @ ref_basis
        u, _, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        return (src @ rotation).astype(np.float32)
    except np.linalg.LinAlgError:
        return src.copy()


def _segment_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= EPS or norm_b <= EPS:
        return 0.0
    cos_angle = float(np.dot(a, b) / (norm_a * norm_b))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _finger_base_angles(joints: np.ndarray) -> dict[str, float]:
    arr = np.asarray(joints, dtype=np.float32)
    if arr.shape != (HAND_JOINT_COUNT, 3):
        raise ValueError(
            f"finger_base_angle requires full {HAND_JOINT_COUNT}-point hand, got {arr.shape}"
        )
    angles = {}
    for finger_name, (wrist_idx, base_idx, tip_idx) in FINGER_BASE_ANGLE_TRIPLES.items():
        base = arr[base_idx]
        wrist_vec = arr[wrist_idx] - base
        tip_vec = arr[tip_idx] - base
        angles[finger_name] = _segment_angle_degrees(wrist_vec, tip_vec)
    return angles


def _to_compact_hand_points(joints: np.ndarray) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float32)
    if arr.shape == (COMPACT_HAND_POINT_COUNT, 3):
        return arr
    if arr.shape == (HAND_JOINT_COUNT, 3):
        return arr[list(COMPACT_FROM_FULL_INDICES)]
    raise ValueError(
        f"expected {HAND_JOINT_COUNT} full joints or {COMPACT_HAND_POINT_COUNT} compact points, got {arr.shape}"
    )


def _prepare_comparable_hand_points(head: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    head = np.asarray(head, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if head.shape == (HAND_JOINT_COUNT, 3) and reference.shape == (HAND_JOINT_COUNT, 3):
        return head, reference, "full21"
    return _to_compact_hand_points(head), _to_compact_hand_points(reference), "compact6"


def compare_hand_joints_for_occlusion_label(
    head_joints,
    reference_joints,
    distance_threshold: float = 0.35,
    angle_threshold_deg: float = 35.0,
    align_palm: bool = True,
    compare_mode: str = "finger_base_angle",
) -> tuple[int, dict]:
    """Return label 1 when two hand pose sets agree, else 0.

    The intended use is to treat the lower/reference VR as a visibility oracle.
    If any one finger differs from the head-mounted VR by the selected threshold,
    the hand is labelled unreliable. Supports either 21-point full hands or the
    compact 6-point [wrist, five fingertips] layout used by OpenXR/MediaPipe topics.
    """
    compare_mode = str(compare_mode).strip().lower()
    if compare_mode not in ("finger_base_angle", "angle_only", "distance_angle", "distance_only"):
        raise ValueError(
            "compare_mode must be one of: finger_base_angle, angle_only, distance_angle, distance_only"
        )
    use_base_angle = compare_mode == "finger_base_angle"
    use_distance = compare_mode in ("distance_angle", "distance_only")
    use_angle = compare_mode in ("angle_only", "distance_angle")

    head = np.asarray(head_joints, dtype=np.float32)
    reference = np.asarray(reference_joints, dtype=np.float32)
    if not np.all(np.isfinite(head)) or not np.all(np.isfinite(reference)):
        raise ValueError("hand joints contain NaN or inf")

    head, reference, layout = _prepare_comparable_hand_points(head, reference)
    if use_base_angle and layout != "full21":
        raise ValueError("finger_base_angle compare_mode requires full 21-point hands")
    effective_align_palm = bool(align_palm) and layout == "full21"
    aligned_head = _align_hand_joints_to_reference(head, reference) if effective_align_palm else head
    scale = None
    normalized_delta = None
    if use_distance:
        scale = max((_hand_joint_scale(aligned_head) + _hand_joint_scale(reference)) * 0.5, EPS)
        normalized_delta = (aligned_head - reference) / float(scale)

    bad_fingers = []
    max_distance_by_finger = {}
    max_angle_by_finger = {}
    head_base_angle_by_finger = {}
    reference_base_angle_by_finger = {}
    finger_joint_groups = FINGER_JOINT_GROUPS if layout == "full21" else COMPACT_FINGER_JOINT_GROUPS
    finger_segment_groups = FINGER_SEGMENT_GROUPS if layout == "full21" else COMPACT_FINGER_SEGMENT_GROUPS
    if use_base_angle:
        head_base_angle_by_finger = _finger_base_angles(aligned_head)
        reference_base_angle_by_finger = _finger_base_angles(reference)

    for finger_name, joint_indices in finger_joint_groups.items():
        max_distance = None
        if use_distance:
            distances = np.linalg.norm(normalized_delta[list(joint_indices)], axis=1)
            max_distance = float(np.max(distances)) if distances.size else 0.0
        max_distance_by_finger[finger_name] = max_distance

        max_angle = None
        if use_base_angle:
            max_angle = abs(
                float(head_base_angle_by_finger[finger_name])
                - float(reference_base_angle_by_finger[finger_name])
            )
        elif use_angle:
            angles = []
            for start, end in finger_segment_groups[finger_name]:
                head_vec = aligned_head[end] - aligned_head[start]
                ref_vec = reference[end] - reference[start]
                angles.append(_segment_angle_degrees(head_vec, ref_vec))
            max_angle = float(np.max(angles)) if angles else 0.0
        max_angle_by_finger[finger_name] = max_angle

        distance_bad = use_distance and max_distance is not None and max_distance > float(distance_threshold)
        angle_bad = (
            (use_base_angle or use_angle)
            and max_angle is not None
            and max_angle > float(angle_threshold_deg)
        )
        if distance_bad or angle_bad:
            bad_fingers.append(finger_name)

    details = {
        "bad_fingers": bad_fingers,
        "max_distance_by_finger": max_distance_by_finger,
        "max_angle_by_finger": max_angle_by_finger,
        "head_base_angle_by_finger": head_base_angle_by_finger,
        "reference_base_angle_by_finger": reference_base_angle_by_finger,
        "scale": None if scale is None else float(scale),
        "compare_mode": compare_mode,
        "layout": layout,
        "distance_threshold": float(distance_threshold),
        "angle_threshold_deg": float(angle_threshold_deg),
        "align_palm": bool(effective_align_palm),
    }
    return (0 if bad_fingers else 1), details


def split_pose_flat(pose_flat) -> tuple[np.ndarray, np.ndarray]:
    arr = validate_vector(pose_flat)
    if arr.size == 0 or arr.size % POSE_FLAT_DIM != 0:
        raise ValueError(f"pose vector length must be a positive multiple of {POSE_FLAT_DIM}, got {arr.size}")
    poses = arr.reshape(-1, POSE_FLAT_DIM)
    positions = poses[:, 0:3].astype(np.float32, copy=False)
    orientations = normalize_quaternions(poses[:, 3:7])
    return positions, orientations


def normalize_quaternions(quaternions) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, 4)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"expected quaternion shape (N, 4), got {q.shape}")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm < EPS):
        raise ValueError("zero-length quaternion")
    q = q / norm
    return q.astype(np.float32)


def canonicalize_quaternions(quaternions, previous=None) -> np.ndarray:
    q = normalize_quaternions(quaternions)
    if previous is not None:
        prev = normalize_quaternions(previous)
        if prev.shape == q.shape:
            signs = np.where(np.sum(prev * q, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
            return (q * signs).astype(np.float32)
    signs = np.where(q[:, 3:4] < 0.0, -1.0, 1.0)
    return (q * signs).astype(np.float32)


def quaternion_conjugate(quaternions) -> np.ndarray:
    q = normalize_quaternions(quaternions)
    out = q.copy()
    out[:, 0:3] *= -1.0
    return out.astype(np.float32)


def quaternion_multiply(q1, q2) -> np.ndarray:
    a = normalize_quaternions(q1)
    b = normalize_quaternions(q2)
    if a.shape != b.shape:
        if a.shape[0] == 1:
            a = np.repeat(a, b.shape[0], axis=0)
        elif b.shape[0] == 1:
            b = np.repeat(b, a.shape[0], axis=0)
        else:
            raise ValueError(f"cannot broadcast quaternion shapes {a.shape} and {b.shape}")

    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.column_stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )
    return normalize_quaternions(out)


def rotate_vectors_by_quaternion(vectors, quaternion) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, 3)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"expected vector shape (N, 3), got {vectors.shape}")

    q = normalize_quaternions(quaternion)
    if q.shape[0] == 1 and vectors.shape[0] > 1:
        q = np.repeat(q, vectors.shape[0], axis=0)
    if q.shape[0] != vectors.shape[0]:
        raise ValueError(f"cannot broadcast quaternion shape {q.shape} to vectors {vectors.shape}")

    q_vec = q[:, 0:3]
    q_w = q[:, 3:4]
    t = 2.0 * np.cross(q_vec, vectors)
    rotated = vectors + q_w * t + np.cross(q_vec, t)
    return rotated.astype(np.float32)


def quaternion_angular_velocity(previous_quat, current_quat, dt: float) -> np.ndarray:
    curr = canonicalize_quaternions(current_quat, previous_quat)
    prev = normalize_quaternions(previous_quat)
    if curr.shape != prev.shape:
        raise ValueError(f"quaternion shapes differ: previous={prev.shape}, current={curr.shape}")
    if dt <= EPS:
        return np.zeros((curr.shape[0], 3), dtype=np.float32)

    delta = quaternion_multiply(curr, quaternion_conjugate(prev))
    delta = canonicalize_quaternions(delta)
    vec = delta[:, 0:3]
    w = np.clip(delta[:, 3], -1.0, 1.0)
    vec_norm = np.linalg.norm(vec, axis=1)
    angle = 2.0 * np.arctan2(vec_norm, w)
    axis = np.zeros_like(vec)
    valid = vec_norm > EPS
    axis[valid] = vec[valid] / vec_norm[valid, None]
    return (axis * (angle[:, None] / float(dt))).astype(np.float32)


def _hand_relative_motion_features(
    hand_pose_flat,
    hmd_position: np.ndarray,
    hmd_inverse_orientation: np.ndarray,
    dt: float | None,
    previous_hand_state: dict | None,
) -> tuple[np.ndarray, dict]:
    hand_positions, hand_orientations = split_pose_flat(hand_pose_flat)
    previous_orientations = None
    if previous_hand_state is not None:
        previous_orientations = previous_hand_state.get("orientations")
    hand_orientations = canonicalize_quaternions(hand_orientations, previous_orientations)

    world_delta = hand_positions - hmd_position.reshape(1, 3)
    relative_positions = rotate_vectors_by_quaternion(world_delta, hmd_inverse_orientation)

    if (
        previous_hand_state is None
        or dt is None
        or dt <= EPS
        or previous_hand_state.get("relative_positions", np.empty((0, 3))).shape != relative_positions.shape
    ):
        linear_velocity = np.zeros_like(relative_positions, dtype=np.float32)
        angular_velocity = np.zeros((hand_orientations.shape[0], 3), dtype=np.float32)
    else:
        linear_velocity = (
            (relative_positions - previous_hand_state["relative_positions"]) / float(dt)
        ).astype(np.float32)
        angular_velocity = quaternion_angular_velocity(
            previous_hand_state["orientations"],
            hand_orientations,
            float(dt),
        )

    feature = np.concatenate(
        [
            relative_positions.reshape(-1),
            hand_orientations.reshape(-1),
            linear_velocity.reshape(-1),
            angular_velocity.reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)
    state = {
        "relative_positions": relative_positions.astype(np.float32, copy=True),
        "orientations": hand_orientations.astype(np.float32, copy=True),
    }
    return validate_vector(feature), state


class HmdRelativeMotionFeatureBuilder:
    """Build per-frame features shared by collection and inference."""

    def __init__(self, pose_stale_sec: float = 0.2):
        self.pose_stale_sec = float(pose_stale_sec)
        self.previous_timestamp = None
        self.previous_hmd_orientation = None
        self.previous_left_state = None
        self.previous_right_state = None
        self.hmd_feature_dim = None
        self.left_hand_feature_dim = None
        self.right_hand_feature_dim = None

    def _availability_features(self, is_tracked: bool, pose_age_sec: float | None) -> np.ndarray:
        pose_age = 0.0 if pose_age_sec is None else max(0.0, float(pose_age_sec))
        if self.pose_stale_sec > EPS:
            pose_age_norm = min(pose_age / self.pose_stale_sec, 1.0)
            pose_valid = 1.0 if pose_age <= self.pose_stale_sec else 0.0
        else:
            pose_age_norm = 0.0
            pose_valid = 1.0
        return np.asarray(
            [
                1.0 if bool(is_tracked) else 0.0,
                pose_valid,
                pose_age_norm,
            ],
            dtype=np.float32,
        )

    def build(
        self,
        hmd_pose_flat,
        left_hand_pose_flat,
        right_hand_pose_flat,
        timestamp: float | None = None,
        left_is_tracked: bool = True,
        right_is_tracked: bool = True,
        left_pose_age_sec: float | None = 0.0,
        right_pose_age_sec: float | None = 0.0,
    ):
        hmd_positions, hmd_orientations = split_pose_flat(hmd_pose_flat)
        if hmd_positions.shape[0] != 1:
            raise ValueError(f"hmd pose must contain exactly one pose, got {hmd_positions.shape[0]}")

        hmd_orientation = canonicalize_quaternions(hmd_orientations, self.previous_hmd_orientation)
        hmd_inverse_orientation = quaternion_conjugate(hmd_orientation)

        dt = None
        if timestamp is not None and self.previous_timestamp is not None:
            dt = float(timestamp) - float(self.previous_timestamp)

        hmd_feature = hmd_orientation.reshape(-1).astype(np.float32)
        left_feature, left_state = _hand_relative_motion_features(
            left_hand_pose_flat,
            hmd_positions[0],
            hmd_inverse_orientation,
            dt,
            self.previous_left_state,
        )
        left_feature = np.concatenate(
            [
                left_feature,
                self._availability_features(left_is_tracked, left_pose_age_sec),
            ],
            axis=0,
        ).astype(np.float32)
        right_feature, right_state = _hand_relative_motion_features(
            right_hand_pose_flat,
            hmd_positions[0],
            hmd_inverse_orientation,
            dt,
            self.previous_right_state,
        )
        right_feature = np.concatenate(
            [
                right_feature,
                self._availability_features(right_is_tracked, right_pose_age_sec),
            ],
            axis=0,
        ).astype(np.float32)

        frame_feature = validate_vector(
            np.concatenate([hmd_feature, left_feature, right_feature], axis=0)
        )

        self.previous_timestamp = None if timestamp is None else float(timestamp)
        self.previous_hmd_orientation = hmd_orientation.astype(np.float32, copy=True)
        self.previous_left_state = left_state
        self.previous_right_state = right_state
        self.hmd_feature_dim = int(hmd_feature.shape[0])
        self.left_hand_feature_dim = int(left_feature.shape[0])
        self.right_hand_feature_dim = int(right_feature.shape[0])
        return frame_feature

    def metadata(self) -> dict:
        return {
            "feature_mode": FEATURE_MODE_HMD_RELATIVE_MOTION,
            "hmd_feature_dim": int(self.hmd_feature_dim or 0),
            "left_hand_feature_dim": int(self.left_hand_feature_dim or 0),
            "right_hand_feature_dim": int(self.right_hand_feature_dim or 0),
            "feature_description": (
                "Per frame: hmd orientation quaternion, then for each hand pose "
                "HMD-local relative position, hand orientation quaternion, "
                "HMD-local linear velocity, quaternion angular velocity, plus "
                "hand availability features [is_tracked, pose_valid, pose_age_norm]."
            ),
            "pose_stale_sec": float(self.pose_stale_sec),
        }
