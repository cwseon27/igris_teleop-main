from __future__ import annotations

import math


# ZED 2i-like stereo pair. The simulation uses the existing 720p camera SHM.
STEREO_CAMERA_WIDTH = 1280
STEREO_CAMERA_HEIGHT = 720
STEREO_CAMERA_HZ = 30.0
STEREO_CAMERA_BASELINE_M = 0.120
STEREO_CAMERA_MIN_BASELINE_M = 0.040
STEREO_CAMERA_MAX_BASELINE_M = 0.200
STEREO_CAMERA_VERTICAL_FOV_DEG = 70.0

STEREO_LEFT_CAMERA_NAME = "head_stereo_left"
STEREO_RIGHT_CAMERA_NAME = "head_stereo_right"
STEREO_LEFT_FRAME_ID = "sim_zed_left_camera_optical_frame"
STEREO_RIGHT_FRAME_ID = "sim_zed_right_camera_optical_frame"

ROS_STEREO_LEFT_COMPRESSED_TOPIC = "/left/image_rect/compressed"
ROS_STEREO_RIGHT_COMPRESSED_TOPIC = "/right/image_rect/compressed"
ROS_STEREO_LEFT_INFO_TOPIC = "/left/camera_info"
ROS_STEREO_RIGHT_INFO_TOPIC = "/right/camera_info"


def validate_stereo_baseline(value: float) -> float:
    """Validate the sim-only lateral distance between the stereo cameras."""
    baseline_m = float(value)
    if not math.isfinite(baseline_m):
        raise ValueError("Sim stereo eye distance must be finite")
    if not STEREO_CAMERA_MIN_BASELINE_M <= baseline_m <= STEREO_CAMERA_MAX_BASELINE_M:
        raise ValueError(
            "Sim stereo eye distance must be between "
            f"{STEREO_CAMERA_MIN_BASELINE_M * 1000:.0f} and "
            f"{STEREO_CAMERA_MAX_BASELINE_M * 1000:.0f} mm"
        )
    return baseline_m


def pinhole_intrinsics(
    *,
    width: int = STEREO_CAMERA_WIDTH,
    height: int = STEREO_CAMERA_HEIGHT,
    vertical_fov_deg: float = STEREO_CAMERA_VERTICAL_FOV_DEG,
) -> tuple[float, float, float, float]:
    """Return fx, fy, cx, cy for MuJoCo's vertical-FOV pinhole camera."""
    fy = 0.5 * float(height) / math.tan(0.5 * math.radians(float(vertical_fov_deg)))
    fx = fy
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    return fx, fy, cx, cy


def stereo_projection(*, baseline_m: float, right: bool) -> tuple[float, ...]:
    """Return a rectified pinhole P matrix with translation-only stereo extrinsics."""
    baseline_m = validate_stereo_baseline(baseline_m)
    fx, fy, cx, cy = pinhole_intrinsics()
    tx = -fx * baseline_m if right else 0.0
    return (fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0)
