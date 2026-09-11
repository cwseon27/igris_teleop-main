#shm_schema.py
import numpy as np
WAIST_INDICES = 3
LEG_INDICES = 12
ARM_INDICES = 14
NECK_INDICES = 2

# (field_name, shape, dtype)


ROBOT_OBS = [
    ("obs_seq", (), np.float64),
    ("obs_waist", (3,), np.float64),
    ("obs_leg", (12,), np.float64),
    ("obs_leg_dq", (12,), np.float64),
    ("obs_arm", (14,), np.float64),
    ("obs_neck", (2,), np.float64),
    ("obs_hand", (12,), np.float64),
    ("obs_imu_quat", (4,), np.float64),
    ("obs_imu_gyro", (3,), np.float64),
    ("obs_imu_rpy", (3,), np.float64),
]

ROBOT_TAU = [
    ("tau_est_waist", (3,), np.float64),
    ("tau_est_leg", (12,), np.float64),
    ("tau_est_arm", (14,), np.float64),
    ("tau_est_neck", (2,), np.float64),
    ]
ROBOT_ACTION = [
    ("act_waist", (3,), np.float64),
    ("act_leg", (12,), np.float64),
    ("act_arm", (14,), np.float64),
    ("act_neck", (2,), np.float64),
    ("act_hand", (12,), np.float64),
]

ROBOT_EE = [
    ("head_mat",        (4, 4),    np.float64),
    ("left_wrist_mat",  (4, 4),    np.float64),
    ("right_wrist_mat", (4, 4),    np.float64),
]

IK_TARGET = [
    ("target_valid",    (),        np.float64),
    ("target_seq",      (),        np.float64),
    ("torso_target_valid", (),     np.float64),
    ("torso_alpha",     (),        np.float64),
    ("chest_target_valid", (),     np.float64),
    ("chest_alpha",     (),        np.float64),
    ("head_mat",        (4, 4),    np.float64),
    ("torso_mat",       (4, 4),    np.float64),
    ("chest_mat",       (4, 4),    np.float64),
    ("left_wrist_mat",  (4, 4),    np.float64),
    ("right_wrist_mat", (4, 4),    np.float64),
]

TELEVISION = [
    ("head_mat",        (4, 4),    np.float64),
    ("torso_mat",       (4, 4),    np.float64),
    ("chest_mat",       (4, 4),    np.float64),
    ("left_wrist_mat",   (4, 4),    np.float64),
    ("right_wrist_mat",  (4, 4),    np.float64),
    ("left_hand",        (5, 3),    np.float64),
    ("right_hand",       (5, 3),    np.float64),
    ("left_hand_close",  (5,),      np.float64),
    ("right_hand_close", (5,),      np.float64),
    ("left_hand_close_valid", (),   np.float64),
    ("right_hand_close_valid", (),  np.float64),
    ("torso_alpha",      (),        np.float64),
    ("chest_alpha",      (),        np.float64),
    ("left_controller_confidence", (), np.float64),
    ("right_controller_confidence", (), np.float64),
    ("torso_source_valid", (),      np.float64),
    ("chest_source_valid", (),      np.float64),
    # Append final motor fields to preserve all existing body/legacy offsets.
    ("left_hand_motor", (6,), np.float64),
    ("right_hand_motor", (6,), np.float64),
    ("left_hand_motor_valid", (), np.float64),
    ("right_hand_motor_valid", (), np.float64),
]

CAMERA = [
    ("stereo_left", (480, 640, 3), np.uint8),
    ("stereo_right", (480, 640, 3), np.uint8),
    ("realsense_head", (480, 640, 3), np.uint8),
    ("realsense_wrist_left", (480, 640, 3), np.uint8),
    ("realsense_wrist_right", (480, 640, 3), np.uint8),
]

SIM_CONFIG = [
    ("stereo_baseline_m", (), np.float64),
    ("scene_command_seq", (), np.float64),
    ("scene_task_id", (), np.float64),
    ("scene_applied_seq", (), np.float64),
    ("scene_active_task_id", (), np.float64),
    ("scene_status_code", (), np.float64),
]

MODE_LAYOUT = [
    ("start",          (),    np.bool_),
    ("ready",          (),    np.bool_),
    ("run",             (),    np.bool_),
    ("home",          (),    np.bool_),
    ("teleop",        (),    np.bool_),
    ("walking",       (),    np.bool_),
    ("done",          (),    np.bool_),
    ("reset",          (),    np.bool_),
    ("replay",         (),    np.bool_),
    ("deploy",          (),    np.bool_),
    ("simulator",      (),    np.bool_),
]

WALKING_COMMAND = [
    ("profile_code", (), np.float64),
    ("vx", (), np.float64),
    ("vy", (), np.float64),
    ("dyaw", (), np.float64),
    ("policy_enabled", (), np.float64),
]

WALKING_DEBUG = [
    ("seq", (), np.float64),
    ("policy_tick", (), np.float64),
    ("startup_phase_code", (), np.float64),
    ("blend_alpha", (), np.float64),
    ("pose_err_max", (), np.float64),
    ("dq_max", (), np.float64),
    ("gyro_max", (), np.float64),
    ("loop_dt", (), np.float64),
    ("policy_cmd_vx", (), np.float64),
    ("policy_cmd_vy", (), np.float64),
    ("policy_cmd_dyaw", (), np.float64),
    ("policy_action_raw", (12,), np.float64),
    ("policy_action_applied", (12,), np.float64),
    ("previous_action", (12,), np.float64),
    ("policy_single_obs", (49,), np.float64),
    ("policy_gyro_base", (3,), np.float64),
    ("policy_rpy_base", (3,), np.float64),
    ("policy_target_leg_q", (12,), np.float64),
    ("active_kp_leg", (12,), np.float64),
    ("active_kd_leg", (12,), np.float64),
    ("active_kp_waist", (3,), np.float64),
    ("active_kd_waist", (3,), np.float64),
    ("active_kp_arm", (14,), np.float64),
    ("active_kd_arm", (14,), np.float64),
    ("active_kp_neck", (2,), np.float64),
    ("active_kd_neck", (2,), np.float64),
]

TELEOP_GUARD = [
    ("seq", (), np.float64),
    ("guard_active", (), np.float64),
    ("head_guard_ready", (), np.float64),
    ("head_guard_ok", (), np.float64),
    ("head_pos_err_m", (), np.float64),
    ("head_rot_err_deg", (), np.float64),
    ("head_yaw_correction_deg", (), np.float64),
    ("head_pos_thresh_m", (), np.float64),
    ("head_rot_thresh_deg", (), np.float64),
    ("head_reason_code", (), np.float64),
]

INFERENCE_RESULT_MAX_STEPS = 10000
INFERENCE_RESULT_MAX_ACTION_DIM = 64
INFERENCE_RESULT = [
    ("seq", (), np.float32),
    ("valid", (), np.float32),
    ("request_step", (), np.float32),
    ("timestamp", (), np.float32),
    ("n_action_steps", (), np.float32),
    ("action_dim", (), np.float32),
    ("action_chunk", (INFERENCE_RESULT_MAX_STEPS, INFERENCE_RESULT_MAX_ACTION_DIM), np.float32),
]

RECORD_MODE_LAYOUT = [
    ("record_start", (), np.bool_),
    ("record_done", (), np.bool_),
    ("record_reset", (), np.bool_),
]

RECORD_TASK_NAME_LEN = 512
RECORD_TASK = [
    ("task_valid", (), np.uint8),
    ("task_name", (RECORD_TASK_NAME_LEN,), np.uint8),
]

DATASET_FOLDER_NAME_LEN = 256
POSE_REQUEST = [
    ("request_seq", (), np.uint8),
    ("dataset_key", (DATASET_FOLDER_NAME_LEN,), np.uint8),
]

DATASET_INFO = [
    ("dataset_folder", (DATASET_FOLDER_NAME_LEN,), np.uint8),
]

DATASET_STATS = [
    ("current_episode_frames", (), np.int64),
    ("total_saved_frames", (), np.int64),
    ("num_episodes", (), np.int64),
]
