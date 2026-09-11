from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("proxsuite")

from igris_teleop.robot_control.kinematics.ik.prox_ik_pelvis_env import (  # noqa: E402
    CollisionParams,
    IGRIS_C_UpperIK,
    IKConfig,
    RateLimitParams,
    SolverParams,
)
from igris_teleop.robot_control.kinematics.ik.waist_task_transition import (  # noqa: E402
    WaistTaskTransition,
    WaistTransitionParams,
)
from igris_teleop.robot_control.kinematics.joints import LEG_INDICES  # noqa: E402
from igris_teleop.workers.worker_igris_ik import (  # noqa: E402
    CONTROLLER_CHEST_ROTATION_WEIGHT,
    CONTROLLER_CHEST_TRANSLATION_WEIGHT,
    HMD_HEAD_ROTATION_WEIGHT,
    HMD_HEAD_TRANSLATION_WEIGHT,
    IK_COMMAND_LEAD_LIMIT_ARM,
    IK_COMMAND_LEAD_LIMIT_NECK,
    IK_COMMAND_LEAD_LIMIT_WAIST,
    IK_RATE_LIMIT_REFERENCE_HZ,
    IGRISIKWorker,
    scale_ik_config_rate_limits,
)


class FakeTargetShm:
    def __init__(self) -> None:
        self.writes: list[dict] = []

    def write_data(self, **kwargs) -> None:
        self.writes.append(kwargs)


class FakeActionShm(FakeTargetShm):
    pass


def _solver(cfg: IKConfig | None = None) -> IGRIS_C_UpperIK:
    return IGRIS_C_UpperIK(cfg=cfg)


def _default_q(ik: IGRIS_C_UpperIK) -> np.ndarray:
    q = ik._default_init_data.copy() if ik._default_init_data is not None else ik.init_data.copy()
    return ik._clip_q_to_limits(q)


def _within_limits(ik: IGRIS_C_UpperIK, q: np.ndarray) -> bool:
    model = ik.reduced_robot.model
    return bool(
        np.all(q >= np.asarray(model.lowerPositionLimit) - 1e-9)
        and np.all(q <= np.asarray(model.upperPositionLimit) + 1e-9)
    )


def test_100hz_config_preserves_physical_discrete_rate_limits() -> None:
    cfg = IKConfig(
        rate_limit=RateLimitParams(
            dq_max=0.05,
            ddq_max=0.03,
            dddq_max=0.016,
        ),
        waist_transition=WaistTransitionParams(
            base_max_delta_per_step=0.04,
        ),
    )

    scaled = scale_ik_config_rate_limits(cfg, worker_hz=100.0)

    assert scaled.rate_limit.dq_max == pytest.approx(0.025)
    assert scaled.rate_limit.ddq_max == pytest.approx(0.0075)
    assert scaled.rate_limit.dddq_max == pytest.approx(0.002)
    assert (
        scaled.rate_limit.dq_max * 100.0
        == pytest.approx(cfg.rate_limit.dq_max * IK_RATE_LIMIT_REFERENCE_HZ)
    )
    assert (
        scaled.rate_limit.ddq_max * 100.0**2
        == pytest.approx(
            cfg.rate_limit.ddq_max * IK_RATE_LIMIT_REFERENCE_HZ**2
        )
    )
    assert (
        scaled.rate_limit.dddq_max * 100.0**3
        == pytest.approx(
            cfg.rate_limit.dddq_max * IK_RATE_LIMIT_REFERENCE_HZ**3
        )
    )
    assert scaled.weights == cfg.weights
    assert scaled.collision == cfg.collision
    assert scaled.solver == cfg.solver
    assert scaled.waist_transition.base_max_delta_per_step == pytest.approx(0.02)
    assert (
        scaled.waist_transition.activation_rise_rate
        == cfg.waist_transition.activation_rise_rate
    )
    assert (
        scaled.waist_transition.activation_fall_rate
        == cfg.waist_transition.activation_fall_rate
    )


@pytest.mark.parametrize("worker_hz", (0.0, -1.0, float("nan"), float("inf")))
def test_rate_limit_scaling_rejects_invalid_worker_hz(worker_hz: float) -> None:
    with pytest.raises(ValueError, match="worker_hz"):
        scale_ik_config_rate_limits(IKConfig(), worker_hz=worker_hz)


def test_current_pose_target_returns_finite_solution_within_limits() -> None:
    ik = _solver()
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)

    q, tau = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        current_lr_arm_motor_q=q0,
    )

    info = ik.get_last_solve_info()
    assert info["success"] is True
    assert q.shape == q0.shape
    assert tau.shape == q0.shape
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(tau))
    assert _within_limits(ik, q)
    assert info["max_returned_delta"] <= ik.cfg.rate_limit.dq_max + 1e-9
    assert "max_waist_delta" in info
    assert "max_arm_delta" in info
    assert "max_neck_delta" in info


def test_observation_waist_is_reordered_from_controller_to_ik() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    obs = {
        "obs_waist": np.array([0.3, 0.2, 0.1], dtype=np.float64),
        "obs_arm": np.arange(14, dtype=np.float64),
        "obs_neck": np.array([0.4, 0.5], dtype=np.float64),
    }

    q = worker._read_observation_q(obs)

    assert q is not None
    np.testing.assert_array_equal(q[:3], np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(q[3:17], obs["obs_arm"])
    np.testing.assert_array_equal(q[17:19], obs["obs_neck"])


def test_publish_targets_preserves_latched_leg_pose() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.act_shm = FakeActionShm()
    worker._fixed_leg_q = np.linspace(-0.3, 0.3, len(LEG_INDICES))
    worker._profile = type("Profile", (), {"publish_arm": True})()

    worker._publish_targets(worker._default_home_q())

    payload = worker.act_shm.writes[-1]
    np.testing.assert_array_equal(payload["act_leg"], worker._fixed_leg_q)


def test_publish_targets_does_not_zero_legs_before_observation() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.act_shm = FakeActionShm()
    worker._fixed_leg_q = None
    worker._profile = type("Profile", (), {"publish_arm": True})()

    worker._publish_targets(worker._default_home_q())

    assert "act_leg" not in worker.act_shm.writes[-1]


def test_ik_seed_continues_previous_command_inside_tracking_envelope() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    current = worker._default_home_q()
    previous_command = current + 0.02
    worker._last_sol_q = previous_command.copy()

    seed = worker._resolve_ik_seed_q(current)

    np.testing.assert_allclose(seed, previous_command)


def test_ik_seed_clips_command_lead_by_joint_group() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    current = worker._default_home_q()
    worker._last_sol_q = current + 1.0

    seed = worker._resolve_ik_seed_q(current)
    delta = seed - current

    np.testing.assert_allclose(delta[:3], IK_COMMAND_LEAD_LIMIT_WAIST)
    np.testing.assert_allclose(delta[3:17], IK_COMMAND_LEAD_LIMIT_ARM)
    np.testing.assert_allclose(delta[17:19], IK_COMMAND_LEAD_LIMIT_NECK)


def test_ik_command_accumulates_when_observation_is_temporarily_stalled() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    current = worker._default_home_q()
    worker._last_sol_q = current.copy()

    for _ in range(20):
        seed = worker._resolve_ik_seed_q(current)
        next_command = seed.copy()
        next_command[-2:] -= 0.05
        worker._last_sol_q = worker._limit_command_lead(next_command, current)

    neck_lead = current[-2:] - worker._last_sol_q[-2:]
    np.testing.assert_allclose(neck_lead, IK_COMMAND_LEAD_LIMIT_NECK)


def test_torso_and_chest_target_frames_are_distinct() -> None:
    ik = _solver()
    torso_frame = ik.reduced_robot.model.frames[ik.Torso_id]
    chest_frame = ik.reduced_robot.model.frames[ik.Chest_id]
    np.testing.assert_allclose(torso_frame.placement.translation, [0.05, 0.0, 0.3])
    np.testing.assert_allclose(chest_frame.placement.translation, [0.20, 0.0, 0.3])
    np.testing.assert_allclose(torso_frame.placement.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(chest_frame.placement.rotation, np.eye(3), atol=1e-12)
    assert ik.Torso_id != ik.Chest_id

    q0 = _default_q(ik)
    _, _, head = ik.get_ee_poses(q0)
    torso = ik.get_torso_pose(q0)
    chest = ik.get_chest_pose(q0)

    assert chest.translation[0] > torso.translation[0] + 0.1
    assert chest.translation[0] > head.translation[0] + 0.1
    assert abs((head.translation[2] - chest.translation[2]) - 0.2) < 0.02


def test_wrist_translation_target_reduces_error_over_repeated_steps() -> None:
    ik = _solver()
    q = _default_q(ik)
    left, right, head = ik.get_ee_poses(q)
    left_target = left.homogeneous.copy()
    right_target = right.homogeneous.copy()
    head_target = head.homogeneous.copy()
    left_target[:3, 3] += np.array([0.0, 0.03, 0.0])
    right_target[:3, 3] += np.array([0.0, -0.03, 0.0])

    before = max(
        np.linalg.norm(left_target[:3, 3] - left.translation),
        np.linalg.norm(right_target[:3, 3] - right.translation),
    )
    for _ in range(10):
        q, _ = ik.solve_ik(
            left_target,
            right_target,
            head_target,
            current_lr_arm_motor_q=q,
        )

    left_after, right_after, _ = ik.get_ee_poses(q)
    after = max(
        np.linalg.norm(left_target[:3, 3] - left_after.translation),
        np.linalg.norm(right_target[:3, 3] - right_after.translation),
    )
    assert after < before * 0.1
    assert _within_limits(ik, q)


def test_hand_targets_off_with_arm_lock_keeps_arm_joints() -> None:
    ik = _solver()
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)
    left_target = left.homogeneous.copy()
    right_target = right.homogeneous.copy()
    left_target[:3, 3] += np.array([0.2, 0.2, 0.0])
    right_target[:3, 3] += np.array([0.2, -0.2, 0.0])

    q, _ = ik.solve_ik(
        left_target,
        right_target,
        head.homogeneous,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        arm_lock_q=q0,
        arm_lock_weight=1000.0,
    )

    assert np.allclose(q[ik._arm_slice], q0[ik._arm_slice], atol=1e-6)


def test_seed_near_joint_limits_stays_within_limits_and_rate_limit() -> None:
    ik = _solver()
    model = ik.reduced_robot.model
    q0 = np.asarray(model.upperPositionLimit, dtype=np.float64) - 1e-3
    q0 = ik._clip_q_to_limits(q0)
    left, right, head = ik.get_ee_poses(q0)

    q, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        current_lr_arm_motor_q=q0,
    )

    assert _within_limits(ik, q)
    assert np.max(np.abs(q - q0)) <= ik.cfg.rate_limit.dq_max + 1e-9


def test_collision_constraints_are_added_to_qp_and_reported() -> None:
    cfg = IKConfig(collision=CollisionParams(enabled=True, activation_distance=1.0))
    ik = _solver(cfg)
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)

    q, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        current_lr_arm_motor_q=q0,
    )

    info = ik.get_last_solve_info()
    assert info["success"] is True
    assert info["active_collision_constraints"] == len(ik.collision_pairs)
    assert info["min_collision_margin_before"] is not None
    assert info["min_collision_margin_after"] is not None
    assert _within_limits(ik, q)


def test_collision_slack_keeps_qp_solvable_for_unreachable_margin() -> None:
    cfg = IKConfig(
        collision=CollisionParams(
            enabled=True,
            margin=10.0,
            activation_distance=20.0,
            slack_max=20.0,
        )
    )
    ik = _solver(cfg)
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)

    q, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        current_lr_arm_motor_q=q0,
    )

    info = ik.get_last_solve_info()
    assert info["success"] is True
    assert info["max_collision_slack"] > 0.0
    assert _within_limits(ik, q)


def test_profile_config_loads_from_yaml_and_env(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "ik.yaml"
    config_path.write_text(
        """
default:
  rate_limit:
    dq_max: 0.11
profiles:
  unity:
    weights:
      w_trans: 77.0
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("IGRIS_IK_UNITY_DQ_MAX", "0.07")

    cfg = IKConfig.from_sources(profile="unity", config_path=config_path)

    assert cfg.weights.w_trans == pytest.approx(77.0)
    assert cfg.rate_limit.dq_max == pytest.approx(0.07)


def test_replay_style_target_sequence_stays_finite_and_reports_telemetry() -> None:
    ik = _solver()
    q = _default_q(ik)
    left0, right0, head0 = ik.get_ee_poses(q)
    left_target = left0.homogeneous.copy()
    right_target = right0.homogeneous.copy()
    head_target = head0.homogeneous.copy()

    for step in range(20):
        phase = 2.0 * np.pi * (step / 20.0)
        left_target[:3, 3] = left0.translation + np.array([0.0, 0.02 * np.sin(phase), 0.01 * np.cos(phase)])
        right_target[:3, 3] = right0.translation + np.array([0.0, -0.02 * np.sin(phase), 0.01 * np.cos(phase)])
        q, _ = ik.solve_ik(
            left_target,
            right_target,
            head_target,
            current_lr_arm_motor_q=q,
        )
        info = ik.get_last_solve_info()
        assert info["success"] is True
        assert np.all(np.isfinite(q))
        assert _within_limits(ik, q)
        assert info["max_returned_delta"] <= ik.cfg.rate_limit.dq_max + 1e-9
        assert "max_trans_error_after" in info


def test_unity_target_builder_applies_bridge_relative_pose_to_robot_anchor() -> None:
    robot_init = np.eye(4, dtype=np.float64)
    robot_init[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    robot_init[:3, 3] = np.array([0.4, -0.2, 1.1])

    vr_now = np.eye(4, dtype=np.float64)
    vr_now[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    vr_now[:3, 3] = np.array([0.05, -0.03, 0.02])

    target = IGRISIKWorker._build_target_unity(vr_now, robot_init)

    assert np.allclose(target[:3, 3], robot_init[:3, 3] + vr_now[:3, 3])
    assert np.allclose(target[:3, :3], vr_now[:3, :3] @ robot_init[:3, :3])


def test_robot_home_anchor_can_follow_observation_until_run() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.ik_solver = _solver()
    worker._home_set = False
    worker._last_sol_q = None
    worker._robot_left_init = None
    worker._robot_right_init = None
    worker._robot_head_init = None
    worker._robot_chest_init = None

    q0 = _default_q(worker.ik_solver)
    q1 = q0.copy()
    q1[-1] += 0.1

    assert worker._maybe_set_home(q0)
    first_head = worker._robot_head_init.copy()
    assert worker._maybe_set_home(q1)
    np.testing.assert_array_equal(worker._robot_head_init, first_head)

    assert worker._maybe_set_home(q1, force=True)
    expected_head = worker.ik_solver.get_ee_poses(q1)[2].homogeneous
    np.testing.assert_allclose(worker._robot_head_init, expected_head)


def test_chest_target_builder_applies_home_relative_motion() -> None:
    robot_init = np.eye(4, dtype=np.float64)
    robot_init[:3, 3] = np.array([0.15, 0.0, 0.37])
    relative = np.eye(4, dtype=np.float64)
    relative[:3, 3] = np.array([0.03, -0.02, 0.01])

    target = IGRISIKWorker._build_chest_target_unity(relative, robot_init)

    np.testing.assert_allclose(target[:3, 3], [0.18, -0.02, 0.38])


def test_head_only_yaw_uses_waist_and_neck_without_hidden_torso_target() -> None:
    ik = _solver()
    q = _default_q(ik)
    left, right, head = ik.get_ee_poses(q)
    head_target = head.homogeneous.copy()
    yaw = np.deg2rad(20.0)
    head_target[:3, :3] = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ) @ head_target[:3, :3]

    for _ in range(10):
        q, _ = ik.solve_ik(
            left.homogeneous,
            right.homogeneous,
            head_target,
            current_lr_arm_motor_q=q,
            use_hand_targets=False,
            arm_lock_q=q,
            arm_lock_weight=1000.0,
        )

    solved_head = ik.get_ee_poses(q)[2]
    rotation_delta = solved_head.rotation.T @ head_target[:3, :3]
    rotation_error = np.arccos(
        np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
    )

    assert rotation_error < np.deg2rad(3.0)
    assert abs(float(q[2])) > np.deg2rad(5.0)
    assert abs(float(q[-2])) > np.deg2rad(5.0)


@pytest.mark.parametrize("chest_alpha", [0.0, 1.0])
def test_head_pitch_can_return_home_after_large_orientation_target(
    chest_alpha: float,
) -> None:
    ik = _solver()
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)
    chest = ik.get_chest_pose(q0)
    head_target = head.homogeneous.copy()
    pitch = np.deg2rad(55.0)
    pitch_rotation = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    head_target[:3, :3] = pitch_rotation @ head_target[:3, :3]
    head_trans_w, head_rot_w, chest_trans_w, chest_rot_w = (
        IGRISIKWorker._upper_body_task_weights(chest_alpha)
    )

    def solve_steps(q: np.ndarray, target: np.ndarray, count: int) -> np.ndarray:
        for _ in range(count):
            q, _ = ik.solve_ik(
                left.homogeneous,
                right.homogeneous,
                target,
                chest_pose=chest.homogeneous,
                use_chest_target=chest_alpha > 0.0,
                chest_translation_weight=chest_trans_w,
                chest_rotation_weight=chest_rot_w,
                head_translation_weight=head_trans_w,
                head_rotation_weight=head_rot_w,
                current_lr_arm_motor_q=q,
                use_hand_targets=False,
                arm_lock_q=q0,
                arm_lock_weight=1000.0,
            )
        return q

    q_bowed = solve_steps(q0.copy(), head_target, 50)
    bowed_head = ik.get_ee_poses(q_bowed)[2]
    bowed_rotation_delta = head.rotation.T @ bowed_head.rotation
    bowed_rotation = np.arccos(
        np.clip((np.trace(bowed_rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
    )
    assert bowed_rotation > np.deg2rad(20.0)
    bowed_joint_delta = q_bowed[[0, 1, 2, -2, -1]] - q0[[0, 1, 2, -2, -1]]
    assert np.max(np.abs(bowed_joint_delta)) > np.deg2rad(10.0)

    q_returned = solve_steps(q_bowed, head.homogeneous, 50)
    returned_head = ik.get_ee_poses(q_returned)[2]
    rotation_delta = returned_head.rotation.T @ head.rotation
    rotation_error = np.arccos(
        np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
    )

    assert np.linalg.norm(returned_head.translation - head.translation) < 0.002
    assert rotation_error < np.deg2rad(1.0)
    returned_joint_delta = q_returned[[0, 1, 2, -2, -1]] - q0[[0, 1, 2, -2, -1]]
    assert np.max(np.abs(returned_joint_delta)) < np.deg2rad(1.0)


def test_alpha_zero_uses_head_only_task_set_with_strong_hmd_rotation() -> None:
    class CapturingSolver:
        def __init__(self) -> None:
            self.kwargs = None

        def solve_ik(self, *args, **kwargs):
            self.kwargs = kwargs
            return np.zeros(19, dtype=np.float64), np.zeros(19, dtype=np.float64)

    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.ik_solver = CapturingSolver()
    worker._waist_transition = WaistTaskTransition(WaistTransitionParams())
    worker._profile = type(
        "Profile",
        (),
        {"use_hand_targets": False, "freeze_arms": True},
    )()
    pose = np.eye(4, dtype=np.float64)
    q = np.zeros(19, dtype=np.float64)

    worker._solve_active_targets(
        pose,
        pose,
        pose,
        chest_target=None,
        waist_activation=0.0,
        previous_published_q=q,
        waist_delta_limit=0.0,
        waist_hold_weight=1.0,
        ik_seed_q=q,
        current_q=q,
    )

    kwargs = worker.ik_solver.kwargs
    assert kwargs["use_chest_target"] is False
    assert kwargs["head_translation_weight"] == HMD_HEAD_TRANSLATION_WEIGHT
    assert kwargs["head_rotation_weight"] == HMD_HEAD_ROTATION_WEIGHT
    assert kwargs["chest_translation_weight"] == 0.0
    assert kwargs["chest_rotation_weight"] == 0.0
    assert kwargs["use_hand_targets"] is False
    assert kwargs["waist_delta_limit"] == 0.0
    assert kwargs["waist_lock_weight"] == 1.0
    assert kwargs["commit_state"] is False
    assert "torso_pose" not in kwargs
    assert "use_torso_target" not in kwargs


def test_hybrid_task_weights_preserve_head_position_and_scale_chest() -> None:
    class CapturingSolver:
        def __init__(self) -> None:
            self.kwargs = None

        def solve_ik(self, *args, **kwargs):
            self.kwargs = kwargs
            return np.zeros(19, dtype=np.float64), np.zeros(19, dtype=np.float64)

    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.ik_solver = CapturingSolver()
    worker._waist_transition = WaistTaskTransition(WaistTransitionParams())
    worker._profile = type(
        "Profile",
        (),
        {"use_hand_targets": False, "freeze_arms": True},
    )()
    pose = np.eye(4, dtype=np.float64)
    q = np.zeros(19, dtype=np.float64)

    worker._solve_active_targets(
        pose,
        pose,
        pose,
        chest_target=pose,
        waist_activation=0.75,
        previous_published_q=q,
        waist_delta_limit=0.0375,
        waist_hold_weight=0.25,
        ik_seed_q=q,
        current_q=q,
    )

    kwargs = worker.ik_solver.kwargs
    assert kwargs["head_translation_weight"] == HMD_HEAD_TRANSLATION_WEIGHT
    assert kwargs["head_rotation_weight"] == HMD_HEAD_ROTATION_WEIGHT
    assert kwargs["chest_translation_weight"] == pytest.approx(
        CONTROLLER_CHEST_TRANSLATION_WEIGHT * 0.75
    )
    assert kwargs["chest_rotation_weight"] == pytest.approx(
        CONTROLLER_CHEST_ROTATION_WEIGHT * 0.75
    )
    assert kwargs["waist_delta_limit"] == pytest.approx(0.0375)
    assert kwargs["waist_lock_weight"] == pytest.approx(0.25)
    assert kwargs["commit_state"] is False


def test_missing_chest_target_keeps_full_head_position_weight() -> None:
    class CapturingSolver:
        def __init__(self) -> None:
            self.kwargs = None

        def solve_ik(self, *args, **kwargs):
            self.kwargs = kwargs
            return np.zeros(19, dtype=np.float64), np.zeros(19, dtype=np.float64)

    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.ik_solver = CapturingSolver()
    worker._waist_transition = WaistTaskTransition(WaistTransitionParams())
    worker._profile = type(
        "Profile",
        (),
        {"use_hand_targets": False, "freeze_arms": True},
    )()
    pose = np.eye(4, dtype=np.float64)
    q = np.zeros(19, dtype=np.float64)

    worker._solve_active_targets(
        pose,
        pose,
        pose,
        chest_target=None,
        waist_activation=0.75,
        previous_published_q=q,
        waist_delta_limit=0.0375,
        waist_hold_weight=0.25,
        ik_seed_q=q,
        current_q=q,
    )

    kwargs = worker.ik_solver.kwargs
    assert kwargs["head_translation_weight"] == HMD_HEAD_TRANSLATION_WEIGHT
    assert kwargs["chest_translation_weight"] == 0.0
    assert kwargs["chest_rotation_weight"] == 0.0


@pytest.mark.parametrize(
    ("alpha", "expected_alpha"),
    [
        (-1.0, 0.0),
        (0.0, 0.0),
        (0.4, 0.4),
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_hybrid_task_weight_alpha_is_clipped(
    alpha: float,
    expected_alpha: float,
) -> None:
    head_trans, head_rot, chest_trans, chest_rot = (
        IGRISIKWorker._upper_body_task_weights(alpha)
    )

    assert head_trans == HMD_HEAD_TRANSLATION_WEIGHT
    assert head_rot == HMD_HEAD_ROTATION_WEIGHT
    assert chest_trans == pytest.approx(
        CONTROLLER_CHEST_TRANSLATION_WEIGHT * expected_alpha
    )
    assert chest_rot == pytest.approx(
        CONTROLLER_CHEST_ROTATION_WEIGHT * expected_alpha
    )


def test_publish_ik_targets_includes_optional_torso_target() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.ik_target_shm = FakeTargetShm()

    left = np.eye(4, dtype=np.float64)
    right = np.eye(4, dtype=np.float64)
    head = np.eye(4, dtype=np.float64)
    torso = np.eye(4, dtype=np.float64)
    torso[:3, 3] = np.array([0.2, 0.0, 0.3])

    worker._publish_ik_targets(left, right, head, torso, 0.75)

    payload = worker.ik_target_shm.writes[-1]
    assert payload["target_valid"] == 1.0
    assert payload["torso_target_valid"] == 1.0
    assert payload["torso_alpha"] == pytest.approx(0.75)
    np.testing.assert_array_equal(payload["torso_mat"], torso)
    assert payload["chest_target_valid"] == 0.0


def test_publish_ik_targets_includes_separate_chest_target() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.ik_target_shm = FakeTargetShm()

    pose = np.eye(4, dtype=np.float64)
    chest = np.eye(4, dtype=np.float64)
    chest[:3, 3] = np.array([0.2, 0.0, 0.3])

    worker._publish_ik_targets(
        pose,
        pose,
        pose,
        chest_target=chest,
        chest_alpha=0.65,
    )

    payload = worker.ik_target_shm.writes[-1]
    assert payload["torso_target_valid"] == 0.0
    assert payload["chest_target_valid"] == 1.0
    assert payload["chest_alpha"] == pytest.approx(0.65)
    np.testing.assert_array_equal(payload["chest_mat"], chest)


def test_publish_ik_targets_hides_stale_chest_pose_at_zero_alpha() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.ik_target_shm = FakeTargetShm()

    pose = np.eye(4, dtype=np.float64)
    chest = np.eye(4, dtype=np.float64)
    chest[:3, 3] = np.array([0.2, 0.0, 0.3])

    worker._publish_ik_targets(
        pose,
        pose,
        pose,
        chest_target=chest,
        chest_alpha=0.0,
    )

    payload = worker.ik_target_shm.writes[-1]
    assert payload["chest_target_valid"] == 0.0
    assert payload["chest_alpha"] == 0.0


def test_publish_ik_targets_can_show_valid_chest_pose_at_zero_confidence() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.ik_target_shm = FakeTargetShm()

    pose = np.eye(4, dtype=np.float64)
    chest = np.eye(4, dtype=np.float64)
    chest[:3, 3] = np.array([0.2, 0.0, 0.3])

    worker._publish_ik_targets(
        pose,
        pose,
        pose,
        chest_target=chest,
        chest_alpha=0.0,
        chest_target_valid=True,
    )

    payload = worker.ik_target_shm.writes[-1]
    assert payload["chest_target_valid"] == 1.0
    assert payload["chest_alpha"] == 0.0
    np.testing.assert_array_equal(payload["chest_mat"], chest)


def test_command_lead_limit_can_preserve_qp_authoritative_waist() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    current = worker._default_home_q()
    command = current + 1.0

    limited = worker._limit_command_lead(
        command,
        current,
        preserve_waist=True,
    )

    np.testing.assert_allclose(limited[:3], command[:3])
    np.testing.assert_allclose(
        limited[3:17] - current[3:17],
        IK_COMMAND_LEAD_LIMIT_ARM,
    )
    np.testing.assert_allclose(
        limited[17:19] - current[17:19],
        IK_COMMAND_LEAD_LIMIT_NECK,
    )


def test_run_blend_preserves_qp_authoritative_waist(monkeypatch) -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker._run_blend_active = True
    worker._run_blend_start_t = 10.0
    worker._run_blend_duration = 2.0
    worker._run_blend_q0 = np.zeros(19, dtype=np.float64)
    solution = np.ones(19, dtype=np.float64)

    monkeypatch.setattr(
        "igris_teleop.workers.worker_igris_ik.time.monotonic",
        lambda: 11.0,
    )
    blended = worker._apply_run_blend(
        solution,
        preserve_waist=True,
    )

    np.testing.assert_array_equal(blended[:3], solution[:3])
    np.testing.assert_allclose(blended[3:], 0.5)


def test_record_published_command_uses_command_not_encoder_observation() -> None:
    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    worker.ik_solver = _solver()
    worker._waist_transition = WaistTaskTransition(
        worker.ik_solver.cfg.waist_transition
    )
    worker._previous_published_q = None
    worker._last_sol_q = worker._default_home_q()

    first_command = _default_q(worker.ik_solver)
    worker._record_published_command(
        first_command,
        reset_motion_state=True,
    )

    encoder_q = first_command.copy()
    encoder_q[:3] += np.array([0.2, -0.2, 0.2])
    second_command = first_command.copy()
    second_command[:3] += np.array([0.01, -0.01, 0.01])
    worker._record_published_command(second_command)

    np.testing.assert_array_equal(
        worker._previous_published_q,
        second_command,
    )
    np.testing.assert_array_equal(
        worker._waist_transition.previous_published_waist,
        second_command[:3],
    )
    assert not np.array_equal(
        worker._previous_published_q[:3],
        encoder_q[:3],
    )
    np.testing.assert_allclose(
        worker.ik_solver._last_dq,
        second_command - first_command,
    )


def test_torso_task_weight_is_applied_inside_single_qp() -> None:
    ik = _solver()
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)
    torso = ik.get_torso_pose(q0)
    torso_target = torso.homogeneous.copy()
    yaw = 0.15
    rotation_delta = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    torso_target[:3, :3] = torso_target[:3, :3] @ rotation_delta

    disabled, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        torso_pose=torso_target,
        use_torso_target=True,
        torso_target_weight=0.0,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        use_head_target=False,
    )
    enabled, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        torso_pose=torso_target,
        use_torso_target=True,
        torso_target_weight=1.0,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        use_head_target=False,
    )

    assert np.max(np.abs(disabled - q0)) < 1e-6
    assert np.max(np.abs(enabled[:3] - q0[:3])) > 1e-4


def test_chest_task_weight_is_applied_inside_single_qp() -> None:
    ik = _solver()
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)
    chest_target = ik.get_chest_pose(q0).homogeneous.copy()
    yaw = 0.15
    rotation_delta = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    chest_target[:3, :3] = chest_target[:3, :3] @ rotation_delta

    disabled, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        chest_pose=chest_target,
        use_chest_target=True,
        chest_target_weight=0.0,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        use_head_target=False,
    )
    enabled, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        chest_pose=chest_target,
        use_chest_target=True,
        chest_target_weight=1.0,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        use_head_target=False,
    )

    assert np.max(np.abs(disabled - q0)) < 1e-6
    assert np.max(np.abs(enabled[:3] - q0[:3])) > 1e-4


def test_high_alpha_hybrid_tracks_reachable_chest_pose_and_preserves_head() -> None:
    ik = _solver()
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)
    left_target = left.homogeneous.copy()
    right_target = right.homogeneous.copy()
    head_target = head.homogeneous.copy()
    chest_initial = ik.get_chest_pose(q0).homogeneous.copy()

    reachable_q = q0.copy()
    reachable_q[2] += np.deg2rad(15.0)
    chest_target = ik.get_chest_pose(reachable_q).homogeneous.copy()
    ik.reset_motion_state(q0)

    def rotation_error(current: np.ndarray, target: np.ndarray) -> float:
        delta = current[:3, :3].T @ target[:3, :3]
        return float(
            np.arccos(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
        )

    initial_position_error = float(
        np.linalg.norm(chest_initial[:3, 3] - chest_target[:3, 3])
    )
    initial_rotation_error = rotation_error(chest_initial, chest_target)
    head_trans_w, head_rot_w, chest_trans_w, chest_rot_w = (
        IGRISIKWorker._upper_body_task_weights(1.0)
    )

    q = q0.copy()
    for _ in range(30):
        q, _ = ik.solve_ik(
            left_target,
            right_target,
            head_target,
            chest_pose=chest_target,
            use_chest_target=True,
            chest_translation_weight=chest_trans_w,
            chest_rotation_weight=chest_rot_w,
            head_translation_weight=head_trans_w,
            head_rotation_weight=head_rot_w,
            current_lr_arm_motor_q=q,
            use_hand_targets=False,
            arm_lock_q=q0,
            arm_lock_weight=1000.0,
        )

    chest_after = ik.get_chest_pose(q).homogeneous.copy()
    head_after = ik.get_ee_poses(q)[2].homogeneous.copy()
    info = ik.get_last_solve_info()

    assert np.linalg.norm(chest_after[:3, 3] - chest_target[:3, 3]) < (
        initial_position_error * 0.1
    )
    assert rotation_error(chest_after, chest_target) < initial_rotation_error * 0.1
    assert np.linalg.norm(head_after[:3, 3] - head_target[:3, 3]) < 0.005
    assert rotation_error(head_after, head_target) < np.deg2rad(2.0)
    assert abs(float(q[2] - q0[2])) > np.deg2rad(10.0)
    assert "head_trans_error_before" in info
    assert "head_rot_error_after" in info
    assert "chest_trans_error_before" in info
    assert "chest_rot_error_after" in info


def test_waist_transition_config_loads_from_yaml_and_profile_env(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "ik_transition.yaml"
    config_path.write_text(
        """
default:
  waist_transition:
    low_threshold: 0.15
    scale_head_translation_with_activation: true
profiles:
  unity_hybrid:
    waist_transition:
      activation_rise_rate: 3.0
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "IGRIS_IK_UNITY_HYBRID_WAIST_TRANSITION_HIGH_THRESHOLD",
        "0.9",
    )

    cfg = IKConfig.from_sources(
        profile="unity_hybrid",
        config_path=config_path,
    )

    assert cfg.waist_transition.low_threshold == pytest.approx(0.15)
    assert cfg.waist_transition.high_threshold == pytest.approx(0.9)
    assert cfg.waist_transition.activation_rise_rate == pytest.approx(3.0)
    assert cfg.waist_transition.scale_head_translation_with_activation is True


def test_zero_waist_window_holds_previous_published_command_and_moves_neck() -> None:
    ik = _solver()
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)
    head_target = head.homogeneous.copy()
    yaw = np.deg2rad(20.0)
    head_target[:3, :3] = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ) @ head_target[:3, :3]

    q, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head_target,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        arm_lock_q=q0,
        arm_lock_weight=1000.0,
        head_translation_weight=0.0,
        head_rotation_weight=6.0,
        waist_motion_reference_q=q0,
        waist_delta_limit=0.0,
        commit_state=False,
    )

    np.testing.assert_allclose(q[ik._waist_slice], q0[ik._waist_slice], atol=1e-9)
    assert np.max(np.abs(q[ik._neck_slice] - q0[ik._neck_slice])) > 1e-5
    info = ik.get_last_solve_info()
    assert info["waist_bound_active"] is True
    assert info["waist_delta_limit"] == pytest.approx([0.0, 0.0, 0.0])


def test_zero_waist_window_is_anchored_to_published_command_not_seed() -> None:
    ik = _solver()
    published_q = _default_q(ik)
    seed_q = published_q.copy()
    seed_q[ik._waist_slice] += np.array([0.01, -0.01, 0.01])
    left, right, head = ik.get_ee_poses(seed_q)

    q, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        current_lr_arm_motor_q=seed_q,
        use_hand_targets=False,
        use_head_target=False,
        waist_motion_reference_q=published_q,
        waist_delta_limit=0.0,
        commit_state=False,
    )

    np.testing.assert_allclose(
        q[ik._waist_slice],
        published_q[ik._waist_slice],
        atol=1e-9,
    )


def test_waist_total_delta_budget_is_preserved_across_sqp_iterations() -> None:
    cfg = IKConfig(
        solver=SolverParams(linearization_iters=4),
        waist_transition=WaistTransitionParams(),
    )
    ik = _solver(cfg)
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)
    reachable_q = q0.copy()
    reachable_q[2] += np.deg2rad(30.0)
    chest_target = ik.get_chest_pose(reachable_q).homogeneous
    limit = 0.01

    q, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        chest_pose=chest_target,
        use_chest_target=True,
        chest_translation_weight=2.0,
        chest_rotation_weight=2.0,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        use_head_target=False,
        waist_motion_reference_q=q0,
        waist_delta_limit=limit,
        commit_state=False,
    )

    waist_delta = np.abs(q[ik._waist_slice] - q0[ik._waist_slice])
    assert np.max(waist_delta) <= limit + 1e-9
    assert np.max(waist_delta) > 1e-4


def test_zero_waist_hold_overrides_stale_acceleration_history() -> None:
    ik = _solver()
    q0 = _default_q(ik)
    ik._last_dq[ik._waist_slice] = ik.cfg.rate_limit.dq_max
    left, right, head = ik.get_ee_poses(q0)

    q, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        use_head_target=False,
        waist_motion_reference_q=q0,
        waist_delta_limit=0.0,
        commit_state=False,
    )

    np.testing.assert_allclose(q[ik._waist_slice], q0[ik._waist_slice], atol=1e-9)
    assert ik.get_last_solve_info()["waist_acceleration_override"] is True


def test_solve_without_commit_does_not_advance_motion_state() -> None:
    ik = _solver()
    q0 = _default_q(ik)
    left, right, head = ik.get_ee_poses(q0)
    init_before = ik.init_data.copy()
    last_dq_before = ik._last_dq.copy()
    prev_dq_before = ik._prev_dq.copy()

    q, _ = ik.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        current_lr_arm_motor_q=q0,
        commit_state=False,
    )

    np.testing.assert_array_equal(ik.init_data, init_before)
    np.testing.assert_array_equal(ik._last_dq, last_dq_before)
    np.testing.assert_array_equal(ik._prev_dq, prev_dq_before)

    published_q = q.copy()
    published_q[-1] += 0.01
    ik.commit_published_command(published_q, q0)

    np.testing.assert_array_equal(ik.init_data, published_q)
    np.testing.assert_allclose(ik._last_dq, published_q - q0)
    np.testing.assert_array_equal(ik._prev_dq, last_dq_before)


def test_full_waist_window_matches_existing_full_confidence_solution() -> None:
    baseline = _solver()
    transitioned = _solver()
    q0 = _default_q(baseline)
    left, right, head = baseline.get_ee_poses(q0)
    reachable_q = q0.copy()
    reachable_q[2] += np.deg2rad(15.0)
    chest_target = baseline.get_chest_pose(reachable_q).homogeneous

    baseline_q, _ = baseline.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        chest_pose=chest_target,
        use_chest_target=True,
        chest_translation_weight=2.0,
        chest_rotation_weight=2.0,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        commit_state=False,
    )
    transitioned_q, _ = transitioned.solve_ik(
        left.homogeneous,
        right.homogeneous,
        head.homogeneous,
        chest_pose=chest_target,
        use_chest_target=True,
        chest_translation_weight=2.0,
        chest_rotation_weight=2.0,
        current_lr_arm_motor_q=q0,
        use_hand_targets=False,
        waist_motion_reference_q=q0,
        waist_delta_limit=transitioned.cfg.rate_limit.dq_max,
        commit_state=False,
    )

    np.testing.assert_allclose(transitioned_q, baseline_q, atol=1e-8, rtol=0.0)
