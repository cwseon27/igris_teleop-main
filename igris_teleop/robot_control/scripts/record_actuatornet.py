from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from igris_teleop.core.project_paths import ACTUATORNET_DATASETS_ROOT
from igris_teleop.robot_control.kinematics.joints import (
    ARM_INDICES,
    LEG_INDICES,
    NECK_INDICES,
    WAIST_INDICES,
)
from igris_teleop.robot_control.scripts.common_rollout import (
    make_chirp_profile,
    make_hold_profile,
    make_sine_profile,
    run_rollout,
    save_episode_npz,
)


def resolve_joint_group(name: str):
    groups = {
        "waist": list(WAIST_INDICES),
        "leg": list(LEG_INDICES),
        "arm": list(ARM_INDICES),
        "neck": list(NECK_INDICES),
        "all": list(sorted(set(WAIST_INDICES) | set(LEG_INDICES) | set(ARM_INDICES) | set(NECK_INDICES))),
    }
    if name not in groups:
        raise KeyError(f"Unknown joint group: {name}")
    return groups[name]


def build_profile(args, q_ref: np.ndarray):
    joint_indices = resolve_joint_group(args.joint_group)
    if args.profile == "hold":
        return make_hold_profile(q_ref)
    if args.profile == "sine":
        return make_sine_profile(q_ref, joint_indices, amplitude=args.amplitude, frequency=args.frequency)
    if args.profile == "chirp":
        return make_chirp_profile(q_ref, joint_indices, amplitude=args.amplitude, f0=args.f0, f1=args.f1, duration=args.duration)
    raise ValueError(f"Unsupported profile: {args.profile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--control-hz", type=float, default=300.0)
    parser.add_argument("--record-hz", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--joint-group", type=str, default="waist")
    parser.add_argument("--profile", type=str, choices=["hold", "sine", "chirp"], default="sine")
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--f0", type=float, default=0.2)
    parser.add_argument("--f1", type=float, default=2.0)
    parser.add_argument("--move-to-default", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from igris_teleop.robot_control.controller.igris_controller import IgrisController

    ctrl = IgrisController(domain_id=args.domain_id, control_hz=args.control_hz)
    try:
        if not ctrl.wait_for_state(timeout=5.0):
            raise RuntimeError("LowState timeout")
        if args.move_to_default:
            ctrl.move_to_pose("default_pos", duration=2.0)
            time.sleep(0.5)
        q_ref = ctrl.get_joint_q()
        if q_ref is None:
            raise RuntimeError("No joint state available")
        profile = build_profile(args, q_ref=q_ref)
        log = run_rollout(ctrl, trajectory_fn=profile, duration=args.duration, record_hz=args.record_hz)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = (
            Path(args.output)
            if args.output
            else ACTUATORNET_DATASETS_ROOT / f"episode_{args.profile}_{args.joint_group}_{stamp}.npz"
        )
        metadata = {
            "profile": args.profile,
            "joint_group": args.joint_group,
            "duration": args.duration,
            "record_hz": args.record_hz,
            "control_hz": args.control_hz,
            "amplitude": args.amplitude,
            "frequency": args.frequency,
            "f0": args.f0,
            "f1": args.f1,
        }
        save_episode_npz(out_path, log, metadata=metadata)
        print(f"saved: {out_path}")
        print(f"samples: {int(log['t'].shape[0])}")
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
