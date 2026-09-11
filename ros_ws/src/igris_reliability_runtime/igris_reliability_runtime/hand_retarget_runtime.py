"""Same DexRetargeting model per sensor, fusion only after motor conversion."""
from __future__ import annotations

import numpy as np

from igris_teleop.hand_control.retarget_reference import (
    openxr_fingertips_to_retarget_reference as openxr_to_retarget,
    mediapipe_landmarks_to_retarget_reference as mediapipe_to_retarget,
)
from openxr_hand_to_igris_viewer.hand_fusion import ReliabilityAwareHandCommandFusion


class RetargetedHandFusion:
    def __init__(self, side: str, *, openxr_only=False, retargeter_factory=None, **fusion_options):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if retargeter_factory is None:
            from igris_teleop.hand_control.hand_retargeting import HandRetargeting
            retargeter_factory = lambda: HandRetargeting(hand_side=side)
        self.side = side
        self.openxr_only = bool(openxr_only)
        self.retargeters = {"openxr": retargeter_factory()}
        if not self.openxr_only:
            self.retargeters["mediapipe"] = retargeter_factory()
        self.tokens = {}
        self.cached = {}
        self.last_error = {}
        self.fusion = ReliabilityAwareHandCommandFusion(
            openxr_only=self.openxr_only, command_size=6, **fusion_options
        )

    def _command(self, source, observation):
        if observation is None or source not in self.retargeters:
            return None
        points, token = observation
        if self.tokens.get(source) == token:
            return self.cached.get(source)
        self.tokens[source] = token
        try:
            adapter = openxr_to_retarget if source == "openxr" else mediapipe_to_retarget
            reference = adapter(points, self.side)
            # Same method as the VR retarget framework, including its joint order
            # and close gain exactly once. Each sensor owns its optimizer history.
            command = np.asarray(self.retargeters[source].retarget_single(
                reference, left_hand=self.side == "left"
            ), dtype=np.float64)
            if command.shape != (6,) or not np.all(np.isfinite(command)):
                raise ValueError("invalid retargeted motor vector")
            if np.any(command < 0.0) or np.any(command > 1.0):
                raise ValueError("retargeted motors outside normalized range")
            self.cached[source] = command
            self.last_error.pop(source, None)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            self.cached[source] = None
            self.last_error[source] = str(exc)
        return self.cached[source]

    def update(self, *, openxr=None, mediapipe=None, confidence=0.0, now):
        xr = self._command("openxr", openxr)
        mp = self._command("mediapipe", mediapipe) if not self.openxr_only else None
        return self.fusion.update(
            openxr_close=xr, mediapipe_close=mp, confidence=confidence,
            openxr_ready=xr is not None, mediapipe_ready=mp is not None, now=now,
        )
