from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from igris_reliability_runtime.hand_retarget_runtime import RetargetedHandFusion
from igris_teleop.hand_control import retarget_reference as geometry


def landmarks(side):
    tree = ET.parse(Path(geometry.__file__).parent / "hand_urdf" / f"{side}_hand_igris_c.urdf")
    points = np.zeros((21, 3))
    for i, name in zip((5, 9, 13, 17), geometry._MCP_JOINT_NAMES):
        points[i] = np.fromstring(tree.find(f"joint[@name='{name}']/origin").get("xyz"), sep=" ")
    points[[4, 8, 12, 16, 20]] = np.array([
        [.06, .02, .06], [.035, .01, .13], [.015, .01, .15], [-.005, .01, .14], [-.03, .01, .11]
    ])
    return points


class FakeRetargeter:
    def __init__(self, output):
        self.output = np.array(output)
        self.calls = []

    def retarget_single(self, reference, *, left_hand):
        self.calls.append((reference.copy(), left_hand))
        return self.output.copy()


def engine(side="right", *, only=False):
    solvers = [FakeRetargeter([.1, .2, .3, .4, .5, .9]), FakeRetargeter([.9, .8, .7, .6, .5, .1])]
    remaining = iter(solvers)
    node = RetargetedHandFusion(side, openxr_only=only, retargeter_factory=lambda: next(remaining),
                               close_rate_per_sec=1000, open_rate_per_sec=1000)
    return node, solvers


@pytest.mark.parametrize("side", ["left", "right"])
def test_both_sensors_call_same_retarget_method_then_blend_six_motors(side):
    node, solvers = engine(side)
    result = node.update(openxr=(np.ones((5, 3))*.1, 1), mediapipe=(landmarks(side), 1),
                         confidence=.25, now=1.)
    assert result.command == pytest.approx(.25*solvers[0].output+.75*solvers[1].output)
    assert len(result.command) == 6
    assert result.command[0] != result.command[5]  # independent thumb axes
    assert all(len(s.calls) == 1 and s.calls[0][1] == (side == "left") for s in solvers)
    node.update(openxr=(np.ones((5, 3))*.1, 1), mediapipe=(landmarks(side), 1), confidence=.25, now=1.01)
    assert all(len(s.calls) == 1 for s in solvers)  # no filter advancement on duplicate frame


def test_always_one_never_constructs_or_calls_camera_solver():
    node, solvers = engine(only=True)
    first = node.update(openxr=(np.ones((5, 3))*.1, 1), mediapipe=(landmarks("right"),1), confidence=0.,now=1.)
    assert first.command == pytest.approx(solvers[0].output)
    lost = node.update(openxr=None,mediapipe=(landmarks("right"),2),confidence=1.,now=1.1)
    assert lost.command == first.command and not lost.tracked
    assert len(solvers[1].calls) == 0
    assert list(node.retargeters) == ["openxr"]


def test_invalid_observation_holds_not_open_or_closed_and_nan_confidence_not_trusted():
    node, solvers = engine(only=True)
    first = node.update(openxr=(np.ones((5, 3))*.1,1), confidence=1.,now=1.)
    for i, points in enumerate((np.zeros((5,3)), np.full((5,3),np.nan)),2):
        held=node.update(openxr=(points,i),confidence=1.,now=1.+i*.1)
        assert held.command == first.command and not held.tracked
    assert len(solvers[0].calls) == 1
    node, solvers = engine()
    result=node.update(openxr=(np.ones((5,3))*.1,1),mediapipe=(landmarks("right"),1),confidence=float("nan"),now=1.)
    assert result.command == pytest.approx(solvers[1].output)


@pytest.mark.parametrize("side", ["left", "right"])
def test_equal_geometry_produces_equal_commands_with_real_framework(side):
    pytest.importorskip("torch")
    pytest.importorskip("pinocchio")
    from igris_teleop.hand_control.hand_retargeting import HandRetargeting
    points=landmarks(side)
    reference=geometry.mediapipe_landmarks_to_retarget_reference(points,side)
    basis=(geometry.lefthand2igris if side=="left" else geometry.righthand2igris).T @ geometry.grd_yup2grd_zup
    xr=reference @ basis[:3,:3]
    node=RetargetedHandFusion(side,close_rate_per_sec=1000,open_rate_per_sec=1000)
    direct=HandRetargeting(hand_side=side)
    for i in range(3):
        node.update(openxr=(xr,i),mediapipe=(points,i),confidence=.5,now=1.+i/30.)
        expected=direct.retarget_single(reference,left_hand=side=="left")
        assert node.cached["openxr"] == pytest.approx(expected,abs=1e-5)
        assert node.cached["mediapipe"] == pytest.approx(expected,abs=1e-5)
