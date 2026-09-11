from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HAND_PACKAGE = ROOT / "ros_ws/src/igris_c_ros_bridge/igris_c_hand"


def test_hand_bridge_uses_robot_matched_sdk_snapshot() -> None:
    cmake = (HAND_PACKAGE / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "target_link_libraries(igris_c_hand_bridge_node igris_sdk::igris_sdk)" in cmake
    assert "robot_hand_sdk" in cmake
    assert "ROBOT_HAND_SDK" in cmake


def test_hand_command_matches_headered_installed_robot_schema() -> None:
    source = (HAND_PACKAGE / "src/igris_c_hand_bridge_node.cpp").read_text(encoding="utf-8")

    assert "command.motor_cmd().resize(kHandMotorCount)" in source
    assert "command.header()" in source
    assert "stamp_header" in source
