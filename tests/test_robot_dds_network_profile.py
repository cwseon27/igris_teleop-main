from pathlib import Path
import xml.etree.ElementTree as ET


def test_robot_dds_example_uses_explicit_pc_lan_address() -> None:
    # Per-PC local_state is intentionally not distributed. Verify the shipped
    # example, not the developer's private settings or a required PC address.
    profile = Path(__file__).resolve().parents[1] / "docs" / "examples" / "cyclonedds_igris_lan.xml"
    root = ET.parse(profile).getroot()
    interfaces = root.findall(".//NetworkInterface")

    assert interfaces
    assert [str(item.attrib.get("address", "")).strip() for item in interfaces] == [
        "192.168.11.18"
    ]
    assert all(not str(item.attrib.get("name", "")).strip() for item in interfaces)
    assert all(
        str(item.attrib.get("autodetermine", "false")).strip().lower()
        not in {"1", "true", "yes", "on"}
        for item in interfaces
    )
    assert root.findtext(".//DontRoute", "").strip().lower() == "true"
