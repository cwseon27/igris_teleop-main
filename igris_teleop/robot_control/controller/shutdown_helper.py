from __future__ import annotations

import argparse
import json
import sys

import igris_c_sdk as igc_sdk


def _message(result) -> str:
    try:
        return str(result.message())
    except Exception:
        return "<no message>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--timeout-ms", type=int, required=True)
    args = parser.parse_args()

    channel_factory = igc_sdk.ChannelFactory.Instance()
    channel_factory.Init(int(args.domain_id))

    client = igc_sdk.IgrisC_Client()
    client.Init()
    client.SetTimeout(float(args.timeout_ms) / 1000.0)

    mode_res = client.SetControlMode(
        igc_sdk.ControlMode.CONTROL_MODE_HIGH_LEVEL,
        int(args.timeout_ms),
    )
    torque_res = client.SetTorque(
        igc_sdk.TorqueType.TORQUE_OFF,
        int(args.timeout_ms),
    )

    payload = {
        "domain_id": int(args.domain_id),
        "timeout_ms": int(args.timeout_ms),
        "mode_success": bool(mode_res.success()),
        "mode_message": _message(mode_res),
        "torque_success": bool(torque_res.success()),
        "torque_message": _message(torque_res),
    }
    payload["summary"] = (
        "SetControlMode (shutdown): "
        f"{payload['mode_message']}; "
        "Torque OFF: "
        f"{payload['torque_message']}"
    )

    if payload["torque_success"]:
        print(json.dumps(payload, sort_keys=True))
        return 0

    payload["error"] = "Shutdown service calls failed: " + payload["summary"]
    print(json.dumps(payload, sort_keys=True))
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": f"shutdown helper crashed: {exc}"}), file=sys.stderr)
        raise SystemExit(2)
