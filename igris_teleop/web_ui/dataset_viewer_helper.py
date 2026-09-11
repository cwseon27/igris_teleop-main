from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

os.environ["IGRIS_DATASET_VIEWER_DISABLE_EXTERNAL"] = "1"

from igris_teleop.web_ui.server import (  # noqa: E402
    DEFAULT_CAMERA_MODE,
    DATASET_VIEWER_DEFAULT_MAX_POINTS,
    WebUIState,
    _json_scalar,
)


class _HelperBus:
    def __init__(self) -> None:
        self.levels: dict[str, bool] = {}

    def set_level(self, name: str) -> None:
        self.levels[name] = True

    def clear_level(self, name: str) -> None:
        self.levels[name] = False

    def is_level_set(self, name: str) -> bool:
        return bool(self.levels.get(name, False))


class _HelperSupervisor:
    def alive_map(self) -> dict[str, bool]:
        return {}

    def current_manual_workers(self) -> set[str]:
        return set()

    def current_mode_workers(self) -> set[str]:
        return set()

    def list_manual_workers(self) -> list[str]:
        return []

    def list_selectable_mode_workers(
        self,
        mode: str | None,
        teleop_device: str | None,
        inference_policy: str | None = None,
    ) -> list[str]:
        return []


def _make_state() -> WebUIState:
    return WebUIState(
        bus=_HelperBus(),  # type: ignore[arg-type]
        supervisor=_HelperSupervisor(),
        shared_memory={},
        log_queue=None,
        initial_mode=None,
        initial_teleop_device=None,
        initial_teleop_hand_source=None,
        initial_walking_policy_profile="v1",
        initial_walking_policy_path=None,
        initial_camera_mode=DEFAULT_CAMERA_MODE,
    )


def _episode(args: argparse.Namespace) -> dict[str, Any]:
    state = _make_state()
    try:
        return state.dataset_viewer_episode(
            root=args.root,
            dataset=args.dataset,
            episode=args.episode,
            image_key=args.image_key,
            max_points=args.max_points,
        )
    finally:
        state._close_dataset_viewer_cache()


def _save_labels(args: argparse.Namespace) -> dict[str, Any]:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        raise ValueError(f"Invalid save-labels JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("save-labels payload must be a JSON object")

    state = _make_state()
    try:
        return state.dataset_viewer_save_labels(
            root=None,
            dataset=args.dataset,
            episode=args.episode,
            classes=payload.get("classes", []),
            segments=payload.get("segments", []),
        )
    finally:
        state._close_dataset_viewer_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    episode = sub.add_parser("episode")
    episode.add_argument("--root", required=True)
    episode.add_argument("--dataset", required=True)
    episode.add_argument("--episode", type=int, required=True)
    episode.add_argument("--image-key", default=None)
    episode.add_argument("--max-points", type=int, default=DATASET_VIEWER_DEFAULT_MAX_POINTS)
    save_labels = sub.add_parser("save-labels")
    save_labels.add_argument("--dataset", required=True)
    save_labels.add_argument("--episode", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "episode":
        payload = _episode(args)
    elif args.command == "save-labels":
        payload = _save_labels(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, default=_json_scalar), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
