from __future__ import annotations

import atexit
import os
import select
import signal
import sys
import termios
import tty

import logging_mp
import numpy as np
from ..core.worker_base import SingleRateWorker, WorkerContext
from ..core.events import EventSnapshot
from ..core.state_machine import TransitionResult
from ..head_start_guard import format_head_guard_message, head_guard_is_blocking
from ..policies.walking import (
    DEFAULT_WALKING_POLICY_PROFILE,
    clamp_walking_command,
    resolve_walking_policy_profile_code,
    walking_cmd_step,
    walking_policy_profile_code,
)


logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

ESC = "\x1b"
SPACE = " "

_TTY_FD = None
_OLD_ATTR = None
_CLOSE_FD = False


def _open_tty_once() -> None:
    """Capture the controlling terminal fd and original attrs once."""
    global _TTY_FD, _OLD_ATTR, _CLOSE_FD
    if _TTY_FD is not None:
        return
    try:
        _TTY_FD = os.open("/dev/tty", os.O_RDONLY)
        _CLOSE_FD = True
    except OSError:
        _TTY_FD = sys.stdin.fileno()
        _CLOSE_FD = False
    if not os.isatty(_TTY_FD):
        if _CLOSE_FD:
            try:
                os.close(_TTY_FD)
            except Exception:
                pass
        _TTY_FD = None
        _CLOSE_FD = False
        raise RuntimeError("no controlling tty available")
    _OLD_ATTR = termios.tcgetattr(_TTY_FD)


def _configure_tty() -> None:
    """Set the tty to a non-echo cbreak mode that reads one char at a time."""
    if _TTY_FD is None:
        raise RuntimeError("tty fd not opened")
    tty.setcbreak(_TTY_FD)
    attrs = termios.tcgetattr(_TTY_FD)
    attrs[3] &= ~termios.ECHO
    attrs[1] |= termios.OPOST
    try:
        attrs[1] |= termios.ONLCR
    except AttributeError:
        pass
    attrs[6][termios.VMIN] = 1
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(_TTY_FD, termios.TCSADRAIN, attrs)


def _restore_tty() -> None:
    """Restore the previous tty attributes and close the fd if we opened it."""
    global _TTY_FD, _OLD_ATTR, _CLOSE_FD
    try:
        if _TTY_FD is not None and _OLD_ATTR is not None:
            termios.tcsetattr(_TTY_FD, termios.TCSADRAIN, _OLD_ATTR)
    except Exception:
        pass
    finally:
        if _CLOSE_FD and _TTY_FD not in (None, getattr(sys.stdin, "fileno", lambda: None)()):
            try:
                os.close(_TTY_FD)
            except Exception:
                pass
        _TTY_FD = None
        _OLD_ATTR = None
        _CLOSE_FD = False


atexit.register(_restore_tty)


class KeyboardWorker(SingleRateWorker):
    """단일 키 입력으로 이벤트를 발생시키는 워커.

    - Linux/Unix 환경에서 /dev/tty를 사용해 raw 모드로 1글자 입력을 읽는다.
    - mp.Event(level) 기반 EventBus에 이벤트를 기록한다.

    키 매핑(기본):
      q      -> shutdown (level set)
      r      -> ready (level set)
      s      -> start toggle (level; run)
      h      -> home toggle (start 자동 clear)
      p      -> pause (start/home level clear)
    """

    def __init__(self, ctx: WorkerContext, hz: float = 60.0) -> None:
        super().__init__(ctx, hz=hz)
        self._fd = None
        self._record_shm = None
        self._record_warned = False
        self._mode_shm = None
        self._teleop_guard_shm = None
        self._walking_cmd_shm = None
        self._walking_warned = False

    def on_start(self) -> None:
        try:
            _open_tty_once()
            _configure_tty()
        except Exception as exc:
            logger.warning("[keyboard] raw mode init failed (%s); keyboard worker disabled", exc)
            self._fd = None
            return
        self._fd = _TTY_FD

        logger.info(
            "[keyboard] keys: r(ready) s(start) h(home) p(pause) q(quit) "
            "(ready -> start 순으로 입력하면 RUN 진입) 1(record start) 2(record done) 3(record reset) "
            "i/k(vx +/-) j/l(vy +/-) u/o(yaw +/-) space(cmd zero)"
        )
        if self.ctx.shared_memory:
            self._record_shm = self.ctx.shared_memory.get("record_shm")
            self._mode_shm = self.ctx.shared_memory.get("mode_shm")
            self._teleop_guard_shm = self.ctx.shared_memory.get("teleop_guard_shm")
            self._walking_cmd_shm = self.ctx.shared_memory.get("walking_cmd_shm")
        else:
            self._record_shm = None
            self._mode_shm = None
            self._teleop_guard_shm = None
            self._walking_cmd_shm = None
        self._record_warned = False
        self._walking_warned = False

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if self._fd is None:
            return

        r, _, _ = select.select([self._fd], [], [], 0.0)
        if not r:
            return

        ch = os.read(self._fd, 1)
        if not ch:
            return

        c = ch.decode(errors="ignore")

        # level events
        if c == "q":
            logger.info("[keyboard] q pressed -> request shutdown")
            self.ctx.bus.set_level("shutdown")
            return

        if c == "r":
            if not self._mode_applied_for_level_events():
                logger.info("[keyboard] ready ignored: apply mode workers first")
                return
            self.ctx.bus.set_level("ready")
            return

        if c == "s":
            if not self._start_level_controls_enabled():
                logger.info("[keyboard] start ignored: apply mode workers and set ready first")
                return
            # toggle start (pause clears both start/home)
            if self.ctx.bus.is_level_set("start"):
                self.ctx.bus.clear_level("start")
                self.ctx.bus.clear_level("home")
            else:
                guard_status = self._read_teleop_guard_status()
                if head_guard_is_blocking(guard_status):
                    logger.info("[keyboard] start blocked: %s", format_head_guard_message(guard_status))
                    return
                self.ctx.bus.set_level("start")
                self.ctx.bus.clear_level("home")
            return

        if c == "h":
            if not self._start_level_controls_enabled():
                logger.info("[keyboard] home ignored: apply mode workers and set ready first")
                return
            # toggle home; entering home always clears start to keep them exclusive
            if self.ctx.bus.is_level_set("home"):
                self.ctx.bus.clear_level("home")
            else:
                self.ctx.bus.set_level("home")
                self.ctx.bus.clear_level("start")
            return

        if c == "p":
            self.ctx.bus.clear_level("start")
            self.ctx.bus.clear_level("home")
            return

        if c == "1":
            self._set_record_flag(
                "record_start",
                on_updates={"record_done": False, "record_reset": False},
            )
            return

        if c == "2":
            self._set_record_flag("record_done", on_updates={"record_start": False})
            return

        if c == "3":
            self._set_record_flag(
                "record_reset",
                on_updates={"record_start": False, "record_done": False},
            )
            return

        if c == "i":
            self._adjust_walking_command("vx", self._walking_command_step("vx"))
            return

        if c == "k":
            self._adjust_walking_command("vx", -self._walking_command_step("vx"))
            return

        if c == "j":
            self._adjust_walking_command("vy", self._walking_command_step("vy"))
            return

        if c == "l":
            self._adjust_walking_command("vy", -self._walking_command_step("vy"))
            return

        if c == "u":
            self._adjust_walking_command("dyaw", self._walking_command_step("dyaw"))
            return

        if c == "o":
            self._adjust_walking_command("dyaw", -self._walking_command_step("dyaw"))
            return

        if c == SPACE:
            self._zero_walking_commands()
            return

    def on_stop(self) -> None:
        _restore_tty()
        self._fd = None
        logger.info("[keyboard] stop")

    def _set_record_flag(self, field: str, on_updates: dict[str, bool] | None = None) -> None:
        shm = self._record_shm
        if shm is None:
            if not self._record_warned:
                logger.warning("[keyboard] record shared memory unavailable; 1/2/3 disabled")
                self._record_warned = True
            return
        try:
            data = shm.read_data()
        except Exception:
            logger.exception("[keyboard] failed to read record shared memory")
            return
        updates: dict[str, bool] = {field: True}
        if on_updates:
            updates.update(on_updates)
        try:
            shm.write_data(**updates)
        except Exception:
            logger.exception("[keyboard] failed to write record shared memory")
            return
        logger.info("[keyboard] %s triggered", field)

    def _walking_mode_enabled(self) -> bool:
        if self._mode_shm is None:
            return False
        try:
            data = self._mode_shm.read_data()
            return bool(data.get("walking", False))
        except Exception:
            return False

    def _mode_applied_for_level_events(self) -> bool:
        if self._mode_shm is None:
            return False
        try:
            data = self._mode_shm.read_data()
            return any(bool(data.get(name, False)) for name in ("teleop", "walking", "replay", "deploy"))
        except Exception:
            return False

    def _start_level_controls_enabled(self) -> bool:
        return self._mode_applied_for_level_events() and self.ctx.bus.is_level_set("ready")

    def _read_teleop_guard_status(self) -> dict[str, object] | None:
        shm = self._teleop_guard_shm
        if shm is None:
            return None
        try:
            return shm.read_data()
        except Exception:
            logger.debug("[keyboard] failed to read teleop_guard_shm", exc_info=True)
            return None

    def _walking_profile_name(self) -> str:
        shm = self._walking_cmd_shm
        if shm is None:
            return DEFAULT_WALKING_POLICY_PROFILE
        try:
            data = shm.read_data()
        except Exception:
            return DEFAULT_WALKING_POLICY_PROFILE
        return resolve_walking_policy_profile_code(data.get("profile_code", 0.0))

    def _walking_command_step(self, field: str) -> float:
        return float(walking_cmd_step(self._walking_profile_name()).get(field, 0.05))

    def _adjust_walking_command(self, field: str, delta: float) -> None:
        shm = self._walking_cmd_shm
        if shm is None:
            if not self._walking_warned:
                logger.warning("[keyboard] walking command shared memory unavailable; walking keys disabled")
                self._walking_warned = True
            return
        if not self._walking_mode_enabled():
            return
        try:
            data = shm.read_data()
            current = float(np.asarray(data.get(field, 0.0)).reshape(()).item())
            profile = self._walking_profile_name()
            updated = clamp_walking_command(field, current + delta, profile=profile)
            shm.write_data(
                profile_code=np.float64(walking_policy_profile_code(profile)),
                **{field: np.float64(updated)},
            )
            logger.info("[keyboard] walking %s=%.3f", field, updated)
        except Exception:
            logger.exception("[keyboard] failed to update walking command %s", field)

    def _zero_walking_commands(self) -> None:
        shm = self._walking_cmd_shm
        if shm is None:
            if not self._walking_warned:
                logger.warning("[keyboard] walking command shared memory unavailable; walking keys disabled")
                self._walking_warned = True
            return
        if not self._walking_mode_enabled():
            return
        try:
            profile = self._walking_profile_name()
            shm.write_data(
                profile_code=np.float64(walking_policy_profile_code(profile)),
                vx=np.float64(0.0),
                vy=np.float64(0.0),
                dyaw=np.float64(0.0),
            )
            logger.info("[keyboard] walking commands reset to zero")
        except Exception:
            logger.exception("[keyboard] failed to reset walking commands")
