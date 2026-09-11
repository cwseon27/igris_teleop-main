from __future__ import annotations

import sys
import time

import logging_mp
from ..core.worker_base import SingleRateWorker, WorkerContext
from ..core.events import EventSnapshot
from ..core.state_machine import ModeState, TransitionResult


logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


class CliEventWorker(SingleRateWorker):
    """표준입력으로 이벤트를 발생시키는 간단한 워커.

    입력은 라인 단위로 받으며, 플랫폼/터미널 제약을 최소화한다.
    """

    def __init__(self, ctx: WorkerContext, hz: float = 10.0) -> None:
        super().__init__(ctx, hz=hz)

    def on_start(self) -> None:
        logger.info(
            "[cli] commands: ready | start | pause | quit\n"
            "      (start는 toggle, pause는 start clear와 동일; 참고용 워커)"
        )

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        # 입력은 non-blocking이 아니므로, 작은 예제에서는 polling 대신
        # stdin에 데이터가 있을 때만 처리한다.
        # (간단성을 위해 select를 쓰지 않고, 입력이 없으면 그냥 넘어감)
        if not sys.stdin.closed and sys.stdin in getattr(sys, "stdin", []):
            pass

        # 매우 단순: stdin에 줄이 들어오면 처리
        # 주의: 일반적으로는 별도 스레드 + queue가 안정적이다.
        if sys.stdin.closed:
            return

        if sys.stdin.readable():
            # peek가 어렵기 때문에, 예제에서는 사용자에게 "엔터" 입력을 전제로 한다.
            # 입력이 없으면 EOFError가 날 수 있어 try로 감싼다.
            try:
                if sys.stdin in (None,):
                    return
            except Exception:
                return

        # input()을 쓰면 블로킹이므로, 예제의 루프와 충돌한다.
        # 따라서 cli 워커는 main에서 별도 프로세스/스레드로 실행하는 것을 권장한다.
        # (이 파일은 참고용이며, main에서는 스레드 기반 CLI를 사용한다.)
        return
