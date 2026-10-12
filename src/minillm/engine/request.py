"""Per-request generation state independent from scheduler admission state."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .config import GenerationConfig


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED_TRANSITIONS = {
    RequestStatus.WAITING: {
        RequestStatus.RUNNING,
        RequestStatus.FAILED,
        RequestStatus.CANCELLED,
    },
    RequestStatus.RUNNING: {
        RequestStatus.FINISHED,
        RequestStatus.FAILED,
        RequestStatus.CANCELLED,
    },
}


@dataclass
class RequestState:
    """Mutable lifecycle state owned by one GenerationRunner invocation."""

    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex}")
    prompt: str = ""
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    status: RequestStatus = RequestStatus.WAITING
    created_time: float = field(default_factory=time.time)
    started_time: Optional[float] = None
    finished_time: Optional[float] = None
    generated_tokens: list[int] = field(default_factory=list)
    kv_cache_ref: Any = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            RequestStatus.FINISHED,
            RequestStatus.FAILED,
            RequestStatus.CANCELLED,
        }

    def _transition(self, target: RequestStatus, *, now: Optional[float] = None) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise RuntimeError(f"invalid request transition: {self.status.value} -> {target.value}")

        timestamp = time.time() if now is None else now
        self.status = target
        if target is RequestStatus.RUNNING:
            self.started_time = timestamp
        elif target in {
            RequestStatus.FINISHED,
            RequestStatus.FAILED,
            RequestStatus.CANCELLED,
        }:
            self.finished_time = timestamp

    def mark_running(self, *, now: Optional[float] = None) -> None:
        self._transition(RequestStatus.RUNNING, now=now)

    def mark_finished(self, *, now: Optional[float] = None) -> None:
        self._transition(RequestStatus.FINISHED, now=now)

    def mark_failed(self, *, now: Optional[float] = None) -> None:
        self._transition(RequestStatus.FAILED, now=now)

    def mark_cancelled(self, *, now: Optional[float] = None) -> None:
        self._transition(RequestStatus.CANCELLED, now=now)

    def append_token(self, token_id: int) -> None:
        if self.status is not RequestStatus.RUNNING:
            raise RuntimeError("tokens can only be appended while a request is running")
        self.generated_tokens.append(token_id)
