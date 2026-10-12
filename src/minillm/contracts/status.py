"""Runtime statuses and state requirements for execution validation."""

from enum import Enum
from typing import Protocol

from .phase import ExecutionPhase


class DecodeStatus(str, Enum):
    ACTIVE = "active"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"


class DecodeStateProtocol(Protocol):
    """State view consumed by execution validators without runtime imports."""

    request_id: str
    phase: ExecutionPhase

    def can_prefill(self) -> bool: ...

    def can_decode(self) -> bool: ...
