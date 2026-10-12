"""Structural worker contract for phase-separated LLM execution."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..engine import RequestState
from ..contracts import DecodeStepResult, PrefillStepResult


@runtime_checkable
class ExecutionWorkerProtocol(Protocol):
    """Backend-neutral interface implemented by future model workers."""

    def prefill(
        self,
        requests: Sequence[RequestState],
    ) -> Sequence[PrefillStepResult]: ...

    def decode(
        self,
        requests: Sequence[RequestState],
    ) -> Sequence[DecodeStepResult]: ...
