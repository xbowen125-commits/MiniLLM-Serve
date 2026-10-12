"""Worker protocol for model-independent batch execution."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..engine import RequestState
from .result import BatchStepResult


@runtime_checkable
class BatchWorkerProtocol(Protocol):
    """Produce per-request results for one logical batch step."""

    def execute_step(
        self,
        requests: Sequence[RequestState],
    ) -> Sequence[BatchStepResult]: ...
