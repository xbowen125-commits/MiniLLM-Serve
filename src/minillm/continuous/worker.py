"""Logical batch worker protocol; no tensor or model backend dependency."""

from typing import Protocol, Sequence, runtime_checkable

from ..runtime.context import RuntimeRequestContext
from .result import ContinuousBatchStepResult


@runtime_checkable
class ContinuousBatchWorkerProtocol(Protocol):
    """Return one result per context without changing request state."""

    def prefill_batch(self, requests: Sequence[RuntimeRequestContext]) -> Sequence[ContinuousBatchStepResult]: ...

    def decode_batch(self, requests: Sequence[RuntimeRequestContext]) -> Sequence[ContinuousBatchStepResult]: ...
