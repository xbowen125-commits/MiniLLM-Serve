"""Request ordering policies for logical batches."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..engine import RequestState


@runtime_checkable
class BatchPolicy(Protocol):
    """Select requests for a batch without mutating their lifecycle state."""

    def select(
        self,
        requests: Sequence[RequestState],
        max_batch_size: int,
    ) -> list[RequestState]: ...


class FIFOBatchPolicy:
    """Choose the oldest non-terminal requests first."""

    def select(
        self,
        requests: Sequence[RequestState],
        max_batch_size: int,
    ) -> list[RequestState]:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        active = (request for request in requests if not request.is_terminal)
        ordered = sorted(active, key=lambda request: request.created_time)
        return ordered[:max_batch_size]
