"""Multi-step decode coordination for model-independent logical batches."""

from __future__ import annotations

from ..engine import RequestState
from .batch import BatchState
from .executor import BatchExecutor


class DecodeStepLimitExceeded(RuntimeError):
    """Raised when active requests remain after the configured step limit."""


class DecodeCoordinator:
    """Drive repeated batch steps until every request reaches a terminal state."""

    def __init__(self, executor: BatchExecutor) -> None:
        self.executor = executor

    def step(self, batch: BatchState) -> list[RequestState]:
        """Execute one decode step after retiring externally terminated requests."""
        if batch.is_finished:
            return []

        retired = batch.remove_completed()
        if not batch.requests:
            if batch.started_time is None:
                batch.start()
            batch.finish_if_idle()
            return retired

        retired.extend(self.executor.execute_step(batch))
        return retired

    def run(self, batch: BatchState, *, max_steps: int = 1024) -> BatchState:
        """Run bounded decode steps until the batch finishes."""
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        steps = 0
        while not batch.is_finished and steps < max_steps:
            self.step(batch)
            steps += 1

        if not batch.is_finished:
            raise DecodeStepLimitExceeded(
                f"batch {batch.batch_id} still has {len(batch)} active requests "
                f"after {max_steps} steps"
            )
        return batch

    def cancel_request(
        self,
        batch: BatchState,
        request_id: str,
        *,
        now: float | None = None,
    ) -> bool:
        """Cooperatively cancel and retire one active request from a batch."""
        for request in batch.requests:
            if request.request_id != request_id:
                continue
            if request.is_terminal:
                return False
            request.mark_cancelled(now=now)
            batch.remove_completed()
            if not batch.requests:
                if batch.started_time is None:
                    batch.start(now=now)
                batch.finish_if_idle(now=now)
            return True
        return False
