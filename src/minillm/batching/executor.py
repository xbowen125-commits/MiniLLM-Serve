"""Logical batch-step coordinator without tensor batching."""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

from ..engine import RequestState
from .batch import BatchState
from .policy import BatchPolicy, FIFOBatchPolicy
from .result import BatchStepResult
from .worker import BatchWorkerProtocol


class BatchStep(Protocol):
    """Extension point for a future worker-backed batch forward step."""

    def __call__(self, requests: Sequence[RequestState]) -> None: ...


def _no_op_step(_requests: Sequence[RequestState]) -> None:
    """Keep architecture tests independent from model execution."""


class BatchExecutor:
    """Create batches and coordinate their request lifecycle one step at a time."""

    def __init__(
        self,
        *,
        policy: BatchPolicy | None = None,
        worker: BatchWorkerProtocol | None = None,
        step: BatchStep | Callable[[Sequence[RequestState]], None] | None = None,
    ) -> None:
        if worker is not None and step is not None:
            raise ValueError("worker and legacy step callback are mutually exclusive")
        self.policy = policy or FIFOBatchPolicy()
        self.worker = worker
        self._step = step or _no_op_step

    def create_batch(
        self,
        requests: Sequence[RequestState],
        *,
        max_batch_size: int,
    ) -> BatchState:
        batch = BatchState()
        for request in self.policy.select(requests, max_batch_size):
            batch.add_request(request)
        return batch

    def execute_step(self, batch: BatchState) -> list[RequestState]:
        """Run one abstract batch step and retire requests made terminal by it."""
        if batch.is_finished:
            raise RuntimeError("cannot execute a finished batch")

        batch.start()
        active = tuple(batch.requests)
        if active:
            if self.worker is not None:
                results = list(self.worker.execute_step(active))
                self._consume_results(batch, active, results)
            else:
                self._step(active)
            batch.step_count += 1
        completed = batch.remove_completed()
        batch.finish_if_idle()
        return completed

    @staticmethod
    def _consume_results(
        batch: BatchState,
        active: Sequence[RequestState],
        results: Sequence[BatchStepResult],
    ) -> None:
        requests_by_id = {request.request_id: request for request in active}
        seen: set[str] = set()
        for result in results:
            if result.request_id not in requests_by_id:
                raise ValueError(f"batch worker returned unknown request_id: {result.request_id}")
            if result.request_id in seen:
                raise ValueError(f"batch worker returned duplicate result: {result.request_id}")
            seen.add(result.request_id)

        for result in results:
            request = requests_by_id[result.request_id]
            if result.token_id is not None:
                request.append_token(result.token_id)
            if result.error is not None:
                request.mark_failed()
            elif result.finished:
                if result.finish_reason_value == "cancelled":
                    request.mark_cancelled()
                elif result.finish_reason_value == "error":
                    request.mark_failed()
                else:
                    request.mark_finished()
            batch.step_results.append(result)
