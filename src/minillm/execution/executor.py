"""Validated execution entry points for phase-separated backend workers."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..engine import RequestState
from .metrics import ExecutionMetrics
from ..contracts import DecodeStateProtocol, ExecutionPhase, DecodeStepResult, PrefillStepResult
from .protocol import ExecutionWorkerProtocol

class ExecutionValidationError(RuntimeError):
    """Input phases or worker results violate the execution contract."""


class _PhaseExecutor:
    def __init__(self, worker: ExecutionWorkerProtocol) -> None:
        self.worker = worker

    def validate_requests(
        self,
        requests: Sequence[RequestState],
        states: Mapping[str, DecodeStateProtocol],
        *,
        prefill: bool,
    ) -> None:
        seen: set[str] = set()
        for request in requests:
            request_id = request.request_id
            if not request_id or request_id in seen:
                raise ExecutionValidationError("empty or duplicate input request_id")
            seen.add(request_id)
            state = states.get(request_id)
            if state is None or state.request_id != request_id:
                raise ExecutionValidationError(f"missing runtime state: {request_id}")
            allowed = state.can_prefill() if prefill else (
                state.can_decode() and state.phase is ExecutionPhase.DECODE
            )
            if request.is_terminal or not allowed:
                raise ExecutionValidationError(f"invalid execution phase: {request_id}")

    @staticmethod
    def validate_results(requests, results, result_type) -> None:
        expected = {request.request_id for request in requests}
        seen: set[str] = set()
        for result in results:
            if not isinstance(result, result_type):
                raise ExecutionValidationError("unexpected worker result type")
            if not result.request_id or result.request_id not in expected:
                raise ExecutionValidationError(f"unknown result request_id: {result.request_id}")
            if result.request_id in seen:
                raise ExecutionValidationError(f"duplicate result request_id: {result.request_id}")
            seen.add(result.request_id)
        if seen != expected:
            raise ExecutionValidationError("worker result count does not match input requests")


class PrefillExecutor(_PhaseExecutor):
    """Call prefill and return a complete validated result set."""

    def execute(self, requests: Sequence[RequestState], states: Mapping[str, DecodeStateProtocol]) -> list[PrefillStepResult]:
        requests = tuple(requests)
        self.validate_requests(requests, states, prefill=True)
        results = list(self.worker.prefill(requests)) if requests else []
        self.validate_results(requests, results, PrefillStepResult)
        return results


class DecodeExecutor(_PhaseExecutor):
    """Call decode only for active requests that completed prefill."""

    def execute(self, requests: Sequence[RequestState], states: Mapping[str, DecodeStateProtocol]) -> list[DecodeStepResult]:
        requests = tuple(requests)
        self.validate_requests(requests, states, prefill=False)
        results = list(self.worker.decode(requests)) if requests else []
        self.validate_results(requests, results, DecodeStepResult)
        return results


class ExecutionManager:
    """Own phase executors and count validated per-request outcomes."""

    def __init__(self, worker: ExecutionWorkerProtocol) -> None:
        self.prefill_executor = PrefillExecutor(worker)
        self.decode_executor = DecodeExecutor(worker)
        self.metrics = ExecutionMetrics()

    def execute_prefill(self, requests, states) -> list[PrefillStepResult]:
        results = self.prefill_executor.execute(requests, states)
        for result in results:
            self.metrics.record_prefill(result.prompt_tokens)
            if result.error is not None:
                self.metrics.record_failure()
        return results

    def execute_decode(self, requests, states) -> list[DecodeStepResult]:
        results = self.decode_executor.execute(requests, states)
        for result in results:
            self.metrics.record_decode(result.token_id if result.error is None else None)
            reason = getattr(result.finish_reason, "value", result.finish_reason)
            if result.error is not None or reason == "error":
                self.metrics.record_failure()
        return results
