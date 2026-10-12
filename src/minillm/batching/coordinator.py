"""Multi-step decode coordination for model-independent logical batches."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..engine import RequestState, RequestStatus
from ..contracts import DecodeStepResult, ExecutionPhase, PrefillStepResult
from ..execution.executor import ExecutionManager
from .batch import BatchState
from .executor import BatchExecutor
from .result import BatchStepResult
from .snapshot import RuntimeSnapshot
from ..contracts import DecodeStatus, FinishReason
from ..runtime import RuntimeRequestContext
from ..runtime.adapter import ContextStateViews, DecodeStateAdapter


class DecodeStepLimitExceeded(RuntimeError):
    """Raised when active requests remain after the configured step limit."""


class DecodeCoordinator:
    """Drive repeated batch steps until every request reaches a terminal state."""

    def __init__(self, executor: BatchExecutor, *, execution_manager: ExecutionManager | None = None) -> None:
        self.executor = executor
        self.execution_manager = execution_manager
        self.contexts: dict[str, RuntimeRequestContext] = {}

    @property
    def decode_states(self) -> Mapping[str, DecodeStateAdapter]:
        """Legacy state views; contexts retain all authoritative data."""
        return ContextStateViews(self.contexts)

    def execute_prefill(self, batch: BatchState) -> list[RequestState]:
        """Execute validated prefill through the optional execution manager."""
        if self.execution_manager is None:
            raise RuntimeError("execution manager is not configured")
        self._ensure_states(batch)
        results = self.execution_manager.execute_prefill(tuple(batch.requests), self.decode_states)
        return self.apply_prefill_results(batch, results)

    def execute_decode(self, batch: BatchState) -> list[RequestState]:
        """Execute validated decode through the optional execution manager."""
        if self.execution_manager is None:
            raise RuntimeError("execution manager is not configured")
        self._ensure_states(batch)
        results = self.execution_manager.execute_decode(tuple(batch.requests), self.decode_states)
        return self.apply_decode_results(batch, results)

    def get_state(self, request_id: str) -> DecodeStateAdapter:
        """Return runtime state for a request already observed by this coordinator."""
        try:
            return self.decode_states[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown decode request: {request_id}") from exc

    def snapshot(self, *, now: float | None = None) -> RuntimeSnapshot:
        """Return aggregate runtime counters without prompt or user content."""
        return RuntimeSnapshot.from_states(
            self.decode_states.values(),
            snapshot_time=now,
        )

    def apply_prefill_results(
        self,
        batch: BatchState,
        results: Sequence[PrefillStepResult],
    ) -> list[RequestState]:
        """Apply validated prefill results and advance successful requests."""
        self._ensure_states(batch)
        requests_by_id = {request.request_id: request for request in batch.requests}
        self._validate_phase_results(results, requests_by_id, prefill=True)

        if batch.started_time is None:
            batch.start()
        for result in results:
            request = requests_by_id[result.request_id]
            state = self.decode_states[result.request_id]
            state.start()
            state.transition_phase(ExecutionPhase.PREFILL)
            if result.error is not None:
                state.fail(str(result.error))
                if not request.is_terminal:
                    request.mark_failed(now=state.finished_time)
            else:
                self.contexts[result.request_id].decode_state.record_prefill(result.prompt_tokens)
                state.transition_phase(ExecutionPhase.DECODE)

        retired = batch.remove_completed()
        batch.finish_if_idle()
        return retired

    def apply_decode_results(
        self,
        batch: BatchState,
        results: Sequence[DecodeStepResult],
    ) -> list[RequestState]:
        """Apply results only to requests already in the decode phase."""
        self._ensure_states(batch)
        requests_by_id = {request.request_id: request for request in batch.requests}
        self._validate_phase_results(results, requests_by_id, prefill=False)

        for result in results:
            request = requests_by_id[result.request_id]
            state = self.decode_states[result.request_id]
            state.record_step()
            if result.error is not None:
                state.fail(str(result.error))
                continue

            if result.token_id is not None:
                state.append_token(result.token_id)
            if result.finished and not state.is_terminal:
                reason = result.finish_reason or FinishReason.STOP
                if reason is FinishReason.CANCELLED:
                    state.cancel()
                elif reason is FinishReason.ERROR:
                    state.fail("worker reported request error")
                else:
                    state.finish(reason)
            elif state.is_terminal and not request.is_terminal:
                request.mark_finished(now=state.finished_time)

        retired = batch.remove_completed()
        batch.finish_if_idle()
        return retired

    def step(self, batch: BatchState) -> list[RequestState]:
        """Execute one decode step after retiring externally terminated requests."""
        if batch.is_finished:
            return []

        if self.execution_manager is not None:
            self._ensure_states(batch)
            self._sync_external_terminal_states(batch)
            retired = batch.remove_completed()
            if not batch.requests:
                batch.start()
                batch.finish_if_idle()
                return retired
            phases = {self.decode_states[r.request_id].phase for r in batch.requests}
            if phases == {ExecutionPhase.WAITING}:
                retired.extend(self.execute_prefill(batch))
            elif phases == {ExecutionPhase.DECODE}:
                retired.extend(self.execute_decode(batch))
            else:
                raise RuntimeError("phase-aware step requires a homogeneous batch")
            return retired

        self._ensure_states(batch)
        self._sync_external_terminal_states(batch)
        retired = batch.remove_completed()
        if not batch.requests:
            if batch.started_time is None:
                batch.start()
            batch.finish_if_idle()
            return retired

        active = tuple(batch.requests)
        for request in active:
            state = self.decode_states[request.request_id]
            state.start()
            if state.phase is ExecutionPhase.WAITING:
                state.transition_phase(ExecutionPhase.PREFILL)
            state.record_step()

        result_offset = len(batch.step_results)
        retired.extend(self.executor.execute_step(batch))
        new_results = batch.step_results[result_offset:]
        results_by_id = {result.request_id: result for result in new_results}
        requests_by_id = {request.request_id: request for request in batch.all_requests}
        for result in new_results:
            self._apply_result(result, requests_by_id[result.request_id])

        for request in active:
            state = self.decode_states[request.request_id]
            result = results_by_id.get(request.request_id)
            made_progress = result is not None and (
                result.token_id is not None or result.is_terminal
            )
            if not state.is_terminal and not made_progress:
                state.record_no_progress()
                if state.status is DecodeStatus.FAILED and not request.is_terminal:
                    request.mark_failed(now=state.finished_time)

        length_finished = batch.remove_completed()
        retired.extend(request for request in length_finished if request not in retired)
        batch.finish_if_idle()
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
        self._ensure_states(batch)
        for request in batch.requests:
            if request.request_id != request_id:
                continue
            if request.is_terminal:
                return False
            state = self.decode_states[request_id]
            if not state.is_terminal:
                state.cancel(now=now)
            batch.remove_completed()
            if not batch.requests:
                if batch.started_time is None:
                    batch.start(now=now)
                batch.finish_if_idle(now=now)
            return True
        return False

    def _ensure_states(self, batch: BatchState) -> None:
        for request in batch.all_requests:
            if request.request_id not in self.contexts:
                self.contexts[request.request_id] = RuntimeRequestContext(request)
            elif self.contexts[request.request_id].request is not request:
                raise ValueError("request_id reused with a different request object")

    def _validate_phase_results(
        self,
        results: Sequence[PrefillStepResult] | Sequence[DecodeStepResult],
        requests_by_id: dict[str, RequestState],
        *,
        prefill: bool,
    ) -> None:
        seen: set[str] = set()
        for result in results:
            if result.request_id not in requests_by_id:
                raise ValueError(f"unknown execution request_id: {result.request_id}")
            if result.request_id in seen:
                raise ValueError(f"duplicate execution result: {result.request_id}")
            seen.add(result.request_id)
            state = self.decode_states[result.request_id]
            if prefill and not state.can_prefill():
                raise RuntimeError(
                    f"request {result.request_id} cannot prefill from {state.phase.value}"
                )
            if not prefill and state.phase is not ExecutionPhase.DECODE:
                raise RuntimeError(
                    f"request {result.request_id} cannot decode from {state.phase.value}"
                )

    def _apply_result(
        self,
        result: BatchStepResult,
        request: RequestState,
    ) -> None:
        state = self.decode_states[result.request_id]
        if state.phase is ExecutionPhase.FINISHED:
            return

        if result.error is not None:
            if result.phase is not None:
                state.transition_phase(result.phase)
            state.fail(str(result.error))
        else:
            if result.phase is not None:
                state.transition_phase(result.phase)
            if result.token_id is not None:
                state.append_token(result.token_id)
            if result.finished and state.phase is not ExecutionPhase.FINISHED:
                reason = self._finish_reason(result.finish_reason_value)
                if reason is FinishReason.CANCELLED:
                    state.cancel()
                elif reason is FinishReason.ERROR:
                    state.fail("worker reported request error")
                else:
                    state.finish(reason)

        if state.status is DecodeStatus.FINISHED and not request.is_terminal:
            request.mark_finished(now=state.finished_time)
        elif state.status is DecodeStatus.FAILED and not request.is_terminal:
            request.mark_failed(now=state.finished_time)
        elif state.status is DecodeStatus.CANCELLED and not request.is_terminal:
            request.mark_cancelled(now=state.finished_time)

    def _sync_external_terminal_states(self, batch: BatchState) -> None:
        for request in batch.all_requests:
            state = self.decode_states[request.request_id]
            if state.phase is ExecutionPhase.FINISHED or not request.is_terminal:
                continue
            if request.status is RequestStatus.CANCELLED:
                state.cancel(now=request.finished_time)
            elif request.status is RequestStatus.FAILED:
                state.fail("request failed outside decode runtime", now=request.finished_time)
            elif request.status is RequestStatus.FINISHED:
                state.finish(FinishReason.STOP, now=request.finished_time)

    @staticmethod
    def _finish_reason(reason: str | None) -> FinishReason:
        if reason is None:
            return FinishReason.STOP
        return FinishReason(reason)
