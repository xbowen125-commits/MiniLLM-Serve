"""Persistent offline step driver using the model-independent worker protocol."""

from dataclasses import dataclass

from ..batching.batch import BatchState
from ..batching.coordinator import DecodeCoordinator
from ..batching.executor import BatchExecutor as LegacyBatchExecutor
from ..execution.executor import ExecutionManager
from ..execution.protocol import ExecutionWorkerProtocol
from ..runtime.adapter import DecodeStateAdapter
from ..runtime.context import RuntimeRequestContext
from .batch import ContinuousBatch
from .scheduler import BatchScheduler
from .plan import ContinuousBatchPlan
from .step import ContinuousBatchStepResult
from ..contracts import DecodeStepResult, ExecutionPhase, PrefillStepResult, FinishReason
from .worker import ContinuousBatchWorkerProtocol


@dataclass(frozen=True, slots=True)
class ContinuousStep:
    decoded: tuple[str, ...]
    prefilled: tuple[str, ...]
    retired: tuple[str, ...]
    results: tuple[ContinuousBatchStepResult, ...] = ()


class BatchExecutor:
    """One serial cycle: decode existing contexts, then prefill new contexts.

    Worker calls are logical groups, not tensor batching. Intended for fake
    workers and manually driven offline experiments. The caller owns the worker.
    """

    def __init__(self, worker: ExecutionWorkerProtocol | ContinuousBatchWorkerProtocol, *, scheduler: BatchScheduler | None = None) -> None:
        self.batch = ContinuousBatch()
        self.scheduler = scheduler or BatchScheduler()
        self.batch_worker = worker if isinstance(worker, ContinuousBatchWorkerProtocol) else None
        self.coordinator = DecodeCoordinator(
            LegacyBatchExecutor(), execution_manager=ExecutionManager(worker) if self.batch_worker is None else None,
        )
        self._closed = False

    def add(self, context: RuntimeRequestContext) -> None:
        self._require_open()
        self.scheduler.add(self.batch, context)
        self.coordinator.contexts[context.request_id] = context

    @staticmethod
    def _phase_batch(contexts) -> BatchState:
        batch = BatchState()
        for context in contexts:
            batch.add_request(context.request)
        return batch

    def step(self) -> ContinuousStep:
        self._require_open()
        retired = list(self.batch.retire())
        result = self.execute_plan(self.scheduler.build_plan(self.batch))
        return ContinuousStep(result.decoded, result.prefilled,
                              tuple(retired) + result.retired, result.results)

    def execute_plan(self, plan: ContinuousBatchPlan) -> ContinuousStep:
        """Execute only validated plan members; no new selection during forward."""
        self._require_open()
        self._validate_plan(plan)
        retired = []
        results = []
        decode = plan.decode_requests
        previous_counts = {c.request_id: len(c.request.generated_tokens) for c in decode}
        if decode:
            if self.batch_worker is not None:
                self._execute_batch_phase(decode, ExecutionPhase.DECODE)
            else:
                self.coordinator.execute_decode(self._phase_batch(decode))
            for context in decode:
                view = self.coordinator.get_state(context.request_id)
                token = (context.request.generated_tokens[-1]
                         if len(context.request.generated_tokens) > previous_counts[context.request_id] else None)
                token_result = DecodeStepResult(context.request_id, token,
                    context.request.is_terminal, view.finish_reason,
                    RuntimeError(view.error) if view.error is not None else None)
                results.append(ContinuousBatchStepResult(context.request_id,
                    ExecutionPhase.DECODE, token_result, context.request.is_terminal, view.error))
        retired.extend(self.batch.retire())
        slots = max(0, self.scheduler.max_active - len(self.batch.active_requests))
        prefill = plan.prefill_requests[:slots]
        for context in prefill:
            self.batch.active_requests[context.request_id] = self.batch.waiting_requests.pop(context.request_id)
        if prefill:
            try:
                if self.batch_worker is not None:
                    self._execute_batch_phase(prefill, ExecutionPhase.PREFILL)
                else:
                    self.coordinator.execute_prefill(self._phase_batch(prefill))
            except Exception:
                # Whole-step failure leaves WAITING contexts retryable instead
                # of occupying active slots without a valid prefill result.
                for context in prefill:
                    self.batch.waiting_requests[context.request_id] = self.batch.active_requests.pop(context.request_id)
                raise
            for context in prefill:
                view = self.coordinator.get_state(context.request_id)
                results.append(ContinuousBatchStepResult(context.request_id,
                    ExecutionPhase.PREFILL, None, context.request.is_terminal, view.error,
                    finish_reason=view.finish_reason))
        retired.extend(self.batch.retire())
        return ContinuousStep(tuple(c.request_id for c in decode),
                              tuple(c.request_id for c in prefill), tuple(retired), tuple(results))

    def _execute_batch_phase(self, contexts, phase):
        method = self.batch_worker.prefill_batch if phase is ExecutionPhase.PREFILL else self.batch_worker.decode_batch
        results = list(method(contexts))
        expected = {c.request_id for c in contexts}
        seen = set()
        for result in results:
            if not isinstance(result, ContinuousBatchStepResult):
                raise RuntimeError("unexpected batch worker result type")
            if result.request_id not in expected or result.request_id in seen:
                raise RuntimeError("unknown or duplicate batch worker request_id")
            if result.phase is not phase:
                raise RuntimeError("batch worker returned incorrect phase")
            if phase is ExecutionPhase.PREFILL and result.token_id is not None:
                raise RuntimeError("prefill result cannot contain a generated token")
            seen.add(result.request_id)
        if seen != expected:
            raise RuntimeError("batch worker result count mismatch")
        batch = self._phase_batch(contexts)
        if phase is ExecutionPhase.DECODE:
            self.coordinator.apply_decode_results(batch, [DecodeStepResult(
                r.request_id, r.token_id, r.finished, r.finish_reason,
                r.error if isinstance(r.error, Exception) else RuntimeError(r.error) if r.error is not None else None,
            ) for r in results])
        else:
            self.coordinator.apply_prefill_results(batch, [PrefillStepResult(
                r.request_id, 0, False, False,
                r.error if isinstance(r.error, Exception) else RuntimeError(r.error) if r.error is not None else None,
            ) for r in results])
            for result in results:
                view = self.coordinator.get_state(result.request_id)
                if result.finished and not view.is_terminal:
                    if result.finish_reason is FinishReason.CANCELLED:
                        view.cancel()
                    elif result.finish_reason is FinishReason.ERROR:
                        view.fail("worker reported request error")
                    else:
                        view.finish(result.finish_reason or FinishReason.STOP)

    def _validate_plan(self, plan: ContinuousBatchPlan) -> None:
        if plan.active_request_ids != tuple(self.batch.active_requests):
            raise RuntimeError("stale plan active request set")
        seen = set()
        for contexts, group, phase in (
            (plan.decode_requests, self.batch.active_requests, ExecutionPhase.DECODE),
            (plan.prefill_requests, self.batch.waiting_requests, ExecutionPhase.WAITING),
        ):
            for context in contexts:
                if context.request_id in seen or group.get(context.request_id) is not context:
                    raise ValueError("duplicate or foreign plan context")
                seen.add(context.request_id)
                if context.request.is_terminal or context.decode_state.execution_phase is not phase:
                    raise RuntimeError("stale plan request phase or lifecycle")

    def close(self) -> None:
        if self._closed:
            return
        for context in tuple(self.batch.active_requests.values()) + tuple(self.batch.waiting_requests.values()):
            if not context.request.is_terminal:
                DecodeStateAdapter(context).cancel()
        self.batch.retire()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("continuous executor is closed")
