"""Synchronous offline runtime entry point; no serving or GPU integration."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

from ..contracts import ExecutionPhase, FinishReason
from ..engine.config import GenerationConfig
from ..engine.request import RequestState, RequestStatus
from ..execution.protocol import ExecutionWorkerProtocol
from .context import RuntimeRequestContext


@dataclass(frozen=True, slots=True)
class RuntimeResultMetrics:
    first_token_latency_ms: float | None
    total_latency_ms: float | None
    decode_steps: int
    prefill_tokens: int
    decode_tokens: int
    arrival_ttft_ms: float | None = None
    tpot_ms: float | None = None


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    """Immutable observation of one request; never includes its prompt."""

    request_id: str
    status: RequestStatus
    generated_tokens: tuple[int, ...]
    finish_reason: FinishReason | None
    error: str | None
    metrics: RuntimeResultMetrics


class RuntimeFacade:
    """Manually advance homogeneous bounded batches using a supplied worker.

    The caller owns the worker. This facade owns request contexts, cancels them
    on close, and retains immutable results while discarding mutable contexts.
    Calls are serialized; close waits for any synchronous worker call to return.
    """

    def __init__(self, worker: ExecutionWorkerProtocol, *, max_batch_size: int = 4,
                 execution_mode: str = "legacy", scheduler=None, owns_worker: bool = False) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        # Local imports avoid making runtime package initialization depend on
        # batching, whose compatibility adapter imports runtime.context.
        from ..batching import BatchExecutor, DecodeCoordinator
        from ..execution.executor import ExecutionManager
        if execution_mode not in {"legacy", "continuous"}:
            raise ValueError("unknown execution mode")
        self._continuous = None
        self._binding = None
        self._owner = object()
        self._worker = worker
        self._owns_worker = owns_worker
        if execution_mode == "continuous":
            from ..continuous.executor import BatchExecutor as ContinuousExecutor
            from ..continuous.iteration_scheduler import IterationScheduler
            from ..continuous.worker import ManagedBatchWorkerProtocol
            from .resources import bind_worker
            if not isinstance(worker, ManagedBatchWorkerProtocol):
                raise TypeError("continuous mode requires batch methods and release_request")
            limit = getattr(worker, "max_batch_size", None)
            capacity = getattr(getattr(worker, "pool", None), "capacity", None)
            if limit is None or capacity is None or max_batch_size > min(limit, capacity):
                raise ValueError("Worker batch/slot capacity mismatch")
            scheduler = scheduler or IterationScheduler(max_batch_size=max_batch_size)
            if not isinstance(scheduler, IterationScheduler) or scheduler.max_active != max_batch_size:
                raise ValueError("iteration scheduler capacity mismatch")
            if owns_worker and not callable(getattr(worker, "close", None)):
                raise TypeError("owned Worker requires close")
            self._continuous = ContinuousExecutor(worker, scheduler=scheduler)
            self._manager = None
            self._coordinator = self._continuous.coordinator
            self._binding = bind_worker(worker)
            self._binding.attach(self._owner, owns_worker)
        else:
            if scheduler is not None or owns_worker:
                raise ValueError("scheduler/ownership options require continuous mode")
            self._manager = ExecutionManager(worker)
            self._coordinator = DecodeCoordinator(BatchExecutor(), execution_manager=self._manager)
        self.max_batch_size = max_batch_size
        self._closed = False
        self._executing = False
        self._lock = self._binding.lock if self._binding else threading.RLock()
        self._archive: dict[str, RuntimeResult] = {}

    def submit(self, prompt: str, generation_config: GenerationConfig | None = None,
               *, request_id: str | None = None,
               request_state: RequestState | None = None) -> RuntimeRequestContext:
        """Create and register the only context for a request ID."""
        with self._lock:
            self._require_open()
            config = generation_config or (request_state.generation_config if request_state is not None else GenerationConfig())
            if config.max_tokens < 1:
                raise ValueError("max_tokens must be at least 1")
            request = request_state or RequestState(prompt=prompt, generation_config=config)
            if request_state is not None:
                if request.status is not RequestStatus.WAITING:
                    raise ValueError("handoff requires a WAITING RequestState")
                if request_id is not None and request_id != request.request_id:
                    raise ValueError("handoff request_id mismatch")
                request.prompt = prompt
                request.generation_config = config
            if request_id is not None:
                if not request_id:
                    raise ValueError("request_id must not be empty")
                request.request_id = request_id
            if request.request_id in self._coordinator.contexts:
                raise ValueError("duplicate request_id")
            context = RuntimeRequestContext(request)
            if self._continuous is not None:
                if request.request_id in self._binding.claims:
                    raise ValueError("request_id already claimed on shared Worker")
                self._continuous.add(context)
                self._binding.claims[request.request_id] = self._owner
            else:
                self._coordinator.contexts[request.request_id] = context
            return context

    def step(self) -> tuple[str, ...]:
        """Execute one phase for at most max_batch_size active requests."""
        from ..batching.batch import BatchState
        from ..scheduler.phase_policy import PhaseAwareBatchPolicy
        with self._lock:
            self._require_open()
            if self._executing:
                raise RuntimeError("runtime step cannot be reentered")
            if self._continuous is not None:
                self._executing = True
                try:
                    self._continuous.release_terminal(strict=False)
                    plan = self._continuous.scheduler.build_plan(self._continuous.batch)
                    free = self._worker.pool.capacity - self._worker.pool.snapshot()["allocated_slots"]
                    plan = replace(plan, prefill_requests=plan.prefill_requests[:max(0, free)])
                    result = self._continuous.execute_plan(plan)
                    return result.decoded + result.prefilled
                finally:
                    self._executing = False
            requests = [c.request for c in self._coordinator.contexts.values() if not c.request.is_terminal]
            if not requests:
                return ()
            policy = PhaseAwareBatchPolicy(self._coordinator.decode_states)
            ordered = policy.select_batch(requests, len(requests))
            phase = policy.phase_for(ordered[0])
            selected = [r for r in ordered if policy.phase_for(r) is phase][:self.max_batch_size]
            batch = BatchState()
            for request in selected:
                batch.add_request(request)
            self._executing = True
            try:
                self._coordinator.step(batch)
            finally:
                self._executing = False
            return tuple(r.request_id for r in selected)

    def cancel(self, request_id: str) -> bool:
        from .adapter import DecodeStateAdapter
        with self._lock:
            self._require_open()
            context = self._coordinator.contexts[request_id]
            if context.request.is_terminal:
                if self._continuous is not None:
                    self._continuous.release_terminal(request_id)
                return False
            if self._executing:
                raise RuntimeError("cancel must occur between runtime steps")
            DecodeStateAdapter(context).cancel()
            if self._continuous is not None:
                self._continuous.release_terminal(request_id)
            return True

    def get_result(self, request_id: str) -> RuntimeResult:
        from .adapter import DecodeStateAdapter
        with self._lock:
            if request_id in self._archive:
                return self._archive[request_id]
            context = self._coordinator.contexts[request_id]
            view = DecodeStateAdapter(context)
            decode = context.decode_state
            return RuntimeResult(
                request_id, context.request.status, tuple(context.request.generated_tokens),
                view.finish_reason, view.error,
                RuntimeResultMetrics(context.metrics.first_token_latency_ms,
                                     context.metrics.total_latency_ms, decode.decode_steps,
                                     decode.prefill_tokens, decode.decode_tokens,
                                     None if decode.first_token_time is None else
                                     (decode.first_token_time - context.request.created_time) * 1000,
                                     None if decode.decode_tokens < 2 or decode.first_token_time is None or decode.last_token_time is None else
                                     (decode.last_token_time - decode.first_token_time) * 1000 / (decode.decode_tokens - 1)),
            )

    def snapshot(self):
        from ..batching.snapshot import RuntimeSnapshot
        with self._lock:
            if not self._closed:
                return self._coordinator.snapshot()
            results = tuple(self._archive.values())
            return RuntimeSnapshot(
                0,
                sum(r.status is RequestStatus.FINISHED for r in results),
                sum(r.status is RequestStatus.FAILED for r in results),
                sum(r.status is RequestStatus.CANCELLED for r in results),
                sum(len(r.generated_tokens) for r in results),
                sum(r.metrics.decode_steps for r in results), time.time(),
            )

    def acknowledge(self, request_id: str) -> None:
        """Forget a consumed terminal result only after backend cleanup succeeds.

        The caller must first copy its result. Failure retains all ownership and
        can be retried; no execution or active result can be forgotten.
        """
        with self._lock:
            self._require_open()
            context = self._coordinator.contexts.get(request_id)
            if context is None:
                return
            if not context.request.is_terminal:
                raise RuntimeError("cannot acknowledge an active request")
            if self._continuous is not None:
                self._continuous.release_terminal(request_id)
                if request_id not in self._continuous._released:
                    raise RuntimeError("request cleanup not confirmed")
                self._continuous.batch.finished_requests.pop(request_id, None)
                self._continuous._released.discard(request_id)
                self._continuous.scheduler._observed_finished.discard(request_id)
                self._binding.claims.pop(request_id, None)
            self._coordinator.contexts.pop(request_id)

    def retention(self) -> dict[str, int]:
        """Prompt-free retained references, for bounded-history verification."""
        with self._lock:
            return {
                "contexts": len(self._coordinator.contexts),
                "claims": sum(v is self._owner for v in self._binding.claims.values()) if self._binding else 0,
                "released_ids": len(self._continuous._released) if self._continuous else 0,
                "finished_ids": len(self._continuous.scheduler._observed_finished) if self._continuous else 0,
            }

    def close(self) -> None:
        """Cancel unfinished requests, drop contexts, and preserve results."""
        with self._lock:
            if self._closed:
                return
            if self._executing:
                raise RuntimeError("close must occur between runtime steps")
            if self._continuous is not None:
                self._continuous.close()
                if self._owns_worker:
                    self._worker.close()
            for request_id, context in tuple(self._coordinator.contexts.items()):
                if not context.request.is_terminal:
                    self.cancel(request_id)
                context.request.kv_cache_ref = None
                self._archive[request_id] = self.get_result(request_id)
                if self._binding is not None:
                    self._binding.claims.pop(request_id, None)
            self._coordinator.contexts.clear()
            if self._continuous is not None:
                self._continuous.batch.finished_requests.clear()
                self._binding.clients.pop(self._owner, None)
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
