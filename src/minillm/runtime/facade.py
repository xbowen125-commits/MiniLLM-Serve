"""Synchronous offline runtime entry point; no serving or GPU integration."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

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

    def __init__(self, worker: ExecutionWorkerProtocol, *, max_batch_size: int = 4) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        # Local imports avoid making runtime package initialization depend on
        # batching, whose compatibility adapter imports runtime.context.
        from ..batching import BatchExecutor, DecodeCoordinator
        from ..execution.executor import ExecutionManager
        self._manager = ExecutionManager(worker)
        self._coordinator = DecodeCoordinator(BatchExecutor(), execution_manager=self._manager)
        self.max_batch_size = max_batch_size
        self._closed = False
        self._executing = False
        self._lock = threading.RLock()
        self._archive: dict[str, RuntimeResult] = {}

    def submit(self, prompt: str, generation_config: GenerationConfig | None = None,
               *, request_id: str | None = None) -> RuntimeRequestContext:
        """Create and register the only context for a request ID."""
        with self._lock:
            self._require_open()
            config = generation_config or GenerationConfig()
            if config.max_tokens < 1:
                raise ValueError("max_tokens must be at least 1")
            request = RequestState(prompt=prompt, generation_config=config)
            if request_id is not None:
                if not request_id:
                    raise ValueError("request_id must not be empty")
                request.request_id = request_id
            if request.request_id in self._coordinator.contexts:
                raise ValueError("duplicate request_id")
            context = RuntimeRequestContext(request)
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
                return False
            if self._executing:
                raise RuntimeError("cancel must occur between runtime steps")
            DecodeStateAdapter(context).cancel()
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
                                     decode.prefill_tokens, decode.decode_tokens),
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

    def close(self) -> None:
        """Cancel unfinished requests, drop contexts, and preserve results."""
        with self._lock:
            if self._closed:
                return
            if self._executing:
                raise RuntimeError("close must occur between runtime steps")
            for request_id, context in tuple(self._coordinator.contexts.items()):
                if not context.request.is_terminal:
                    self.cancel(request_id)
                context.request.kv_cache_ref = None
                self._archive[request_id] = self.get_result(request_id)
            self._coordinator.contexts.clear()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
