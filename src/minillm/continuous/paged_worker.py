"""Offline batch-worker decorator with simulated paged KV accounting."""

from collections.abc import Callable, Sequence

from ..contracts import ExecutionPhase, FinishReason
from ..memory import PagedKVCacheManager
from ..runtime.context import RuntimeRequestContext
from .result import ContinuousBatchStepResult
from .worker import ContinuousBatchWorkerProtocol


class PagedContinuousBatchWorker:
    """Reserve prompt pages and grow request-local pages after decode tokens.

    The backend still owns real cache tensors and model execution. Synthetic
    block IDs never enter model.forward or attention. This decorator is serial,
    offline, and caller-owned: call close after executor shutdown/cancellation.
    Token counts use the backend input builder or an explicit counting callback;
    they are never estimated from character or word counts.
    """

    def __init__(self, worker: ContinuousBatchWorkerProtocol,
                 cache_manager: PagedKVCacheManager, *,
                 prompt_token_count: Callable[[RuntimeRequestContext], int] | None = None):
        self.worker = worker
        self.cache_manager = cache_manager
        self._count_prompt = prompt_token_count or self._builder_token_count
        self._contexts: dict[str, RuntimeRequestContext] = {}
        self._closed = False

    def _builder_token_count(self, context):
        inputs = self.worker.builder.build_prefill([context])
        return int(inputs.attention_mask.sum().item())

    @staticmethod
    def _error(request_id, phase, error):
        return ContinuousBatchStepResult(request_id, phase, finished=True,
                                         finish_reason=FinishReason.ERROR, error=error)

    def _prepare(self, requests):
        if self._closed:
            raise RuntimeError("paged batch worker is closed")
        requests = tuple(requests)
        if not requests or len({c.request_id for c in requests}) != len(requests):
            raise ValueError("batch must be nonempty with unique request IDs")
        for request_id, context in tuple(self._contexts.items()):
            if context.request.is_terminal:
                self.release_request(request_id)
        return requests

    def _execute(self, requests, phase):
        """Validate complete results before changing any token accounting."""
        try:
            method = (self.worker.prefill_batch if phase is ExecutionPhase.PREFILL
                      else self.worker.decode_batch)
            results = list(method(requests))
            expected = {c.request_id for c in requests}
            mapped = {}
            for result in results:
                if (not isinstance(result, ContinuousBatchStepResult)
                        or result.request_id not in expected
                        or result.request_id in mapped or result.phase is not phase
                        or (phase is ExecutionPhase.PREFILL and result.token_id is not None)):
                    raise RuntimeError("invalid paged worker result mapping or phase")
                mapped[result.request_id] = result
            if set(mapped) != expected:
                raise RuntimeError("paged worker result count mismatch")
            return mapped
        except Exception:
            for context in requests:
                self.release_request(context.request_id)
            raise

    def prefill_batch(self, requests: Sequence[RuntimeRequestContext]):
        requests = self._prepare(requests)
        accepted, results = [], {}
        for context in requests:
            request_id = context.request_id
            allocated = False
            try:
                if request_id in self._contexts:
                    raise ValueError("request already owns paged cache")
                if context.request.is_terminal or context.decode_state.execution_phase is not ExecutionPhase.WAITING:
                    raise RuntimeError("prefill requires a live WAITING request")
                count = self._count_prompt(context)
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError("prompt token count must be a positive integer")
                self.cache_manager.allocate_request(request_id)
                self._contexts[request_id] = context
                allocated = True
                self.cache_manager.append_tokens(request_id, count)
                accepted.append(context)
            except Exception as error:
                # Never release a foreign context or an existing registration.
                if allocated:
                    self.release_request(request_id)
                results[request_id] = self._error(request_id, ExecutionPhase.PREFILL, error)
        if accepted:
            results.update(self._execute(accepted, ExecutionPhase.PREFILL))
            for context in accepted:
                result = results[context.request_id]
                if result.error is not None or result.finished:
                    self.release_request(context.request_id)
        return [results[c.request_id] for c in requests]

    def decode_batch(self, requests: Sequence[RuntimeRequestContext]):
        requests = self._prepare(requests)
        accepted, results = [], {}
        for context in requests:
            try:
                if self._contexts.get(context.request_id) is not context:
                    raise KeyError("request does not own paged cache")
                if context.decode_state.execution_phase is not ExecutionPhase.DECODE:
                    raise RuntimeError("decode requires DECODE phase")
                self.cache_manager.get_block_table(context.request_id)
                accepted.append(context)
            except Exception as error:
                results[context.request_id] = self._error(context.request_id, ExecutionPhase.DECODE, error)
        if accepted:
            decoded = self._execute(accepted, ExecutionPhase.DECODE)
            for context in accepted:
                result = decoded[context.request_id]
                try:
                    if result.error is None and result.token_id is not None:
                        # Capacity is simulated, not rollback of backend forward.
                        self.cache_manager.append_tokens(context.request_id, 1)
                        if not result.finished and len(context.request.generated_tokens) + 1 >= context.request.generation_config.max_tokens:
                            result = ContinuousBatchStepResult(context.request_id,
                                ExecutionPhase.DECODE, token_id=result.token_id,
                                finished=True, finish_reason=FinishReason.LENGTH)
                except Exception as error:
                    result = self._error(context.request_id, ExecutionPhase.DECODE, error)
                if result.error is not None or result.finished:
                    self.release_request(context.request_id)
                results[context.request_id] = result
        return [results[c.request_id] for c in requests]

    def get_block_table(self, request_id):
        return self.cache_manager.get_block_table(request_id)

    def snapshot(self):
        return self.cache_manager.snapshot()

    def release_request(self, request_id):
        """Release both owned bookkeeping and backend references, idempotently."""
        if request_id not in self._contexts:
            return
        try:
            release = getattr(self.worker, "release_request", None)
            if release is not None:
                release(request_id)
        finally:
            self.cache_manager.release_request(request_id)
            self._contexts.pop(request_id, None)

    def close(self):
        """Release this wrapper's requests; do not close a shared backend."""
        for request_id in tuple(self._contexts):
            self.release_request(request_id)
        self._closed = True
