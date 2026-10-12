"""Experimental CPU batch forward adapter with basic copied DynamicCache rows."""

from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..contracts import ExecutionPhase, FinishReason
from ..continuous.inputs import BatchInputBuilder
from ..continuous.kv import BatchKVAdapter
from ..continuous.result import ContinuousBatchStepResult
from ..engine.cache import KVCacheManager
from ..engine.generation import sample
from .model_loader import LoadedModel


@dataclass
class _BatchSession:
    context: Any
    mask: torch.Tensor
    logits: torch.Tensor
    generator: Any = None
    pending_first: bool = True
    emitted: int = 0


class TransformersBatchWorker:
    """CPU-only, at most two contexts per call; no scheduling integration.

    Basic K/V concatenation and cloned row splitting are correctness scaffolding,
    not a cache optimization. Only layers exposing keys/values and update() are
    supported. Full padded masks and position IDs are carried across decode.
    First-token decode consumes prefill logits; later decode performs one forward
    for all eligible rows. Caller applies token/lifecycle results and owns cleanup.
    """

    def __init__(self, loaded: LoadedModel, *, cache_factory: Callable[[Any], Any] | None = None):
        if loaded.device != "cpu":
            raise ValueError("TransformersBatchWorker supports CPU only")
        self.loaded = loaded
        self.builder = BatchInputBuilder(loaded.tokenizer)
        self.cache_factory = cache_factory or KVCacheManager().create
        self.kv = BatchKVAdapter()
        self._sessions = {}

    @staticmethod
    def _validate_batch(requests):
        contexts = tuple(requests)
        if not 1 <= len(contexts) <= 2:
            raise ValueError("batch size must be 1 or 2")
        if len({c.request_id for c in contexts}) != len(contexts):
            raise ValueError("duplicate request_id")
        return contexts

    def _new_cache(self):
        return self.cache_factory(self.loaded.model.config)

    def _split(self, cache, count):
        if cache is None or not getattr(cache, "layers", None):
            raise RuntimeError("model returned an unsupported or empty cache")
        rows = [self._new_cache() for _ in range(count)]
        for index, layer in enumerate(cache.layers):
            keys, values = layer.keys, layer.values
            if keys.shape[0] != count or values.shape[0] != count:
                raise RuntimeError("cache batch dimension mismatch")
            for row, target in enumerate(rows):
                target.update(keys[row:row + 1].clone(), values[row:row + 1].clone(), index)
        return rows

    def _merge(self, caches):
        lengths = [cache.layers[0].keys.shape[-2] for cache in caches]
        width = max(lengths)
        merged = self._new_cache()
        layer_count = len(caches[0].layers)
        if any(len(c.layers) != layer_count for c in caches):
            raise RuntimeError("cache layer counts disagree")
        for index in range(layer_count):
            tensors = []
            for name in ("keys", "values"):
                padded = [torch.nn.functional.pad(getattr(c.layers[index], name),
                          (0, 0, width - length, 0)) for c, length in zip(caches, lengths)]
                tensors.append(torch.cat(padded, dim=0))
            merged.update(tensors[0], tensors[1], index)
        return merged, width

    def _forward(self, ids, mask, cache):
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        with torch.inference_mode():
            return self.loaded.model.forward(input_ids=ids, attention_mask=mask,
                position_ids=positions[:, -ids.shape[1]:], past_key_values=cache, use_cache=True)

    def release_request(self, request_id):
        session = self._sessions.pop(request_id, None)
        if session is not None:
            self.kv.release_request(request_id)
            session.context.request.kv_cache_ref = None

    def close(self):
        for request_id in tuple(self._sessions):
            self.release_request(request_id)

    def _prune(self):
        for request_id, session in tuple(self._sessions.items()):
            if session.context.request.is_terminal:
                self.release_request(request_id)

    @staticmethod
    def _error(context, phase, error):
        return ContinuousBatchStepResult(context.request_id, phase, error=error)

    def prefill_batch(self, requests):
        contexts = self._validate_batch(requests)
        self._prune()
        errors, good = {}, []
        for context in contexts:
            try:
                if context.request_id in self._sessions:
                    raise ValueError("request already prefilled")
                self.builder.build_prefill([context])
                good.append(context)
            except Exception as error:
                errors[context.request_id] = self._error(context, ExecutionPhase.PREFILL, error)
        if good:
            try:
                inputs = self.builder.build_prefill(good)
                output = self._forward(inputs.input_ids, inputs.attention_mask, self._new_cache())
                caches = self._split(output.past_key_values, len(good))
                for row, (context, cache) in enumerate(zip(good, caches)):
                    generator = None
                    if context.request.generation_config.seed is not None:
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(context.request.generation_config.seed)
                    self.kv.add_request_cache(context.request_id, cache)
                    self._sessions[context.request_id] = _BatchSession(
                        context, inputs.attention_mask[row:row + 1].clone(),
                        output.logits[row:row + 1, -1, :].clone(), generator)
                    context.request.kv_cache_ref = cache
            except Exception as error:
                for context in good:
                    self.release_request(context.request_id)
                    errors[context.request_id] = self._error(context, ExecutionPhase.PREFILL, error)
        return [errors.get(c.request_id) or ContinuousBatchStepResult(c.request_id, ExecutionPhase.PREFILL)
                for c in contexts]

    def decode_batch(self, requests):
        contexts = self._validate_batch(requests)
        self._prune()
        results, ready = {}, []
        for context in contexts:
            try:
                session = self._sessions.get(context.request_id)
                if session is None or session.context is not context:
                    raise ValueError("no matching prefill session")
                if context.decode_state.execution_phase is not ExecutionPhase.DECODE:
                    raise ValueError("request must be in DECODE phase")
                if not session.pending_first:
                    self.builder.build_decode([context])
                    ready.append(context)
            except Exception as error:
                session = self._sessions.get(context.request_id)
                if session is not None and session.context is context:
                    self.release_request(context.request_id)
                results[context.request_id] = self._error(context, ExecutionPhase.DECODE, error)
        if ready:
            try:
                inputs = self.builder.build_decode(ready)
                caches = self.kv.get_batch_kv(inputs.request_ids)
                merged, width = self._merge(caches)
                masks = [torch.nn.functional.pad(self._sessions[c.request_id].mask,
                         (width - self._sessions[c.request_id].mask.shape[1], 0)) for c in ready]
                mask = torch.cat([torch.cat(masks, dim=0), inputs.attention_mask], dim=1)
                output = self._forward(inputs.input_ids, mask, merged)
                updated = self._split(output.past_key_values, len(ready))
                for row, (context, cache) in enumerate(zip(ready, updated)):
                    self.kv.update_request_cache(context.request_id, cache)
                    context.request.kv_cache_ref = cache
                    session = self._sessions[context.request_id]
                    session.mask = mask[row:row + 1].clone()
                    session.logits = output.logits[row:row + 1, -1, :].clone()
            except Exception as error:
                for context in ready:
                    self.release_request(context.request_id)
                    results[context.request_id] = self._error(context, ExecutionPhase.DECODE, error)
        for context in contexts:
            if context.request_id in results:
                continue
            try:
                session = self._sessions[context.request_id]
                config = context.request.generation_config
                with torch.inference_mode():
                    token = int(sample(session.logits, config.temperature, config.top_p,
                                       config.top_k, generator=session.generator).item())
                session.pending_first = False
                session.emitted += 1
                eos = self.loaded.tokenizer.eos_token_id
                stop = token in eos if isinstance(eos, (tuple, list)) else token == eos
                if config.stop:
                    text = self.loaded.tokenizer.decode(context.request.generated_tokens + [token], skip_special_tokens=True)
                    stop = stop or any(value in text for value in config.stop)
                finished = stop or session.emitted >= config.max_tokens
                results[context.request_id] = ContinuousBatchStepResult(context.request_id,
                    ExecutionPhase.DECODE, token_id=token, finished=finished,
                    finish_reason=FinishReason.STOP if stop else FinishReason.LENGTH if finished else None)
                if finished:
                    self.release_request(context.request_id)
            except Exception as error:
                self.release_request(context.request_id)
                results[context.request_id] = self._error(context, ExecutionPhase.DECODE, error)
        return [results[c.request_id] for c in contexts]
