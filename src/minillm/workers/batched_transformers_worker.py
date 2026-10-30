"""Experimental device-aware batched forward backend, separate from serving."""

import torch

from ..memory.slot_kv_pool import SlotKVPool
from .model_loader import LoadedModel
from .transformers_batch_worker import TransformersBatchWorker


class BatchedTransformersWorker(TransformersBatchWorker):
    """Reuse tested row split/merge semantics on CPU or CUDA (batch <= 2).

    One forward per prefill group; one per ready decode group. First-token decode
    samples prefill logits without a redundant forward. Slots own independent
    cloned DynamicCache rows. This copies/left-pads KV tensors each step: it is
    not PagedAttention or an optimized production continuous batching backend.
    Only full-attention cache layers exposing keys/values/update are supported.
    A caller owns cleanup and must serialize worker calls.
    """

    max_batch_size = 2

    def __init__(self, loaded: LoadedModel, *, slot_capacity=4, cache_factory=None):
        device = torch.device(loaded.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("only CPU/CUDA supported")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        # Reuse CPU scaffolding without modifying that worker's device contract.
        super().__init__(LoadedModel(loaded.model, loaded.tokenizer, "cpu"), cache_factory=cache_factory)
        self.loaded = loaded
        self.device = device
        self.pool = SlotKVPool(slot_capacity)
        self.kv = self.pool

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward(self, ids, mask, cache):
        return super()._forward(ids.to(self.device), mask.to(self.device), cache)

    def prefill_batch(self, requests):
        from dataclasses import replace
        results = super().prefill_batch(requests)
        counts = {}
        for context in requests:
            session = self._sessions.get(context.request_id)
            if session is not None and session.context is context:
                counts[context.request_id] = int(session.mask.sum().item())
                seed = context.request.generation_config.seed
                if seed is not None:
                    session.generator = torch.Generator(device=self.device).manual_seed(seed)
        return [replace(r, prompt_tokens=counts.get(r.request_id, 0)) for r in results]

    def release_request(self, request_id):
        self.synchronize()
        # Superclass verifies pool absence and the original request reference;
        # failed attempts retain a cleanup record even after session removal.
        super().release_request(request_id)
