"""Serial per-request causal LM adapter implementing the execution protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..contracts import PrefillStepResult, DecodeStepResult, FinishReason
from ..engine.request import RequestState
from ..engine.cache import KVCacheManager
from ..engine.generation import sample
from ..runtime.context import RuntimeRequestContext
from .model_loader import LoadedModel, ModelLoaderProtocol


@dataclass
class _Session:
    request: RequestState
    mask: Any
    cache: Any
    logits: Any
    generator: Any
    last_token: int | None = None
    emitted: int = 0


class TransformersWorker:
    """Experimental single-device worker; calls are sequential, not batched.

    Accepts Context or RequestState without mutating lifecycle or token lists.
    The coordinator applies results. Caller must release cancelled/abandoned
    sessions with release_request(), or close() when disposing this worker.
    """

    def __init__(self, loaded: LoadedModel | None = None, *,
                 loader: ModelLoaderProtocol | None = None,
                 cache_factory: Callable[[Any], Any] | None = None) -> None:
        if loaded is not None and loader is not None:
            raise ValueError("provide loaded model or loader, not both")
        self.loaded = loaded
        self.loader = loader
        self.cache_factory = cache_factory or KVCacheManager().create
        self._sessions: dict[str, _Session] = {}

    def load(self) -> None:
        if self.loaded is None:
            if self.loader is None:
                raise RuntimeError("no model loader configured")
            self.loaded = self.loader.load()

    @staticmethod
    def _request(item) -> RequestState:
        return item.request if isinstance(item, RuntimeRequestContext) else item

    def release_request(self, request_id: str) -> None:
        session = self._sessions.pop(request_id, None)
        if session is not None:
            session.request.kv_cache_ref = None

    def close(self) -> None:
        for request_id in tuple(self._sessions):
            self.release_request(request_id)

    def _prune(self) -> None:
        for request_id, session in tuple(self._sessions.items()):
            if session.request.is_terminal:
                self.release_request(request_id)

    def _forward(self, ids, mask, cache):
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        outputs = self.loaded.model(
            input_ids=ids, attention_mask=mask,
            position_ids=positions[:, -ids.shape[1]:],
            past_key_values=cache, use_cache=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("model did not return a KV cache")
        return outputs

    def prefill(self, requests: Sequence[RequestState | RuntimeRequestContext]) -> list[PrefillStepResult]:
        import torch
        self._prune()
        results = []
        for item in requests:
            request = self._request(item)
            try:
                if self.loaded is None:
                    raise RuntimeError("worker is not loaded; call load() explicitly")
                if request.is_terminal or request.request_id in self._sessions:
                    raise RuntimeError("request is terminal or already prefilled")
                encoded = self.loaded.tokenizer(request.prompt, return_tensors="pt")
                ids = encoded["input_ids"].to(self.loaded.device)
                mask = encoded.get("attention_mask", torch.ones_like(ids)).to(self.loaded.device)
                if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                    raise ValueError("prefill requires one non-empty prompt")
                cache = self.cache_factory(self.loaded.model.config)
                with torch.inference_mode():
                    output = self._forward(ids, mask, cache)
                generator = None
                if request.generation_config.seed is not None:
                    generator = torch.Generator(device=self.loaded.device)
                    generator.manual_seed(request.generation_config.seed)
                self._sessions[request.request_id] = _Session(
                    request, mask, output.past_key_values, output.logits[:, -1, :], generator,
                )
                request.kv_cache_ref = output.past_key_values
                results.append(PrefillStepResult(request.request_id, ids.shape[1], True, True))
            except Exception as error:
                # Duplicate prefill must not destroy an existing valid session.
                results.append(PrefillStepResult(request.request_id, 0, False, False, error))
        return results

    def decode(self, requests: Sequence[RequestState | RuntimeRequestContext]) -> list[DecodeStepResult]:
        import torch
        self._prune()
        results = []
        for item in requests:
            request = self._request(item)
            try:
                session = self._sessions.get(request.request_id)
                if session is None or session.request is not request:
                    raise RuntimeError("request has no matching prefill session")
                with torch.inference_mode():
                    if session.last_token is not None:
                        ids = torch.tensor([[session.last_token]], device=self.loaded.device)
                        session.mask = torch.cat([session.mask, torch.ones_like(ids)], dim=1)
                        output = self._forward(ids, session.mask, session.cache)
                        session.cache = output.past_key_values
                        request.kv_cache_ref = session.cache
                        session.logits = output.logits[:, -1, :]
                    config = request.generation_config
                    token = int(sample(session.logits, config.temperature, config.top_p,
                                       config.top_k, generator=session.generator).item())
                session.last_token = token
                session.emitted += 1
                eos = self.loaded.tokenizer.eos_token_id
                stopped = token in eos if isinstance(eos, (list, tuple)) else token == eos
                if config.stop:
                    text = self.loaded.tokenizer.decode(request.generated_tokens + [token], skip_special_tokens=True)
                    stopped = stopped or any(stop in text for stop in config.stop)
                finished = stopped or session.emitted >= config.max_tokens
                reason = FinishReason.STOP if stopped else FinishReason.LENGTH if finished else None
                results.append(DecodeStepResult(request.request_id, token, finished, reason))
                if finished:
                    self.release_request(request.request_id)
            except Exception as error:
                session = self._sessions.get(request.request_id)
                if session is not None and session.request is request:
                    self.release_request(request.request_id)
                results.append(DecodeStepResult(request.request_id, error=error))
        return results
