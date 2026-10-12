"""Sampling, incremental decoding, and the single-request token loop."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from .cache import KVCacheManager
from .config import GenerationConfig
from .protocol import WorkerProtocol
from .request import RequestState, RequestStatus


@dataclass
class GenerationResult:
    text: str
    finish_reason: str  # "stop" | "length"
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GenerationChunk:
    """One streaming delta; the final chunk contains usage and finish_reason."""

    text: str
    finish_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GenerationMetrics:
    """Per-request model step timings used by the benchmark tools."""

    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    decode_steps: int = 0


def sample(logits, temperature: float = 0.7, top_p: float = 1.0, top_k: int = -1, generator=None):
    """Apply temperature, top-k, and top-p sampling to the last logits dimension."""
    import torch

    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        topk_values, topk_indices = torch.topk(logits, k, dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter(-1, topk_indices, topk_values)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        remove_mask = (cumsum - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        logits = logits.scatter(-1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x2F800 <= codepoint <= 0x2FA1F
    )


class _IncrementalDecoder:
    """Stable text flush policy compatible with Transformers TextStreamer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_cache: list[int] = []
        self.print_len = 0

    def push(self, token_id: int) -> str:
        self.token_cache.append(token_id)
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
        if text.endswith("\n"):
            printable = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        elif text and _is_cjk(text[-1]):
            printable = text[self.print_len :]
            self.print_len += len(printable)
        else:
            printable = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable)
        return printable


def _stop_prefix_length(text: str, stop: Optional[list[str]]) -> int:
    if not stop:
        return 0
    return max(
        (
            size
            for value in stop
            for size in range(1, min(len(value), len(text)) + 1)
            if text.endswith(value[:size])
        ),
        default=0,
    )


class GenerationRunner:
    """Runs the handwritten autoregressive loop for one isolated request."""

    def __init__(self, worker: WorkerProtocol, cache_manager: KVCacheManager):
        self.worker = worker
        self.cache_manager = cache_manager

    def stream(
        self,
        *,
        request: RequestState,
        tokenizer: Any,
        input_ids,
        attention_mask,
        max_model_len: int,
        cancel_event: Optional[threading.Event],
        metrics: Optional[GenerationMetrics],
        use_kv_cache: bool,
        cache_factory: Callable[[], Any],
    ) -> Iterator[GenerationChunk]:
        import torch

        config: GenerationConfig = request.generation_config
        prompt_len = int(input_ids.shape[1])
        request_generator = None
        if config.seed is not None:
            request_generator = torch.Generator(device=input_ids.device)
            request_generator.manual_seed(config.seed)

        eos_token_id = tokenizer.eos_token_id
        generated_ids = request.generated_tokens
        emitted_text = ""
        decoded_text = ""
        pending_text = ""
        incremental_decoder = _IncrementalDecoder(tokenizer)
        if cancel_event is not None and cancel_event.is_set():
            request.mark_cancelled()
            return

        request.mark_running()
        model_input_ids = input_ids
        outputs = None
        try:
            with self.cache_manager.request_cache(
                enabled=use_kv_cache,
                factory=cache_factory,
            ) as request_cache:
                request.kv_cache_ref = request_cache.value
                with torch.inference_mode():
                    for step in range(config.max_tokens):
                        if cancel_event is not None and cancel_event.is_set():
                            request.mark_cancelled()
                            return

                        position_ids = attention_mask.long().cumsum(dim=-1) - 1
                        position_ids.masked_fill_(attention_mask == 0, 0)
                        position_ids = position_ids[:, -model_input_ids.shape[1] :]

                        step_started = time.perf_counter()
                        outputs = self.worker.forward(
                            input_ids=model_input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=request_cache.value,
                            use_cache=use_kv_cache,
                        )
                        next_id = int(
                            sample(
                                outputs.logits[:, -1, :],
                                config.temperature,
                                config.top_p,
                                config.top_k,
                                generator=request_generator,
                            ).item()
                        )
                        step_seconds = time.perf_counter() - step_started
                        if metrics is not None:
                            if step == 0:
                                metrics.prefill_seconds = step_seconds
                            else:
                                metrics.decode_seconds += step_seconds
                                metrics.decode_steps += 1

                        request_cache.update(outputs.past_key_values)
                        request.kv_cache_ref = request_cache.value
                        request.append_token(next_id)
                        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                        stop_index: Optional[int] = None
                        if config.stop:
                            positions = [decoded_text.find(value) for value in config.stop]
                            positions = [position for position in positions if position >= 0]
                            if positions:
                                stop_index = min(positions)

                        if stop_index is not None or next_id == eos_token_id:
                            visible_text = decoded_text if stop_index is None else decoded_text[:stop_index]
                            if not visible_text.startswith(emitted_text):
                                raise RuntimeError("tokenizer 已改写发送过的文本，无法安全完成流式输出")
                            delta = visible_text[len(emitted_text) :]
                            if delta:
                                yield GenerationChunk(delta, None, prompt_len, len(generated_ids))
                            request.mark_finished()
                            yield GenerationChunk("", "stop", prompt_len, len(generated_ids))
                            return

                        pending_text += incremental_decoder.push(next_id)
                        hold_back = _stop_prefix_length(pending_text, config.stop)
                        printable_text = pending_text[:-hold_back] if hold_back else pending_text
                        pending_text = pending_text[-hold_back:] if hold_back else ""
                        if printable_text:
                            emitted_text += printable_text
                            yield GenerationChunk(printable_text, None, prompt_len, len(generated_ids))

                        next_tensor = torch.tensor([[next_id]], device=self.worker.device)
                        attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.ones(
                                    (1, 1),
                                    dtype=attention_mask.dtype,
                                    device=self.worker.device,
                                ),
                            ],
                            dim=1,
                        )
                        if use_kv_cache:
                            model_input_ids = next_tensor
                        else:
                            input_ids = torch.cat([input_ids, next_tensor], dim=1)
                            model_input_ids = input_ids
                        if attention_mask.shape[1] >= max_model_len:
                            break

                if not decoded_text.startswith(emitted_text):
                    raise RuntimeError("tokenizer 解码结果不是单调追加，无法安全完成输出")
                remaining = decoded_text[len(emitted_text) :]
                if remaining:
                    yield GenerationChunk(remaining, None, prompt_len, len(generated_ids))
                request.mark_finished()
                yield GenerationChunk("", "length", prompt_len, len(generated_ids))
        except GeneratorExit:
            if not request.is_terminal:
                request.mark_cancelled()
            raise
        except Exception:
            if request.status in {RequestStatus.WAITING, RequestStatus.RUNNING}:
                request.mark_failed()
            raise
        finally:
            # Drop all request-owned references on normal, error, cancel, and close paths.
            outputs = None
            request.kv_cache_ref = None
