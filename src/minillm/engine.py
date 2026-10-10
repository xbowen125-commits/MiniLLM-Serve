"""推理引擎：模型加载 + 支持 KV Cache 的手写自回归生成循环。

KV Cache 开启时，首步 Prefill 处理完整 Prompt，后续 Decode 只输入新增 Token；
关闭时保留每一步重算完整序列的对照路径。
torch/transformers 均为惰性导入，未安装 torch 时不影响 schema/server 的单元测试。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from .config import Settings
from .schemas import ChatMessage

logger = logging.getLogger("minillm.engine")


@dataclass
class GenerationResult:
    text: str
    finish_reason: str  # "stop" | "length"
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GenerationChunk:
    """一次增量生成结果；最后一个 chunk 只携带 finish_reason 和统计信息。"""

    text: str
    finish_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GenerationMetrics:
    """单次生成的模型步耗时；供 Benchmark 使用，不进入 API 响应。"""

    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    decode_steps: int = 0


def sample(logits, temperature: float = 0.7, top_p: float = 1.0, top_k: int = -1, generator=None):
    """对 logits 最后一维做 温度 + top-k + top-p 采样，返回 token id 张量。

    temperature <= 0 时退化为贪心（argmax）。
    top_k = -1 表示不限制；top_p = 1.0 表示不做 nucleus 截断。
    """
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
    """采用 Transformers 5.17 TextStreamer 的稳定文本刷新策略。"""

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


class GenerationEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: str = "cpu"
        self.dtype: Any = None

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """加载 tokenizer 与模型（首次运行会自动下载权重）。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = self.settings.resolve_device()
        self.dtype = self.settings.resolve_dtype()

        logger.info("loading %s (device=%s, dtype=%s)", self.settings.model_name, self.device, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_name,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        logger.info("model loaded")

    def _create_kv_cache(self):
        """为单次请求创建 Transformers 5.x DynamicCache。"""
        from transformers import DynamicCache

        return DynamicCache(config=self.model.config)

    def build_prompt(self, messages: list[Union[ChatMessage, dict]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("engine 未加载，请先调用 load()")
        if self.tokenizer.chat_template is None:
            raise RuntimeError(f"模型 {self.settings.model_name} 没有 chat_template，无法处理多轮消息")
        normalized = [m if isinstance(m, dict) else m.model_dump() for m in messages]
        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )

    def chat(
        self,
        messages: list[Union[ChatMessage, dict]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> GenerationResult:
        prompt = self.build_prompt(messages)
        return self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            seed=seed,
            cancel_event=cancel_event,
        )

    def stream_chat(
        self,
        messages: list[Union[ChatMessage, dict]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[GenerationChunk]:
        """逐步生成聊天内容；客户端断开时可通过 cancel_event 尽快停止。"""
        prompt = self.build_prompt(messages)
        return self.stream_generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            seed=seed,
            cancel_event=cancel_event,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> GenerationResult:
        """执行完整生成；底层与流式接口共用同一条逐 token 生成路径。"""
        parts: list[str] = []
        final: Optional[GenerationChunk] = None
        for chunk in self.stream_generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            seed=seed,
            cancel_event=cancel_event,
        ):
            if chunk.text:
                parts.append(chunk.text)
            if chunk.finish_reason is not None:
                final = chunk

        if final is None:
            raise RuntimeError("生成意外中止，未返回结束原因")
        return GenerationResult(
            text="".join(parts),
            finish_reason=final.finish_reason,
            prompt_tokens=final.prompt_tokens,
            completion_tokens=final.completion_tokens,
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
        metrics: Optional[GenerationMetrics] = None,
    ) -> Iterator[GenerationChunk]:
        """手写自回归循环，每完成一个模型前向就产出可安全发送的文本。"""
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("engine 未加载，请先调用 load()")

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = int(input_ids.shape[1])
        if prompt_len >= self.settings.max_model_len:
            raise ValueError(f"prompt 长度 {prompt_len} >= max_model_len={self.settings.max_model_len}")

        # A request-local generator avoids seeded requests changing one another's
        # global RNG state when scheduler concurrency is configured above one.
        request_generator = None
        if seed is not None:
            request_generator = torch.Generator(device=input_ids.device)
            request_generator.manual_seed(seed)

        eos_token_id = self.tokenizer.eos_token_id
        generated_ids: list[int] = []
        emitted_text = ""
        decoded_text = ""
        pending_text = ""
        incremental_decoder = _IncrementalDecoder(self.tokenizer)
        if cancel_event is not None and cancel_event.is_set():
            return
        use_kv_cache = self.settings.use_kv_cache
        past_key_values = self._create_kv_cache() if use_kv_cache else None
        model_input_ids = input_ids
        outputs = None

        try:
            with torch.inference_mode():
                for step in range(max_tokens):
                    if cancel_event is not None and cancel_event.is_set():
                        return

                    # 对 batch=1 的无 padding 输入，位置就是 attention mask 的累计位置。
                    # Cache Decode 时只保留与新增 token 对应的最后一个位置。
                    position_ids = attention_mask.long().cumsum(dim=-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 0)
                    position_ids = position_ids[:, -model_input_ids.shape[1] :]

                    step_started = time.perf_counter()
                    outputs = self.model(
                        input_ids=model_input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_kv_cache,
                    )
                    next_id = int(
                        sample(
                            outputs.logits[:, -1, :],
                            temperature,
                            top_p,
                            top_k,
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

                    if use_kv_cache:
                        past_key_values = outputs.past_key_values
                    generated_ids.append(next_id)
                    decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                    stop_index: Optional[int] = None
                    if stop:
                        positions = [decoded_text.find(value) for value in stop]
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
                        yield GenerationChunk("", "stop", prompt_len, len(generated_ids))
                        return

                    pending_text += incremental_decoder.push(next_id)
                    hold_back = _stop_prefix_length(pending_text, stop)
                    printable_text = pending_text[:-hold_back] if hold_back else pending_text
                    pending_text = pending_text[-hold_back:] if hold_back else ""
                    if printable_text:
                        emitted_text += printable_text
                        yield GenerationChunk(printable_text, None, prompt_len, len(generated_ids))

                    next_tensor = torch.tensor([[next_id]], device=self.device)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)],
                        dim=1,
                    )
                    if use_kv_cache:
                        model_input_ids = next_tensor
                    else:
                        input_ids = torch.cat([input_ids, next_tensor], dim=1)
                        model_input_ids = input_ids
                    if attention_mask.shape[1] >= self.settings.max_model_len:
                        break

            # 因长度结束时，不再需要为潜在 stop 前缀保留文本。
            if not decoded_text.startswith(emitted_text):
                raise RuntimeError("tokenizer 解码结果不是单调追加，无法安全完成输出")
            remaining = decoded_text[len(emitted_text) :]
            if remaining:
                yield GenerationChunk(remaining, None, prompt_len, len(generated_ids))
            yield GenerationChunk("", "length", prompt_len, len(generated_ids))
        finally:
            # Cache 必须是请求局部状态；正常结束、异常和 generator.close() 都释放引用。
            outputs = None
            past_key_values = None
