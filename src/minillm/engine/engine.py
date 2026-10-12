"""Public orchestration facade for model loading, chat, and generation."""

from __future__ import annotations

import threading
from typing import Any, Iterator, Optional, Union

from ..config import Settings
from ..schemas import ChatMessage
from .cache import KVCacheManager
from .config import GenerationConfig
from .generation import GenerationChunk, GenerationMetrics, GenerationResult, GenerationRunner
from .request import RequestState
from .worker import ModelWorker


class GenerationEngine:
    """Coordinates prompting and generation while hiding worker/cache details."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.worker = ModelWorker(settings)
        self.cache_manager = KVCacheManager()
        self.generation_runner = GenerationRunner(self.worker, self.cache_manager)

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    @property
    def is_loaded(self) -> bool:
        return self.worker.is_loaded

    # These compatibility properties keep existing integrations and lightweight
    # test doubles working while the worker owns the actual runtime state.
    @property
    def tokenizer(self) -> Any:
        return self.worker.tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Any) -> None:
        self.worker.tokenizer = value

    @property
    def model(self) -> Any:
        return self.worker.model

    @model.setter
    def model(self, value: Any) -> None:
        self.worker.model = value

    @property
    def device(self) -> str:
        return self.worker.device

    @device.setter
    def device(self, value: str) -> None:
        self.worker.device = value

    @property
    def dtype(self) -> Any:
        return self.worker.dtype

    @dtype.setter
    def dtype(self, value: Any) -> None:
        self.worker.dtype = value

    def load(self) -> None:
        self.worker.load()

    def _create_kv_cache(self):
        """Compatibility hook used by tests and custom local integrations."""
        if self.model is None:
            raise RuntimeError("engine 未加载，请先调用 load()")
        return self.cache_manager.create(self.model.config)

    def build_prompt(self, messages: list[Union[ChatMessage, dict]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("engine 未加载，请先调用 load()")
        if self.tokenizer.chat_template is None:
            raise RuntimeError(f"模型 {self.settings.model_name} 没有 chat_template，无法处理多轮消息")
        normalized = [message if isinstance(message, dict) else message.model_dump() for message in messages]
        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _generation_config(
        *,
        config: Optional[GenerationConfig],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop: Optional[list[str]],
        stream: bool,
        seed: Optional[int],
    ) -> GenerationConfig:
        if config is not None:
            return config
        return GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop=stop,
            stream=stream,
            seed=seed,
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
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        generation_config = self._generation_config(
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            stream=False,
            seed=seed,
        )
        return self.generate(
            self.build_prompt(messages),
            cancel_event=cancel_event,
            config=generation_config,
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
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[GenerationChunk]:
        generation_config = self._generation_config(
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            stream=True,
            seed=seed,
        )
        return self.stream_generate(
            self.build_prompt(messages),
            cancel_event=cancel_event,
            config=generation_config,
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
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        generation_config = self._generation_config(
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            stream=False,
            seed=seed,
        )
        parts: list[str] = []
        final: Optional[GenerationChunk] = None
        for chunk in self.stream_generate(
            prompt,
            cancel_event=cancel_event,
            config=generation_config,
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
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[GenerationChunk]:
        if not self.is_loaded or self.tokenizer is None:
            raise RuntimeError("engine 未加载，请先调用 load()")

        generation_config = self._generation_config(
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            stream=True,
            seed=seed,
        )
        request = RequestState(prompt=prompt, generation_config=generation_config)

        try:
            encoded = self.tokenizer(request.prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            prompt_len = int(input_ids.shape[1])
            if prompt_len >= self.settings.max_model_len:
                raise ValueError(
                    f"prompt 长度 {prompt_len} >= max_model_len={self.settings.max_model_len}"
                )
        except Exception:
            request.mark_failed()
            raise

        return self.generation_runner.stream(
            request=request,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_model_len=self.settings.max_model_len,
            cancel_event=cancel_event,
            metrics=metrics,
            use_kv_cache=self.settings.use_kv_cache,
            cache_factory=self._create_kv_cache,
        )
