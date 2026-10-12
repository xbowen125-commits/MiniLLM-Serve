"""Model-independent results for explicit prefill and decode execution."""

from __future__ import annotations

from dataclasses import dataclass
from .status import FinishReason


@dataclass(frozen=True, slots=True)
class PrefillStepResult:
    """Result of processing one request's prompt during a prefill step."""

    request_id: str
    prompt_tokens: int
    kv_cache_created: bool
    first_token_ready: bool
    error: Exception | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")


@dataclass(frozen=True, slots=True)
class DecodeStepResult:
    """Result of one autoregressive decode step for a single request."""

    request_id: str
    token_id: int | None = None
    finished: bool = False
    finish_reason: FinishReason | None = None
    error: Exception | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.finish_reason is not None and not self.finished:
            raise ValueError("finish_reason requires finished=True")
