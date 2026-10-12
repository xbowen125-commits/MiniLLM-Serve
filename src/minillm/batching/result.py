"""Model-independent result emitted for one request in a batch step."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchStepResult:
    """Describe one request's incremental outcome from a batch step."""

    request_id: str
    token_id: int | None = None
    finished: bool = False
    finish_reason: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.finish_reason is not None and not self.finished:
            raise ValueError("finish_reason requires finished=True")

    @property
    def is_terminal(self) -> bool:
        return self.finished or self.error is not None
