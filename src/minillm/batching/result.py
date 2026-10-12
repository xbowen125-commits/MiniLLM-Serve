"""Model-independent result emitted for one request in a batch step."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import ExecutionPhase, FinishReason
from .errors import RequestExecutionError


@dataclass(frozen=True, slots=True)
class BatchStepResult:
    """Describe one request's incremental outcome from a batch step."""

    request_id: str
    token_id: int | None = None
    finished: bool = False
    finish_reason: FinishReason | str | None = None
    error: RequestExecutionError | str | None = None
    phase: ExecutionPhase | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.finish_reason is not None and not self.finished:
            raise ValueError("finish_reason requires finished=True")
        if self.finish_reason is not None:
            value = (
                self.finish_reason.value
                if isinstance(self.finish_reason, FinishReason)
                else self.finish_reason
            )
            if value not in {reason.value for reason in FinishReason}:
                raise ValueError(f"unsupported finish_reason: {value}")

    @property
    def is_terminal(self) -> bool:
        return self.finished or self.error is not None

    @property
    def finish_reason_value(self) -> str | None:
        if isinstance(self.finish_reason, FinishReason):
            return self.finish_reason.value
        return self.finish_reason
