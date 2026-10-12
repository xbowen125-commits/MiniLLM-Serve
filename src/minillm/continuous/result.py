"""Per-request results for model-independent continuous batch execution."""

from dataclasses import dataclass

from ..contracts import DecodeStepResult, ExecutionPhase, FinishReason


@dataclass(frozen=True, slots=True)
class ContinuousBatchStepResult:
    """Structured outcome; token_result remains a legacy compatibility view."""

    request_id: str
    phase: ExecutionPhase
    token_result: DecodeStepResult | None = None
    finished: bool = False
    error: Exception | str | None = None
    token_id: int | None = None
    finish_reason: FinishReason | None = None

    def __post_init__(self):
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.token_result is not None:
            if self.token_result.request_id != self.request_id:
                raise ValueError("token result request_id mismatch")
            object.__setattr__(self, "token_id", self.token_result.token_id)
            object.__setattr__(self, "finish_reason", self.token_result.finish_reason)
        elif self.phase is ExecutionPhase.DECODE:
            error = self.error if isinstance(self.error, Exception) else RuntimeError(self.error) if self.error is not None else None
            object.__setattr__(self, "token_result", DecodeStepResult(
                self.request_id, self.token_id, self.finished, self.finish_reason, error,
            ))
        if self.finish_reason is not None and not self.finished:
            raise ValueError("finish_reason requires finished=True")
