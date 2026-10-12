"""Request-level decode runtime state and model-independent metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from ..contracts import DecodeStatus, ExecutionPhase, FinishReason


_ALLOWED_PHASE_TRANSITIONS = {
    ExecutionPhase.WAITING: {ExecutionPhase.PREFILL, ExecutionPhase.FINISHED},
    ExecutionPhase.PREFILL: {ExecutionPhase.DECODE, ExecutionPhase.FINISHED},
    ExecutionPhase.DECODE: {ExecutionPhase.FINISHED},
}


@dataclass
class DecodeState:
    """Runtime lifecycle and token accounting for one decode request."""

    request_id: str
    max_tokens: int
    generated_tokens: list[int] = field(default_factory=list)
    num_generated_tokens: int = 0
    status: DecodeStatus = DecodeStatus.ACTIVE
    finish_reason: FinishReason | None = None
    created_time: float = field(default_factory=time.time)
    started_time: float | None = None
    finished_time: float | None = None
    first_token_time: float | None = None
    error: str | None = None
    decode_steps: int = 0
    no_progress_steps: int = 0
    max_no_progress_steps: int = 10
    phase: ExecutionPhase = ExecutionPhase.WAITING
    phase_history: list[ExecutionPhase] = field(
        default_factory=lambda: [ExecutionPhase.WAITING]
    )
    phase_updated_time: float | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.num_generated_tokens != len(self.generated_tokens):
            raise ValueError("num_generated_tokens must match generated_tokens")
        if self.max_no_progress_steps < 1:
            raise ValueError("max_no_progress_steps must be at least 1")
        if not self.phase_history or self.phase_history[-1] is not self.phase:
            raise ValueError("phase_history must end with the current phase")
        if self.phase_updated_time is None:
            self.phase_updated_time = self.created_time

    @property
    def is_terminal(self) -> bool:
        return self.status is not DecodeStatus.ACTIVE

    def can_prefill(self) -> bool:
        """Return whether this active request may execute prompt prefill."""
        return not self.is_terminal and self.phase is ExecutionPhase.WAITING

    def can_decode(self) -> bool:
        """Return whether this active request has entered or completed prefill."""
        return not self.is_terminal and self.phase in {
            ExecutionPhase.PREFILL,
            ExecutionPhase.DECODE,
        }

    def start(self, *, now: float | None = None) -> None:
        if self.is_terminal:
            raise RuntimeError("cannot start a terminal decode state")
        if self.started_time is None:
            self.started_time = time.time() if now is None else now

    def record_step(self) -> None:
        if self.is_terminal:
            raise RuntimeError("terminal decode state cannot record another step")
        self.decode_steps += 1

    def transition_phase(
        self,
        target: ExecutionPhase,
        *,
        now: float | None = None,
    ) -> None:
        """Validate and record an execution-phase transition."""
        if target is self.phase:
            return
        allowed = _ALLOWED_PHASE_TRANSITIONS.get(self.phase, set())
        if target not in allowed:
            raise RuntimeError(
                f"invalid execution phase transition: "
                f"{self.phase.value} -> {target.value}"
            )
        self.phase = target
        self.phase_history.append(target)
        self.phase_updated_time = time.time() if now is None else now

    def append_token(self, token_id: int, *, now: float | None = None) -> None:
        """Append one token and enforce the request-local length limit."""
        if self.is_terminal:
            raise RuntimeError("terminal decode state cannot append tokens")
        timestamp = time.time() if now is None else now
        if self.started_time is None:
            self.started_time = timestamp
        if self.first_token_time is None:
            self.first_token_time = timestamp
        if self.phase is ExecutionPhase.PREFILL:
            self.transition_phase(ExecutionPhase.DECODE, now=timestamp)
        self.generated_tokens.append(token_id)
        self.num_generated_tokens += 1
        self.no_progress_steps = 0
        if self.num_generated_tokens >= self.max_tokens:
            self.finish(FinishReason.LENGTH, now=timestamp)

    def record_no_progress(self, *, now: float | None = None) -> None:
        """Fail only this request after repeated steps without useful output."""
        if self.is_terminal:
            raise RuntimeError("terminal decode state cannot record no progress")
        self.no_progress_steps += 1
        if self.no_progress_steps >= self.max_no_progress_steps:
            self.fail("decode made no progress", now=now)

    def finish(
        self,
        reason: FinishReason = FinishReason.STOP,
        *,
        now: float | None = None,
    ) -> None:
        if reason not in {FinishReason.STOP, FinishReason.LENGTH}:
            raise ValueError("finished decode state requires stop or length reason")
        self._transition(DecodeStatus.FINISHED, reason, now=now)

    def fail(self, error: str, *, now: float | None = None) -> None:
        if not error:
            raise ValueError("error must not be empty")
        self._transition(DecodeStatus.FAILED, FinishReason.ERROR, now=now)
        self.error = error

    def cancel(self, *, now: float | None = None) -> None:
        self._transition(DecodeStatus.CANCELLED, FinishReason.CANCELLED, now=now)

    def _transition(
        self,
        status: DecodeStatus,
        reason: FinishReason,
        *,
        now: float | None,
    ) -> None:
        if self.is_terminal:
            raise RuntimeError(
                f"invalid decode transition: {self.status.value} -> {status.value}"
            )
        self.status = status
        self.finish_reason = reason
        timestamp = time.time() if now is None else now
        self.finished_time = timestamp
        self.transition_phase(ExecutionPhase.FINISHED, now=timestamp)


@dataclass(frozen=True, slots=True)
class DecodeMetrics:
    """Request-level decode latency and token metrics, independent from HTTP."""

    first_token_latency_ms: float | None
    total_latency_ms: float | None
    generated_tokens: int
    decode_steps: int
    tpot_ms: float | None

    @classmethod
    def from_state(cls, state: DecodeState) -> "DecodeMetrics":
        ttft = cls._duration_ms(state.started_time, state.first_token_time)
        total = cls._duration_ms(state.started_time, state.finished_time)
        decode_duration = cls._duration_ms(state.first_token_time, state.finished_time)
        tpot = (
            decode_duration / state.num_generated_tokens
            if decode_duration is not None and state.num_generated_tokens > 0
            else None
        )
        return cls(
            first_token_latency_ms=ttft,
            total_latency_ms=total,
            generated_tokens=state.num_generated_tokens,
            decode_steps=state.decode_steps,
            tpot_ms=tpot,
        )

    @staticmethod
    def _duration_ms(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return max(0.0, (end - start) * 1000)
