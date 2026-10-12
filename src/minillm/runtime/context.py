"""Single-owner request context for the offline runtime.

The legacy batching.DecodeState remains available for compatibility. Contexts
use execution-only DecodeState and never copy lifecycle or generated token data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..contracts import ExecutionPhase
from ..engine.request import RequestState, RequestStatus


@dataclass
class DecodeState:
    """Execution observations only; request lifecycle belongs to RequestState."""

    execution_phase: ExecutionPhase = ExecutionPhase.WAITING
    decode_steps: int = 0
    first_token_time: float | None = None
    last_token_time: float | None = None
    prefill_tokens: int = 0
    decode_tokens: int = 0

    def transition_phase(self, phase: ExecutionPhase) -> None:
        if phase is self.execution_phase:
            return
        allowed = {
            ExecutionPhase.WAITING: {ExecutionPhase.PREFILL, ExecutionPhase.FINISHED},
            ExecutionPhase.PREFILL: {ExecutionPhase.DECODE, ExecutionPhase.FINISHED},
            ExecutionPhase.DECODE: {ExecutionPhase.FINISHED},
        }
        if phase not in allowed.get(self.execution_phase, set()):
            raise RuntimeError("invalid execution phase transition")
        self.execution_phase = phase

    def record_prefill(self, token_count: int) -> None:
        if self.execution_phase is not ExecutionPhase.PREFILL:
            raise RuntimeError("prefill observations require PREFILL phase")
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        self.prefill_tokens += token_count

    def record_step(self) -> None:
        if self.execution_phase is not ExecutionPhase.DECODE:
            raise RuntimeError("decode steps require DECODE phase")
        self.decode_steps += 1

    def record_token(self, *, now: float) -> None:
        if self.execution_phase is not ExecutionPhase.DECODE:
            raise RuntimeError("token observations require DECODE phase")
        if self.last_token_time is not None and now < self.last_token_time:
            raise ValueError("token timestamps must be non-decreasing")
        if self.first_token_time is None:
            self.first_token_time = now
        self.last_token_time = now
        self.decode_tokens += 1


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    """Derived request metrics; no duplicate mutable counters or lifecycle."""

    request_state: RequestState = field(repr=False)
    decode_state: DecodeState

    @property
    def first_token_latency_ms(self) -> float | None:
        start = self.request_state.started_time
        first = self.decode_state.first_token_time
        return None if start is None or first is None else (first - start) * 1000

    @property
    def total_latency_ms(self) -> float | None:
        start = self.request_state.started_time
        end = self.request_state.finished_time
        return None if start is None or end is None else (end - start) * 1000


@dataclass
class RuntimeRequestContext:
    """Own one request and its execution-only observations.

    RequestState is the sole source for status, lifecycle timestamps, options,
    prompt, and token IDs. The execution state holds only phase and counters.
    """

    request_state: RequestState
    decode_state: DecodeState = field(default_factory=DecodeState)
    metrics: RuntimeMetrics = field(init=False)
    observations: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.metrics = RuntimeMetrics(self.request_state, self.decode_state)

    @property
    def request_id(self) -> str:
        return self.request_state.request_id

    @property
    def request(self) -> RequestState:
        """Compatibility-friendly name for the sole request state owner."""
        return self.request_state

    @property
    def status(self) -> RequestStatus:
        return self.request_state.status

    def start(self, *, now: float | None = None) -> None:
        self.request_state.mark_running(now=now)

    def append_token(self, token_id: int, *, now: float | None = None) -> None:
        if self.request_state.status is not RequestStatus.RUNNING:
            raise RuntimeError("tokens require a running request")
        timestamp = time.time() if now is None else now
        self.decode_state.record_token(now=timestamp)
        self.request_state.append_token(token_id)

    def finish(self, *, now: float | None = None) -> None:
        self.request_state.mark_finished(now=now)
        self.decode_state.transition_phase(ExecutionPhase.FINISHED)

    def fail(self, *, now: float | None = None) -> None:
        self.request_state.mark_failed(now=now)
        self.decode_state.transition_phase(ExecutionPhase.FINISHED)

    def cancel(self, *, now: float | None = None) -> None:
        self.request_state.mark_cancelled(now=now)
        self.decode_state.transition_phase(ExecutionPhase.FINISHED)
