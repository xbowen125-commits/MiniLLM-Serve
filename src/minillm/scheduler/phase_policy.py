"""Pure phase classification and selection for offline runtime requests."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from ..engine import RequestState
from ..execution import ExecutionPhase


class PhaseState(Protocol):
    request_id: str
    phase: ExecutionPhase


class PhaseSchedulingPolicy:
    """Resolve phases from runtime state without adding fields to RequestState.

    Requests without registered runtime state are waiting. Terminal request
    status takes precedence over registered execution phase.
    """

    def __init__(self, states: Mapping[str, PhaseState] | None = None) -> None:
        self.states = states if states is not None else {}

    def phase_for(self, request: RequestState) -> ExecutionPhase:
        if request.is_terminal:
            return ExecutionPhase.FINISHED
        state = self.states.get(request.request_id)
        if state is None:
            return ExecutionPhase.WAITING
        if state.request_id != request.request_id:
            raise ValueError("runtime state request_id mismatch")
        return state.phase

    def classify(self, requests: Sequence[RequestState]) -> dict[str, list[RequestState]]:
        groups = {phase.value: [] for phase in ExecutionPhase}
        for request in requests:
            groups[self.phase_for(request).value].append(request)
        return groups


class PhaseAwareBatchPolicy(PhaseSchedulingPolicy):
    """Prefer decode, then prefill, then waiting; preserve FIFO within phases."""

    def select_batch(self, requests: Sequence[RequestState], max_size: int) -> list[RequestState]:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        groups = self.classify(requests)
        ordered: list[RequestState] = []
        for phase in (ExecutionPhase.DECODE, ExecutionPhase.PREFILL, ExecutionPhase.WAITING):
            ordered.extend(sorted(groups[phase.value], key=lambda request: request.created_time))
        return ordered[:max_size]
