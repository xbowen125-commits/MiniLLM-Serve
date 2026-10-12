"""Phase weights with request-age bonuses for offline admission selection."""

import math
import time
from typing import Callable

from ..engine import RequestState
from ..execution import ExecutionPhase
from .phase_policy import PhaseSchedulingPolicy


class PhasePriority:
    """Score phase weight plus elapsed seconds since request creation.

    Creation age approximates waiting time in this runtime abstraction; it does
    not measure GPU execution or time since the last decode token.
    """

    WEIGHTS = {
        ExecutionPhase.DECODE: 100,
        ExecutionPhase.PREFILL: 50,
        ExecutionPhase.WAITING: 10,
    }

    def __init__(self, policy: PhaseSchedulingPolicy, *, aging_factor: float = 1.0,
                 clock: Callable[[], float] = time.time) -> None:
        if not math.isfinite(aging_factor) or aging_factor < 0:
            raise ValueError("aging_factor must be finite and non-negative")
        self.policy = policy
        self.aging_factor = aging_factor
        self.clock = clock

    def score(self, request_state: RequestState, *, now: float | None = None) -> float:
        phase = self.policy.phase_for(request_state)
        if phase is ExecutionPhase.FINISHED:
            return float("-inf")
        timestamp = self.clock() if now is None else now
        age = max(0.0, timestamp - request_state.created_time)
        return self.WEIGHTS[phase] + age * self.aging_factor
