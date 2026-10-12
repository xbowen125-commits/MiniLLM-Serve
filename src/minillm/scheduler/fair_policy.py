"""Optional phase-aware batch selection with age-based fairness."""

import time
from typing import Callable, Mapping, Sequence

from ..engine import RequestState
from ..execution import ExecutionPhase
from .phase_policy import PhaseAwareBatchPolicy, PhaseState
from .priority import PhasePriority


class FairPhaseBatchPolicy(PhaseAwareBatchPolicy):
    """Rank by phase plus aging, then creation time and stable input order."""

    def __init__(self, states: Mapping[str, PhaseState] | None = None, *,
                 aging_factor: float = 1.0,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__(states)
        self.priority = PhasePriority(self, aging_factor=aging_factor, clock=clock)

    def select_batch(self, requests: Sequence[RequestState], max_size: int) -> list[RequestState]:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        now = self.priority.clock()
        active = [r for r in requests if self.phase_for(r) is not ExecutionPhase.FINISHED]
        return sorted(active, key=lambda r: (
            -self.priority.score(r, now=now), r.created_time,
        ))[:max_size]
