"""Prompt-free phase counts for local runtime observation."""

from dataclasses import dataclass
import math
import time
from typing import Sequence

from ..engine import RequestState
from .phase_policy import PhaseSchedulingPolicy


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    waiting_count: int
    prefill_count: int
    decode_count: int
    finished_count: int
    active_batch_size: int
    oldest_wait_ms: float = 0.0
    starvation_candidates: tuple[str, ...] = ()

    @classmethod
    def from_requests(
        cls,
        requests: Sequence[RequestState],
        policy: PhaseSchedulingPolicy,
        *,
        active_batch_size: int = 0,
        now: float | None = None,
        starvation_threshold_seconds: float = 60.0,
        waiting_requests: Sequence[RequestState] | None = None,
    ) -> "SchedulerSnapshot":
        if active_batch_size < 0:
            raise ValueError("active_batch_size must be non-negative")
        if not math.isfinite(starvation_threshold_seconds) or starvation_threshold_seconds < 0:
            raise ValueError("starvation threshold must be finite and non-negative")
        groups = policy.classify(requests)
        timestamp = time.time() if now is None else now
        candidates = requests if waiting_requests is None else waiting_requests
        pending = [r for r in candidates if policy.phase_for(r).value != "finished"]
        ages = [(r, max(0.0, timestamp - r.created_time)) for r in pending]
        return cls(
            waiting_count=len(groups["waiting"]),
            prefill_count=len(groups["prefill"]),
            decode_count=len(groups["decode"]),
            finished_count=len(groups["finished"]),
            active_batch_size=active_batch_size,
            oldest_wait_ms=max((age for _, age in ages), default=0.0) * 1000,
            starvation_candidates=tuple(
                r.request_id for r, age in ages if age > starvation_threshold_seconds
            ),
        )
