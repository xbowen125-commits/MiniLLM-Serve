"""Deterministic offline FIFO versus fair-phase admission simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from types import SimpleNamespace

from minillm.engine import RequestState
from minillm.contracts import ExecutionPhase
from minillm.scheduler.fair_policy import FairPhaseBatchPolicy
from .metrics import queue_percentile


@dataclass(frozen=True, slots=True)
class SimulatedRequest:
    request_id: str
    phase: ExecutionPhase
    arrival_time: float
    service_seconds: float = 1.0

    def __post_init__(self):
        if not self.request_id or self.phase is ExecutionPhase.FINISHED:
            raise ValueError("request requires an ID and active phase")
        if not math.isfinite(self.arrival_time) or self.arrival_time < 0:
            raise ValueError("arrival_time must be finite and non-negative")
        if not math.isfinite(self.service_seconds) or self.service_seconds <= 0:
            raise ValueError("service_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class SchedulerComparisonResult:
    policy: str
    completed_requests: int
    average_queue_delay: float
    max_queue_delay: float
    starvation_count: int
    selected_order: tuple[str, ...]
    queue_delays: tuple[tuple[str, float], ...]
    p50_queue_delay: float = 0.0
    p95_queue_delay: float = 0.0
    completed_per_sim_second: float = 0.0
    service_count_distribution: dict[str, int] = field(default_factory=dict)
    simulation_duration: float = 0.0


def mixed_phase_requests() -> tuple[SimulatedRequest, ...]:
    return (
        SimulatedRequest("waiting-old", ExecutionPhase.WAITING, 0, 5),
        SimulatedRequest("prefill", ExecutionPhase.PREFILL, 90, 3),
        SimulatedRequest("decode", ExecutionPhase.DECODE, 99, 1),
    )


class SchedulerComparisonRunner:
    """Compare one-slot non-preemptive selections using a virtual seconds clock.

    Each selected request completes after its fixed synthetic service duration.
    This models admission fairness, not model phases, decode steps, or GPU work.
    Starvation is a threshold exceedance at selection, not permanent starvation.
    """

    def __init__(self, *, start_time: float = 100.0, aging_factor: float = 1.0,
                 starvation_threshold: float = 10.0):
        if not math.isfinite(start_time) or start_time < 0:
            raise ValueError("start_time must be finite and non-negative")
        if not math.isfinite(starvation_threshold) or starvation_threshold < 0:
            raise ValueError("starvation_threshold must be finite and non-negative")
        # Validate even when the workload is empty.
        FairPhaseBatchPolicy(aging_factor=aging_factor)
        self.start_time = start_time
        self.aging_factor = aging_factor
        self.starvation_threshold = starvation_threshold

    def run(self, requests=None) -> dict[str, SchedulerComparisonResult]:
        workload = tuple(mixed_phase_requests() if requests is None else requests)
        if len(workload) > 1000 or len({r.request_id for r in workload}) != len(workload):
            raise ValueError("workload must contain at most 1000 unique request IDs")
        return {name: self._simulate(workload, name) for name in ("FIFO", "FairPhaseBatchPolicy")}

    def _simulate(self, workload, policy_name):
        now = self.start_time
        pending = list(workload)
        delays = []
        contexts = {r.request_id: RequestState(request_id=r.request_id, created_time=r.arrival_time)
                    for r in workload}
        phases = {r.request_id: SimpleNamespace(request_id=r.request_id, phase=r.phase)
                  for r in workload}
        fair = FairPhaseBatchPolicy(phases, aging_factor=self.aging_factor, clock=lambda: now)
        while pending:
            ready = [r for r in pending if r.arrival_time <= now]
            if not ready:
                now = min(r.arrival_time for r in pending)
                continue
            if policy_name == "FIFO":
                chosen = min(ready, key=lambda r: r.arrival_time)
            else:
                selected = fair.select_batch([contexts[r.request_id] for r in ready], 1)[0]
                chosen = next(r for r in ready if r.request_id == selected.request_id)
            delays.append((chosen.request_id, now - chosen.arrival_time))
            now += chosen.service_seconds
            pending.remove(chosen)
        values = [delay for _, delay in delays]
        return SchedulerComparisonResult(
            policy_name, len(delays), sum(values) / len(values) if values else 0.0,
            max(values, default=0.0),
            sum(delay > self.starvation_threshold for delay in values),
            tuple(request_id for request_id, _ in delays), tuple(delays),
            queue_percentile(values, 50), queue_percentile(values, 95),
            len(delays) / (now - self.start_time) if now > self.start_time else 0.0,
            {request_id: 1 for request_id, _ in delays}, now - self.start_time,
        )


def main():
    runner = SchedulerComparisonRunner()
    print(json.dumps({
        "scope": "virtual-clock-admission-simulation",
        "delay_unit": "seconds",
        "start_time": runner.start_time,
        "aging_factor": runner.aging_factor,
        "starvation_threshold": runner.starvation_threshold,
        "results": {name: asdict(result) for name, result in runner.run().items()},
    }, indent=2))


if __name__ == "__main__":
    main()
