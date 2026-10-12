"""Reuse offline FIFO queue and phase policies with a virtual seconds clock."""

from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace

from minillm.engine import RequestState
from minillm.request_queue import RequestQueue
from minillm.scheduler.phase_policy import PhaseAwareBatchPolicy
from minillm.scheduler.fair_policy import FairPhaseBatchPolicy
from .metrics import summarize
from .scenario import ResearchScenario

POLICIES = ("FIFO", "Phase Priority", "Fair Aging")


class SchedulerResearchRunner:
    """Single-slot non-preemptive admission, not token-level scheduling.

    Phase labels remain fixed throughout admission. No fake timings are written
    to online state and no model is loaded. All selected requests complete.
    """

    def run_policy(self, scenario: ResearchScenario, policy: str) -> dict:
        if policy not in POLICIES:
            raise ValueError("unknown research policy")
        now = scenario.start_time
        pending = sorted(scenario.requests, key=lambda r: r.arrival_time)
        specs = {r.request_id: r for r in pending}
        states = {r.request_id: SimpleNamespace(request_id=r.request_id, phase=r.phase) for r in pending}
        fifo = RequestQueue(len(pending))
        ready = []
        phase_policy = PhaseAwareBatchPolicy(states)
        fair_policy = FairPhaseBatchPolicy(states, aging_factor=scenario.aging_factor, clock=lambda: now)
        rows = []
        while pending or ready:
            if not ready and pending:
                now = max(now, pending[0].arrival_time)
            while pending and pending[0].arrival_time <= now:
                spec = pending.pop(0)
                request = RequestState(request_id=spec.request_id, created_time=spec.arrival_time)
                ready.append(request)
                if policy == "FIFO" and not fifo.put(request):
                    raise RuntimeError("unexpected bounded FIFO overflow")
            if policy == "FIFO":
                selected = fifo.get_batch(1)[0]
            else:
                selector = phase_policy if policy == "Phase Priority" else fair_policy
                selected = selector.select_batch(ready, 1)[0]
            spec = specs[selected.request_id]
            wait = now - spec.arrival_time
            rows.append({"request_id": spec.request_id, "phase": spec.phase.value,
                         "arrival_time": spec.arrival_time, "selected_time": now,
                         "finished_time": now + spec.service_seconds,
                         "service_seconds": spec.service_seconds, "queue_delay": wait,
                         "ttft": wait + scenario.first_token_fraction * spec.service_seconds,
                         "total_latency": wait + spec.service_seconds})
            now += spec.service_seconds
            ready.remove(selected)
        return {"metrics": summarize(rows, scenario.starvation_threshold),
                "selected_order": [r["request_id"] for r in rows], "requests": rows,
                "simulation_duration": now - scenario.start_time, "idle_after_test": not ready and not pending}

    def run(self, scenario: ResearchScenario) -> dict:
        return {"schema_version": 1, "scope": "virtual-clock-offline-admission-research",
                "time_unit": "seconds", "timestamp": datetime.now(timezone.utc).isoformat(),
                "scenario": asdict(scenario),
                "results": {p: self.run_policy(scenario, p) for p in POLICIES}}
