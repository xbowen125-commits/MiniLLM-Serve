"""Drive the existing offline Continuous BatchExecutor with a fake worker."""

from dataclasses import asdict
from datetime import datetime, timezone

from minillm.continuous import BatchExecutor, BatchScheduler
from minillm.contracts import PrefillStepResult, DecodeStepResult
from minillm.engine import RequestState, GenerationConfig
from minillm.runtime import RuntimeRequestContext
from .metrics import request_metrics, summarize
from .scenario import Scenario


class FakeWorker:
    """One logical phase call costs base + per_request * group_size seconds.

    No tensors or sleeps. Timings live in a separate experiment ledger, not
    runtime timestamps. Group cost is an assumption, not a measured speedup.
    """

    def __init__(self, scenario):
        self.scenario = scenario
        self.now = 0.0
        self.observations = {}
        self.trace = []

    def prefill(self, requests):
        start = self.now
        self.now += self.scenario.prefill_base_s + self.scenario.prefill_per_request_s * len(requests)
        for c in requests:
            self.observations[c.request_id] = {"started": start}
        self.trace.append({"phase": "prefill", "request_ids": [c.request_id for c in requests],
                           "start_s": start, "end_s": self.now, "batch_size": len(requests)})
        return [PrefillStepResult(c.request_id, 1, False, False) for c in requests]

    def decode(self, requests):
        start = self.now
        self.now += self.scenario.decode_base_s + self.scenario.decode_per_request_s * len(requests)
        for c in requests:
            row = self.observations[c.request_id]
            row.setdefault("first_token", self.now)
            if len(c.generated_tokens) + 1 == c.generation_config.max_tokens:
                row["finished"] = self.now
        self.trace.append({"phase": "decode", "request_ids": [c.request_id for c in requests],
                           "start_s": start, "end_s": self.now, "batch_size": len(requests)})
        return [DecodeStepResult(c.request_id, token_id=7) for c in requests]


def run_policy(scenario: Scenario, policy: str) -> dict:
    if policy not in {"sequential", "continuous_batch"}:
        raise ValueError("unknown policy")
    worker = FakeWorker(scenario)
    executor = BatchExecutor(worker, scheduler=BatchScheduler(
        max_active=1 if policy == "sequential" else scenario.max_active,
        max_waiting=len(scenario.requests)))
    pending = sorted(scenario.requests, key=lambda r: r.arrival_time)
    contexts = {}
    cycles = 0
    try:
        while pending or executor.batch.active_requests or executor.batch.waiting_requests:
            if not executor.batch.active_requests and not executor.batch.waiting_requests and pending:
                worker.now = max(worker.now, pending[0].arrival_time)
            while pending and pending[0].arrival_time <= worker.now:
                request = pending.pop(0)
                context = RuntimeRequestContext(RequestState(request_id=request.request_id,
                    prompt="simulation", created_time=request.arrival_time,
                    generation_config=GenerationConfig(max_tokens=request.max_tokens)))
                contexts[request.request_id] = context
                executor.add(context)
            executor.step()
            cycles += 1
            if cycles > sum(r.max_tokens for r in scenario.requests) + len(scenario.requests):
                raise RuntimeError("simulation exceeded progress bound")
        rows = []
        for request in scenario.requests:
            observations = worker.observations[request.request_id]
            rows.append({"request_id": request.request_id, **request_metrics(
                request.arrival_time, observations["started"], observations["first_token"],
                observations["finished"], len(contexts[request.request_id].request.generated_tokens))})
        sizes = [t["batch_size"] for t in worker.trace if t["phase"] == "decode"]
        elapsed = worker.now - min(r.arrival_time for r in scenario.requests)
        return {"metrics": summarize(rows, elapsed, sizes), "requests": rows,
                "step_trace": worker.trace, "idle_after_test": not (
                    executor.batch.active_requests or executor.batch.waiting_requests)}
    finally:
        executor.close()


def run_comparison(scenario: Scenario) -> dict:
    return {"schema_version": 1, "scope": "fake-worker-virtual-clock",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": asdict(scenario),
            "results": {p: run_policy(scenario, p) for p in ("sequential", "continuous_batch")}}
