"""Bounded runtime benchmark with a deterministic, model-free worker."""

import argparse
from dataclasses import asdict
import json
import time

from minillm.contracts import DecodeStepResult, PrefillStepResult
from minillm.execution.protocol import ExecutionWorkerProtocol
from minillm.runtime import RuntimeFacade

from .metrics import BenchmarkResult, SchedulerDecision, measure_request
from .scenarios import Scenario, mixed_lengths, single_request
from .workload import generate_workload


class FakeWorker:
    """Emit one fixed token per request per decode; never load a model.

    Prompt token counts use whitespace words purely as a simulation convention.
    kv_cache_created=False explicitly reports that no real cache is allocated.
    """

    def prefill(self, requests):
        return [PrefillStepResult(r.request_id, len(r.prompt.split()), False, False)
                for r in requests]

    def decode(self, requests):
        return [DecodeStepResult(r.request_id, token_id=7) for r in requests]


class RuntimeBenchmarkRunner:
    def __init__(self, worker: ExecutionWorkerProtocol | None = None):
        self.worker = worker

    def run(self, scenario: Scenario) -> BenchmarkResult:
        worker = self.worker if self.worker is not None else FakeWorker()
        runtime = RuntimeFacade(worker, max_batch_size=scenario.max_batch_size)
        start = time.perf_counter()
        decisions = []
        contexts = []
        exhausted = False
        try:
            # Warm lazy imports on an empty runtime before submitting requests.
            runtime.step()
            runtime.snapshot()
            start = time.perf_counter()
            contexts = generate_workload(runtime, scenario)
            for cycle in range(1, scenario.max_steps + 1):
                if not runtime.snapshot().active_requests:
                    break
                phases = {c.request_id: c.decode_state.execution_phase.value for c in contexts}
                selected = runtime.step()
                if selected:
                    decisions.append(SchedulerDecision(
                        cycle, phases[selected[0]], selected,
                        (time.perf_counter() - start) * 1000,
                    ))
                    assert all(phases[rid] == phases[selected[0]] for rid in selected)
            exhausted = runtime.snapshot().active_requests > 0
        finally:
            # Step-budget exhaustion is represented as cancelled requests, not
            # fabricated completion. Unexpected worker exceptions propagate.
            runtime.close()
        return BenchmarkResult(
            asdict(scenario), "offline-runtime-simulation", type(worker).__name__,
            tuple(measure_request(c) for c in contexts), tuple(decisions),
            (time.perf_counter() - start) * 1000, exhausted,
            runtime.snapshot().active_requests,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("single", "mixed"), default="single")
    args = parser.parse_args()
    scenario = single_request() if args.scenario == "single" else mixed_lengths()
    print(json.dumps(RuntimeBenchmarkRunner().run(scenario).to_dict(), indent=2))


if __name__ == "__main__":
    main()
