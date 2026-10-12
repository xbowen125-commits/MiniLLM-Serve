"""Simulation workload, metric formulas, and bounded runner tests."""

import json

import pytest

from benchmarks.runtime.workload import generate_workload
from benchmarks.runtime.scenarios import Scenario, mixed_lengths
from benchmarks.runtime.metrics import measure_request
from benchmarks.runtime.runner import FakeWorker, RuntimeBenchmarkRunner
from minillm.runtime import RuntimeFacade


def test_workload_generates_registered_contexts():
    runtime = RuntimeFacade(FakeWorker())
    try:
        contexts = generate_workload(runtime, Scenario(token_limits=(2, 4)))
        assert [c.request_id for c in contexts] == ["request-0000", "request-0001"]
        assert [c.request.generation_config.max_tokens for c in contexts] == [2, 4]
        assert runtime.snapshot().active_requests == 2
    finally:
        runtime.close()


def test_metrics_use_distinct_queue_and_execution_time_origins():
    runtime = RuntimeFacade(FakeWorker())
    context = runtime.submit("synthetic")
    context.request.created_time = 1.0
    context.request.started_time = 1.5
    context.request.finished_time = 3.0
    context.decode_state.first_token_time = 2.0
    context.decode_state.decode_tokens = 2
    observed = measure_request(context)
    assert observed.queue_delay_ms == 500
    assert observed.ttft_ms == 500
    assert observed.total_latency_ms == 2000
    assert observed.decode_tokens == 2
    runtime.close()


def test_runner_executes_prefill_and_decode_cycles():
    result = RuntimeBenchmarkRunner().run(Scenario(token_limits=(2,)))
    assert result.scope == "offline-runtime-simulation"
    assert result.requests[0].status == "finished"
    assert result.requests[0].decode_tokens == 2
    assert result.requests[0].ttft_ms is not None
    assert [d.phase for d in result.scheduler_decisions] == ["waiting", "decode", "decode"]
    assert result.final_active_requests == 0
    assert result.step_budget_exhausted is False
    assert json.loads(json.dumps(result.to_dict()))["worker"] == "FakeWorker"


def test_multi_request_scenario_is_bounded_and_records_selection():
    result = RuntimeBenchmarkRunner().run(mixed_lengths())
    assert [r.decode_tokens for r in result.requests] == [2, 5, 3, 8]
    assert all(r.status == "finished" for r in result.requests)
    assert all(len(d.request_ids) <= 2 for d in result.scheduler_decisions)
    assert result.final_active_requests == 0
    assert len(result.scheduler_decisions) > 1


def test_budget_exhaustion_is_reported_as_cancellation():
    result = RuntimeBenchmarkRunner().run(Scenario(token_limits=(5,), max_steps=1))
    assert result.step_budget_exhausted is True
    assert result.requests[0].status == "cancelled"
    assert result.requests[0].decode_tokens == 0
    assert result.final_active_requests == 0


def test_invalid_scenario_is_rejected():
    with pytest.raises(ValueError):
        Scenario(token_limits=())
    with pytest.raises(ValueError):
        Scenario(max_steps=0)
