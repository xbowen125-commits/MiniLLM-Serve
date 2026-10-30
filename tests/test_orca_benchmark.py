"""Virtual iteration traces and metrics, with no model loading."""

import json
import pytest

from benchmarks.runtime.continuous_batch.scenario import Scenario, SimRequest
from benchmarks.runtime.orca.scenario import heterogeneous_workload, scenarios, OrcaRequest, workload_metadata
from benchmarks.runtime.orca.runner import run_benchmark
from benchmarks.runtime.orca.metrics import decode_fairness
from benchmarks.runtime.orca.report import generate_report, save_reports


@pytest.mark.parametrize("scenario", scenarios())
def test_heterogeneous_completion_and_trace(scenario):
    result = run_benchmark(scenario)
    assert result["idle_after_test"]
    assert result["metrics"]["completed_requests"] == len(scenario.requests)
    assert result["metrics"]["generated_tokens"] == sum(r.max_tokens for r in scenario.requests)
    assert result["metrics"]["decode_fairness"] == 1
    assert 0 < result["metrics"]["batch_utilization"] <= 1
    for index, iteration in enumerate(result["iteration_trace"], 1):
        assert iteration["iteration_id"] == index
        assert len(iteration["decode_request_ids"]) + len(iteration["prefill_request_ids"]) <= scenario.max_active
    assert any(t["decode_request_ids"] and t["prefill_request_ids"] for t in result["iteration_trace"]) or scenario.requests[1].arrival_time == 0


def test_exact_single_request_metrics():
    scenario = Scenario("exact", (SimRequest("one", 0, 2),), max_active=2,
        prefill_base_s=1, prefill_per_request_s=0, decode_base_s=2, decode_per_request_s=0)
    m = run_benchmark(scenario)["metrics"]
    assert m["average_ttft_s"] == 3
    assert m["average_total_latency_s"] == 5
    assert m["simulated_tokens_per_second"] == .4
    assert m["batch_utilization"] == .5
    assert m["average_batch_size"] == 1
    assert m["iteration_count"] == 3


def test_fairness_is_normalized_not_raw_token_count():
    assert decode_fairness({"a": 20, "b": 2}, {"a": 20, "b": 2}) == 1
    assert decode_fairness({"a": 20, "b": 1}, {"a": 20, "b": 2}) < 1
    with pytest.raises(ValueError):
        decode_fairness({"a": 2}, {"a": 1})


def test_dynamic_admission_and_retirement():
    result = run_benchmark(heterogeneous_workload())
    traces = result["iteration_trace"]
    assert any("long" in t["decode_request_ids"] and t["prefill_request_ids"] for t in traces)
    retired = [r for t in traces for r in t["completed_this_iteration"]]
    assert sorted(retired) == sorted(r["request_id"] for r in result["requests"])


def test_reproducibility_and_export(tmp_path):
    first, second = run_benchmark(scenarios()[0]), run_benchmark(scenarios()[0])
    for key in ("metrics", "requests", "iteration_trace", "phase_trace"):
        assert first[key] == second[key]
    json_path, markdown_path = save_reports([first], tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["metrics"] == first["metrics"]
    assert markdown_path.read_text(encoding="utf-8") == generate_report([first])
    for section in ("Experimental Scope", "Methodology", "Results", "Analysis", "Limitations"):
        assert f"## {section}" in generate_report([first])


def test_synthetic_lengths_and_stable_arrival_order():
    scenario = scenarios()[-1]
    metadata = workload_metadata(scenario)
    assert len({r["prompt_length"] for r in metadata}) > 1
    assert len({r["generation_length"] for r in metadata}) > 1
    assert [r["arrival_order"] for r in metadata] == list(range(8))
    assert [r["request_id"] for r in metadata] == [r.request_id for r in scenario.requests]
    assert OrcaRequest("a", 0, 2, 16).generation_length == 2
    with pytest.raises(ValueError):
        OrcaRequest("a", 0, 2, 0)


def test_prompt_metadata_does_not_invent_timing_effect():
    first = Scenario("metadata", (OrcaRequest("a", 0, 2, 16),))
    second = Scenario("metadata", (OrcaRequest("a", 0, 2, 512),))
    assert run_benchmark(first)["metrics"] == run_benchmark(second)["metrics"]
    assert run_benchmark(first)["requests"][0]["prompt_length"] == 16


def test_invalid_workload():
    with pytest.raises(ValueError):
        heterogeneous_workload(short_count=0)
    with pytest.raises(ValueError):
        heterogeneous_workload(arrival_interval=-1)
