"""Offline runtime simulations with explicit costs and exact token accounting."""

import json
import pytest

from benchmarks.runtime.continuous_batch.scenario import Scenario, SimRequest, scenarios
from benchmarks.runtime.continuous_batch.metrics import request_metrics
from benchmarks.runtime.continuous_batch.runner import run_comparison, run_policy
from benchmarks.runtime.continuous_batch.report import generate_report, save_reports


@pytest.mark.parametrize("scenario", scenarios(), ids=lambda s: s.name)
def test_both_policies_complete_identical_workload(scenario):
    results = run_comparison(scenario)["results"]
    for policy, result in results.items():
        assert result["idle_after_test"]
        assert result["metrics"]["completed_requests"] == len(scenario.requests)
        assert result["metrics"]["generated_tokens"] == sum(r.max_tokens for r in scenario.requests)
        assert result["metrics"]["max_active_decode_batch_size"] <= (1 if policy == "sequential" else scenario.max_active)


def test_exact_single_request_metrics():
    scenario = Scenario("one", (SimRequest("a", 0, 2),), prefill_base_s=1,
        prefill_per_request_s=0, decode_base_s=2, decode_per_request_s=0)
    result = run_policy(scenario, "sequential")
    assert result["requests"] == [{"request_id": "a", "queue_delay_s": 0,
        "ttft_s": 3, "total_latency_s": 5, "generated_tokens": 2}]
    assert result["metrics"]["simulated_tokens_per_second"] == 2 / 5


def test_incremental_admission_while_decode_remains_active():
    scenario = Scenario("arrival", (SimRequest("a", 0, 5), SimRequest("b", .04, 1)))
    trace = run_policy(scenario, "continuous_batch")["step_trace"]
    assert any(t["phase"] == "prefill" and t["request_ids"] == ["b"] for t in trace)
    assert any(t["phase"] == "decode" and t["request_ids"] == ["a", "b"] for t in trace)


def test_sequential_finishes_before_next_prefill():
    result = run_policy(scenarios()[0], "sequential")
    trace = result["step_trace"]
    assert [t["phase"] for t in trace[:4]] == ["prefill", "decode", "decode", "prefill"]


def test_no_speedup_assumption_with_pure_per_request_costs():
    scenario = Scenario("linear", (SimRequest("a", 0, 2), SimRequest("b", 0, 2)),
                        prefill_base_s=0, decode_base_s=0)
    results = run_comparison(scenario)["results"]
    assert results["sequential"]["metrics"]["simulated_duration_s"] == pytest.approx(
        results["continuous_batch"]["metrics"]["simulated_duration_s"])


def test_reproducibility_export_and_definitions(tmp_path):
    scenario = scenarios()[1]
    first, second = run_comparison(scenario), run_comparison(scenario)
    assert first["results"] == second["results"]
    json_path, markdown_path = save_reports([first], tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["results"] == first["results"]
    assert markdown_path.read_text(encoding="utf-8") == generate_report([first])


def test_invalid_metrics_and_policy():
    with pytest.raises(ValueError):
        request_metrics(2, 1, 3, 4, 1)
    with pytest.raises(ValueError):
        run_policy(scenarios()[0], "unknown")


def test_invalid_workload():
    with pytest.raises(ValueError):
        Scenario("empty", ())
    with pytest.raises(ValueError):
        SimRequest("a", 0, 0)
