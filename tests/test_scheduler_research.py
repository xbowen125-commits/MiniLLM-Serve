"""Deterministic policy comparisons; no GPU or production fairness claims."""

import json
import pytest

from minillm.contracts import ExecutionPhase as Phase
from benchmarks.runtime.scheduler_compare import SimulatedRequest
from benchmarks.runtime.scheduler_research.scenario import ResearchScenario, scenarios
from benchmarks.runtime.scheduler_research.runner import SchedulerResearchRunner
from benchmarks.runtime.scheduler_research.metrics import fairness_score, summarize
from benchmarks.runtime.scheduler_research.report import generate_report, save_reports


@pytest.mark.parametrize("scenario", scenarios(), ids=lambda s: s.name)
def test_all_policies_complete(scenario):
    results = SchedulerResearchRunner().run(scenario)["results"]
    for result in results.values():
        assert result["metrics"]["completed_requests"] == len(scenario.requests)
        assert len(set(result["selected_order"])) == len(scenario.requests)
        assert result["idle_after_test"]
        assert 0 < result["metrics"]["fairness_score"] <= 1


def test_phase_priority_aging_and_fifo():
    scenario = ResearchScenario("aged", (
        SimulatedRequest("old", Phase.WAITING, 0, 1),
        SimulatedRequest("new", Phase.DECODE, 99, 1)), start_time=100, aging_factor=1)
    results = SchedulerResearchRunner().run(scenario)["results"]
    assert results["FIFO"]["selected_order"] == ["old", "new"]
    assert results["Phase Priority"]["selected_order"] == ["new", "old"]
    assert results["Fair Aging"]["selected_order"] == ["old", "new"]


def test_same_phase_fifo_and_future_arrival():
    scenario = ResearchScenario("fifo", (SimulatedRequest("a", Phase.DECODE, 0, 1),
                                         SimulatedRequest("b", Phase.DECODE, 10, 1)))
    for result in SchedulerResearchRunner().run(scenario)["results"].values():
        assert result["selected_order"] == ["a", "b"]
        assert result["metrics"]["average_queue_delay"] == 0


def test_exact_metrics_and_threshold():
    rows = [{"queue_delay": wait, "ttft": wait + 1,
             "total_latency": wait + 2, "service_seconds": 2} for wait in (0, 2, 4)]
    m = summarize(rows, 2)
    assert m["average_queue_delay"] == 2
    assert m["p50_wait_time"] == 2
    assert m["p95_wait_time"] == pytest.approx(3.8)
    assert m["p99_wait_time"] == pytest.approx(3.96)
    assert m["average_ttft"] == 3
    assert m["average_total_latency"] == 4
    assert m["starvation_count"] == 1
    assert fairness_score([rows[0], rows[0]]) == 1
    assert m["fairness_score"] < 1


def test_export_and_reproducibility(tmp_path):
    runner = SchedulerResearchRunner()
    result = runner.run(scenarios()[0])
    assert result["results"] == runner.run(scenarios()[0])["results"]
    json_path, markdown_path = save_reports([result], tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["results"] == result["results"]
    assert markdown_path.read_text(encoding="utf-8") == generate_report([result])


def test_no_guaranteed_fair_improvement():
    scenario = ResearchScenario("same", tuple(SimulatedRequest(str(i), Phase.DECODE, 0, 2)
                                               for i in range(4)))
    results = SchedulerResearchRunner().run(scenario)["results"]
    assert results["FIFO"] == results["Fair Aging"] == results["Phase Priority"]


def test_invalid_config_and_policy():
    with pytest.raises(ValueError):
        ResearchScenario("empty", ())
    with pytest.raises(ValueError):
        ResearchScenario("invalid", scenarios()[0].requests, first_token_fraction=0)
    with pytest.raises(ValueError):
        SchedulerResearchRunner().run_policy(scenarios()[0], "unknown")
