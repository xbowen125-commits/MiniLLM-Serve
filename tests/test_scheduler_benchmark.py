"""Scheduler comparison math, priority, and arrival simulation tests."""

import pytest

from benchmarks.runtime.scheduler_compare import SchedulerComparisonRunner, SimulatedRequest
from minillm.contracts import ExecutionPhase


def test_both_policies_complete_identical_workload():
    results = SchedulerComparisonRunner().run()
    assert all(r.completed_requests == 3 for r in results.values())
    assert set(results["FIFO"].selected_order) == set(results["FairPhaseBatchPolicy"].selected_order)


def test_queue_delay_calculation():
    result = SchedulerComparisonRunner().run()["FIFO"]
    assert result.queue_delays == (("waiting-old", 100), ("prefill", 15), ("decode", 9))
    assert result.average_queue_delay == pytest.approx(124 / 3)
    assert result.max_queue_delay == 100


def test_fair_aging_and_decode_priority_order():
    result = SchedulerComparisonRunner().run()["FairPhaseBatchPolicy"]
    assert result.selected_order == ("waiting-old", "decode", "prefill")
    assert result.queue_delays == (("waiting-old", 100), ("decode", 6), ("prefill", 16))
    assert result.average_queue_delay == pytest.approx(122 / 3)


def test_starvation_is_strict_threshold_count():
    results = SchedulerComparisonRunner(starvation_threshold=15).run()
    assert results["FIFO"].starvation_count == 1
    assert results["FairPhaseBatchPolicy"].starvation_count == 2


def test_future_arrivals_are_not_selected_early():
    requests = [SimulatedRequest("later", ExecutionPhase.DECODE, 5),
                SimulatedRequest("first", ExecutionPhase.WAITING, 0)]
    results = SchedulerComparisonRunner(start_time=0).run(requests)
    assert all(r.selected_order == ("first", "later") for r in results.values())
    assert all(r.max_queue_delay == 0 for r in results.values())


def test_empty_and_duplicate_workloads():
    assert SchedulerComparisonRunner().run([])["FIFO"].completed_requests == 0
    request = SimulatedRequest("same", ExecutionPhase.DECODE, 0)
    with pytest.raises(ValueError):
        SchedulerComparisonRunner().run([request, request])
