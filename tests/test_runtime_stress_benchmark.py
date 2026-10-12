"""Stress simulation generation, long-tail metrics, and policy tradeoffs."""

import pytest

from benchmarks.runtime.metrics import queue_percentile
from benchmarks.runtime.stress_scenarios import StressScenario
from minillm.contracts import ExecutionPhase


def test_decode_flood_has_later_waiting_and_long_decode_requests():
    scenario = StressScenario.decode_flood()
    assert len(scenario.requests) == 24
    decode = [r for r in scenario.requests if r.phase is ExecutionPhase.DECODE]
    waiting = [r for r in scenario.requests if r.phase is ExecutionPhase.WAITING]
    assert len(decode) == 20 and len(waiting) == 4
    assert min(r.arrival_time for r in waiting) > min(r.arrival_time for r in decode)
    assert all(r.service_seconds == 20 for r in decode)


def test_mixed_length_and_regular_arrivals():
    assert len({r.service_seconds for r in StressScenario.mixed_length().requests}) > 3
    arrivals = StressScenario.continuous_arrival().requests
    assert len(arrivals) == 30
    assert all(b.arrival_time - a.arrival_time == 0.5 for a, b in zip(arrivals, arrivals[1:]))


def test_percentile_formula():
    assert queue_percentile([0, 10, 20, 30], 50) == 15
    assert queue_percentile([0, 10, 20, 30], 95) == pytest.approx(28.5)
    assert queue_percentile([], 95) == 0


@pytest.mark.parametrize("factory", [StressScenario.decode_flood, StressScenario.mixed_length,
                                     StressScenario.continuous_arrival])
def test_both_policies_complete_and_metrics_match(factory):
    scenario = factory()
    for result in scenario.run().values():
        assert result.completed_requests == len(scenario.requests)
        delays = [delay for _, delay in result.queue_delays]
        assert result.p50_queue_delay == queue_percentile(delays, 50)
        assert result.p95_queue_delay == queue_percentile(delays, 95)
        assert result.average_queue_delay == pytest.approx(sum(delays) / len(delays))
        assert result.max_queue_delay == max(delays)
        assert result.completed_per_sim_second == pytest.approx(len(delays) / result.simulation_duration)
        assert sum(result.service_count_distribution.values()) == len(delays)
        assert result.starvation_count == sum(d > scenario.starvation_threshold for d in delays)


def test_fair_does_not_guarantee_better_waiting_request_latency():
    results = StressScenario.decode_flood().run()
    fifo = dict(results["FIFO"].queue_delays)
    fair = dict(results["FairPhaseBatchPolicy"].queue_delays)
    assert fair["waiting-0"] > fifo["waiting-0"]
