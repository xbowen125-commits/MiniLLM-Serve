"""Arrival gating changes actual selected plans, not just reported timestamps."""

from benchmarks.runtime.common.arrival import WorkloadRequest, pop_due
from benchmarks.runtime.orca.scenario import heterogeneous_workload
from benchmarks.runtime.orca.runner import run_benchmark


def test_arrival_changes_iteration_selection():
    simultaneous = run_benchmark(heterogeneous_workload(arrival_interval=0))
    staggered = run_benchmark(heterogeneous_workload(arrival_interval=.04))
    plans = lambda r: [(t["decode_request_ids"], t["prefill_request_ids"]) for t in r["iteration_trace"]]
    assert plans(simultaneous) != plans(staggered)
    assert len(simultaneous["iteration_trace"][0]["prefill_request_ids"]) == 4
    assert staggered["iteration_trace"][0]["prefill_request_ids"] == ("long",)
    assert simultaneous["metrics"]["simulated_duration_s"] != staggered["metrics"]["simulated_duration_s"]
    for result in (simultaneous, staggered):
        arrivals = {r["request_id"]: r["arrival_time"] for r in result["normalized_workload"]}
        for t in result["iteration_trace"]:
            assert all(arrivals[r] <= t["start_s"] for r in t["arrival_request_ids"])
            assert not set(t["waiting_after"]) - {r for r, a in arrivals.items() if a <= t["start_s"]}


def test_due_boundary_and_stable_order():
    pending = [WorkloadRequest("a", 16, 2, 1), WorkloadRequest("b", 32, 4, 1)]
    assert pop_due(pending, .99) == []
    assert [r.request_id for r in pop_due(pending, 1)] == ["a", "b"]
