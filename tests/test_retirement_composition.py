"""Synthetic evidence mutation and explicit bounded-graph verification."""
from copy import deepcopy

import pytest

from benchmarks.runtime.online_http.retirement_composition import check_composition, explore, successors


def example():
    cache = {"A": {"slot": 0, "context_matches_cache": True, "retirement_matches_cache": False}}
    request = {"A": {"status": "running", "has_cache_ref": True, "request_object_identity": 111}}
    worker = {"slots": cache, "retirements": {}, "session_request_ids": ["A"]}
    status = {"pending": 1, "allocated_slots": 1, "contexts": 1, "claims": 1,
              "released_ids": 0, "finished_ids": 0, "runtime_active_requests": 1, "pending_commands": 0}
    rows = [
        dict(kind="lease_granted", request_id="A", request_object_identity=111, lease_id=1),
        dict(kind="owner_publish", owner_thread=7, deliveries={"A": {}}, status=deepcopy(status),
             snapshot={"requests": deepcopy(request), "worker": deepcopy(worker)}),
    ]
    request["A"]["status"] = "cancelled"
    rows.append(dict(kind="cancel_end", snapshot={"requests": deepcopy(request), "worker": deepcopy(worker)}))
    worker["slots"]["A"].update(context_matches_cache=False, retirement_matches_cache=True)
    worker["session_request_ids"] = []
    worker["retirements"] = {"A": {"request_object_identity": 111, "slot_released": False, "attempts": 1}}
    rows += [dict(kind="release_begin", request_id="A", request_object_identity=111),
             dict(kind="release_end", request_id="A", request_object_identity=111, request_cache_ref=True,
                  error="release failed", snapshot=deepcopy(worker))]
    rows.append(dict(kind="owner_publish", owner_thread=7, deliveries={"A": {}}, status=deepcopy(status),
                     snapshot={"requests": deepcopy(request), "worker": deepcopy(worker)}))
    empty = {"slots": {}, "retirements": {}, "session_request_ids": []}
    rows += [dict(kind="release_begin", request_id="A", request_object_identity=111),
             dict(kind="release_end", request_id="A", request_object_identity=111, request_cache_ref=False,
                  error=None, snapshot=deepcopy(empty)),
             dict(kind="retirement_evidence", requests={"A": {"status": "cancelled", "has_cache_ref": False,
                                                            "request_object_identity": 111}}),
             dict(kind="lease_release_begin", request_id="A", request_object_identity=111, lease_id=1, released=False),
             dict(kind="lease_release_end", request_id="A", request_object_identity=111, lease_id=1, released=True,
                  running_lease_ids=[], error=None),
             dict(kind="worker_close_begin", runtime_identity=12, owns_worker=True, snapshot=deepcopy(empty)),
             dict(kind="http_observation", label="owner-stopped", status={k: 0 for k in status},
                  scheduler={"running_requests": 0, "queued_requests": 0}, pending_finalizers=0,
                  owner_thread_alive=False, snapshot=deepcopy(empty), request_evidence={
                      "A": {"has_cache_ref": False, "status": "cancelled", "request_object_identity": 111}})]
    return [{"seq": i, "event_id": str(i), "thread_id": 7, **row} for i, row in enumerate(rows, 1)]


def test_complete_evidence_independent_recompute():
    rows = example()
    assert check_composition(rows) == check_composition(deepcopy(rows))
    assert check_composition(rows)["status"] == "PASS"
    assert check_composition(rows)["admitted"] is False


@pytest.mark.parametrize("mutation,code", [
    ("early", "LEASE_AFTER_RETIREMENT"), ("lost", "RECOVERABLE_RETIREMENT"),
    ("identity", "COMPOSED_IDENTITY"), ("idle", "NO_FALSE_IDLE"),
    ("thread", "OWNER_SERIALIZATION"), ("borrower", "WORKER_CLOSE_OWNERSHIP"),
    ("slot", "RECOVERABLE_RETIREMENT"), ("final_ref", "COMPOSED_FINAL_DRAIN"),
])
def test_contradictions_fail(mutation, code):
    rows = example()
    failure = next(r for r in rows if r["kind"] == "release_end" and r.get("error"))
    if mutation == "early":
        row = next(r for r in rows if r["kind"] == "lease_release_begin")
        rows.remove(row); rows.insert(5, row)
    elif mutation == "lost":
        failure["snapshot"]["retirements"] = {}
    elif mutation == "identity":
        failure["snapshot"]["retirements"]["A"]["request_object_identity"] = 222
    elif mutation == "idle":
        next(r for r in rows if r["kind"] == "owner_publish")["status"]["pending"] = 0
    elif mutation == "thread":
        next(r for r in rows if r["kind"] == "release_begin")["thread_id"] = 99
    elif mutation == "borrower":
        next(r for r in rows if r["kind"] == "worker_close_begin")["owns_worker"] = False
    elif mutation == "slot":
        failure["snapshot"]["retirements"]["A"]["slot_released"] = True
    else:
        rows[-1]["request_evidence"]["A"]["has_cache_ref"] = True
    result = check_composition(rows)
    assert result["status"] == "FAIL"
    assert next(c for c in result["checks"] if c["code"] == code)["status"] == "FAIL"


@pytest.mark.parametrize("mutation", ["final", "cache", "post_object", "lease"])
def test_missing_evidence_is_unknown(mutation):
    rows = example()
    if mutation == "final":
        rows.pop()
    elif mutation == "cache":
        next(r for r in rows if r["kind"] == "owner_publish")["snapshot"]["worker"]["slots"] = None
    elif mutation == "post_object":
        rows[-1]["request_evidence"] = {}
    else:
        next(r for r in rows if r["kind"] == "lease_release_end").pop("running_lease_ids")
    assert check_composition(rows)["status"] == "UNKNOWN"


def test_bounded_search_is_deterministic_and_not_whole_runtime_proof():
    result = explore()
    assert result == explore()
    assert result["exhausted"] and result["unexplored_total"] == 0
    assert result["state_total"] > 0 and result["transition_total"] > result["state_total"]
    assert result["terminal_total"] > 0 and result["safety"] == "PASS"
    assert result["retry_cycle_witnesses"]
    assert result["termination"] == "NOT_GUARANTEED"
    assert result["whole_runtime_equivalence"] == "UNKNOWN"
    from benchmarks.runtime.online_http.retirement_composition import State, Request
    for owned in ("owned", "borrowed"):
        terminal = State(requests=(Request(life="cancelled", lease="released"), Request()),
                         owner="closing", ownership=owned)
        stopped = next(after for action, after in successors(terminal) if action == "stop")
        assert stopped.worker_closed is (owned == "owned")
