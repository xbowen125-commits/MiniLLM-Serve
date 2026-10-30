"""Offline retirement proofs and actual CPU worker fault injection."""
from dataclasses import replace

import pytest

from benchmarks.runtime.online_http.retirement import (
    JointState, ResourceStatus, check_retirement, explore, replay, successors,
)
from minillm.memory.slot_kv_pool import SlotKVPool
from test_runtime_execution_trace import collect, submit, drain


def release_rows(*, error=None, slot=None, ref=False):
    return [
        {"seq": 1, "kind": "cancel_end", "snapshot": {"requests": {
            "A": {"status": "cancelled", "has_cache_ref": True}},
            "worker": {"slots": {"A": {"slot": 0}}}}},
        {"seq": 2, "kind": "release_begin", "request_id": "A", "request_object_identity": 123},
        {"seq": 3, "kind": "release_end", "request_id": "A", "error": error,
         "request_object_identity": 123, "request_cache_ref": ref, "snapshot": {"slots": {} if slot is None else {"A": {"slot": slot}}}},
    ]


@pytest.mark.parametrize("error,slot,ref,status", [
    (None, None, False, "PASS"),
    (None, None, True, "FAIL"),
    (None, 0, True, "FAIL"),
    ("failure", 0, True, "UNKNOWN"),
    ("failure after removal", None, False, "UNKNOWN"),
    (None, None, None, "UNKNOWN"),
])
def test_release_evidence_matrix(error, slot, ref, status):
    assert check_retirement(release_rows(error=error, slot=slot, ref=ref))["status"] == status


def test_missing_slot_evidence_and_empty_trace():
    rows = release_rows()
    rows[-1]["snapshot"] = {}
    assert check_retirement(rows)["status"] == "UNKNOWN"
    assert check_retirement([])["status"] == "UNKNOWN"


def test_retry_does_not_regenerate_terminal_request():
    rows = release_rows(error="failure", slot=0, ref=True)
    rows.append({"seq": 4, "kind": "executor_submit", "phase": "decode", "members": ["A"]})
    assert check_retirement(rows)["status"] == "FAIL"


def test_idempotent_release_and_finalization():
    rows = release_rows()
    rows += [{**r, "seq": r["seq"] + 3} for r in rows[1:]]
    rows += [{"seq": 6, "kind": "retirement_evidence", "requests": {
        "A": {"status": "cancelled", "has_cache_ref": False, "request_object_identity": 123}}}]
    assert check_retirement(rows)["status"] == "PASS"


def test_duplicate_slot_rejected():
    rows = release_rows()
    rows[0]["snapshot"]["worker"]["slots"]["B"] = {"slot": 0}
    assert check_retirement(rows)["status"] == "FAIL"


def test_finite_submodel_and_permanent_failure_cycle():
    result = explore()
    assert result == explore()
    assert result["exhausted"] and result["unexplored_states"] == 0
    assert result["safety"] == "PASS" and result["retry_cycle_witnesses"]
    assert result["whole_runtime_exhaustion"] == "UNKNOWN"
    terminal = JointState(lifecycle="cancelled", resource=ResourceStatus.LIVE, slot=0, cache_ref=True)
    assert not terminal.drained
    assert not any(action in {"allocate", "decode_no_ownership_change", "start"}
                   for action, _ in successors(terminal))
    assert replace(terminal, resource=ResourceStatus.RETIRED, slot=None, cache_ref=False, confirmed=True).drained


def test_normal_worker_retirement_and_hash_integrity(tmp_path):
    _, rows, _ = collect(tmp_path, lambda runtime, worker: (submit(runtime, "A"), drain(runtime)))
    assert check_retirement(rows)["status"] == "PASS"
    assert replay(tmp_path) == replay(tmp_path)
    path = tmp_path / "runtime-trace.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    assert replay(tmp_path)["reason_codes"] == ["TRACE_ARTIFACT_INTEGRITY"]


@pytest.mark.parametrize("failures,after_removal", [(1, False), (3, False), (1, True)])
def test_actual_cleanup_retry_clears_original_reference(tmp_path, monkeypatch, failures, after_removal):
    original = SlotKVPool.release
    attempts = []
    def injected(pool, rid):
        attempts.append(rid)
        if len(attempts) <= failures:
            if after_removal:
                original(pool, rid)
            raise RuntimeError("deterministic cleanup failure")
        return original(pool, rid)
    monkeypatch.setattr(SlotKVPool, "release", injected)
    def action(runtime, worker):
        context = submit(runtime, "A", count=7)
        runtime.step()
        for _ in range(failures):
            with pytest.raises(Exception):
                runtime.cancel("A")
            assert context.request.is_terminal
            assert context.request.kv_cache_ref is not None
            assert "A" not in worker._sessions
            assert worker._retiring["A"].request is context.request
            assert worker._retiring["A"].attempts == len(attempts)
            assert worker._retiring["A"].last_error
            if not after_removal:
                assert worker.pool.slot_id("A") == 0
        runtime.cancel("A")
        assert worker.pool.snapshot()["allocated_slots"] == 0
        assert context.request.kv_cache_ref is None
        assert "A" not in worker._retiring
        runtime.close()
        assert context.request.kv_cache_ref is None
        runtime.close()
    _, rows, _ = collect(tmp_path, action)
    result = check_retirement(rows)
    assert result["status"] == "PASS", result
    assert len({r["request_object_identity"] for r in rows if r["kind"] == "release_end"}) == 1


def test_success_return_with_retained_slot_is_not_retirement(tmp_path, monkeypatch):
    original = SlotKVPool.release
    monkeypatch.setattr(SlotKVPool, "release", lambda pool, rid: None)
    def action(runtime, worker):
        submit(runtime, "A", count=7)
        runtime.step()
        with pytest.raises(Exception, match="still allocated"):
            runtime.cancel("A")
        assert worker.pool.slot_id("A") == 0
        monkeypatch.setattr(SlotKVPool, "release", original)
        runtime.cancel("A")
    _, rows, _ = collect(tmp_path, action)
    assert check_retirement(rows)["status"] == "PASS"
    assert any(r["kind"] == "release_end" and r["error"] for r in rows)


def test_shutdown_failure_preserves_occupied_slot_then_retry(tmp_path, monkeypatch):
    original = SlotKVPool.release
    fail = [True]
    def injected(pool, rid):
        if fail[0]:
            raise RuntimeError("permanent until external recovery")
        original(pool, rid)
    monkeypatch.setattr(SlotKVPool, "release", injected)
    def action(runtime, worker):
        context = submit(runtime, "A", count=7)
        runtime.step()
        for _ in range(3):
            with pytest.raises(Exception):
                runtime.close()
            assert worker.pool.slot_id("A") == 0
            assert context.request.is_terminal
        # A new allocation must not reuse A's retained slot.
        worker.pool.allocate("B", object())
        assert worker.pool.slot_id("B") != worker.pool.slot_id("A")
        with pytest.raises(Exception):
            worker.pool.allocate("C", object())
        original(worker.pool, "B")
        fail[0] = False
        runtime.close()
        assert context.request.kv_cache_ref is None
    _, rows, _ = collect(tmp_path, action)
    assert any(r["kind"] == "close_end" and r["error"] for r in rows)


def test_borrower_cannot_close_shared_worker():
    from test_runtime_continuous import backend, runtime
    worker = backend()
    a, b = runtime(worker), runtime(worker)
    try:
        a.submit("short", request_id="A")
        b.submit("long", request_id="B")
        a.step(); b.step()
        a.close(); a.close()
        assert worker.pool.snapshot()["request_slots"] == {"B": 1}
        b.step()
        with pytest.raises(ValueError, match="shared"):
            runtime(worker, owns_worker=True)
    finally:
        a.close(); b.close(); worker.close()


def test_terminal_lifecycle_cannot_restart():
    rows = release_rows()
    rows.append({"seq": 4, "kind": "step_end", "snapshot": {
        "requests": {"A": {"status": "running", "has_cache_ref": False}}, "worker": {"slots": {}}}})
    assert check_retirement(rows)["status"] == "FAIL"


def test_missing_snapshot_fields_cannot_pass():
    rows = release_rows()
    rows[0]["snapshot"]["requests"]["A"].pop("status")
    rows[0]["snapshot"]["worker"]["slots"]["A"].pop("slot")
    result = check_retirement(rows)
    assert result["status"] == "UNKNOWN"
    assert any(c["code"] == "RELEASE_EVIDENCE" and c["status"] == "UNKNOWN" for c in result["checks"])
