"""Deterministic retirement faults: actual Runtime + CPU tensor Worker."""
import copy

import pytest

from benchmarks.runtime.online_http.retirement import check_retirement, replay
from benchmarks.runtime.online_http.artifacts import git_snapshot
from minillm.memory.slot_kv_pool import SlotKVPool
from test_runtime_execution_trace import collect as _collect, submit


def collect(directory, action):
    source = git_snapshot()
    return _collect(directory, action, source_identity={
        key: source[key] for key in ("sha", "dirty", "source_tree_sha256")})


@pytest.mark.parametrize("fault,count", [
    ("slot_before", 1), ("slot_before", 2), ("slot_after", 1),
    ("slot_noop", 1), ("clear_before", 1), ("clear_after", 1),
    ("clear_noop", 1), ("query_unknown", 1),
])
def test_fault_then_acknowledge_original_object(tmp_path, monkeypatch, fault, count):
    observed = {}
    def action(runtime, worker):
        context = submit(runtime, "A", count=7)
        original_request = context.request
        runtime.step()
        peer = submit(runtime, "B", "long", count=7)
        runtime.step()
        peer_cache = peer.request.kv_cache_ref
        peer_slot = worker.pool.slot_id("B")
        calls = len(worker.loaded.model.calls)
        tokens = tuple(original_request.generated_tokens)
        release = SlotKVPool.release
        clear = worker._clear_request_cache
        query = worker._cache_absent
        remaining = [count]
        def release_fault(pool, rid):
            if rid == "A" and remaining[0] > 0:
                remaining[0] -= 1
                if fault == "slot_after":
                    release(pool, rid)
                if fault != "slot_noop":
                    raise RuntimeError("injected slot release error")
                return
            return release(pool, rid)
        def clear_fault(request):
            if request is original_request and remaining[0] > 0:
                remaining[0] -= 1
                if fault == "clear_after":
                    clear(request)
                if fault != "clear_noop":
                    raise RuntimeError("injected reference clearing error")
                return
            clear(request)
        def query_fault(rid):
            if rid == "A" and remaining[0] > 0:
                raise RuntimeError("resource state UNKNOWN")
            return query(rid)
        if fault.startswith("slot_"):
            monkeypatch.setattr(SlotKVPool, "release", release_fault)
        elif fault.startswith("clear_"):
            monkeypatch.setattr(worker, "_clear_request_cache", clear_fault)
        else:
            monkeypatch.setattr(worker, "_cache_absent", query_fault)
        for attempt in range(count):
            with pytest.raises(Exception):
                runtime.cancel("A")
            assert original_request.is_terminal
            assert "A" not in worker._sessions
            record = worker._retiring["A"]
            assert record.request is original_request
            assert record.attempts == attempt + 1
            assert record.last_error
            assert "A" not in runtime._continuous._released
            assert worker.pool.lookup("B") is peer_cache
            assert worker.pool.slot_id("B") == peer_slot
            assert tuple(original_request.generated_tokens) == tokens
            assert len(worker.loaded.model.calls) == calls
            if fault in {"slot_before", "slot_noop", "query_unknown"}:
                assert worker.pool.slot_id("A") != peer_slot
                assert record.slot_released is (None if fault == "query_unknown" else False)
                with pytest.raises(MemoryError):
                    worker.pool.allocate("new", object())
            else:
                with pytest.raises(KeyError):
                    worker.pool.lookup("A")
                assert record.slot_released is True
            if fault != "clear_after":
                assert original_request.kv_cache_ref is not None
        remaining[0] = 0
        runtime.cancel("A")
        runtime.cancel("A")
        assert original_request.kv_cache_ref is None
        assert "A" not in worker._retiring and "A" not in worker._sessions
        assert "A" not in worker.pool.snapshot()["request_slots"]
        runtime.acknowledge("A")
        runtime.acknowledge("A")
        assert original_request.kv_cache_ref is None
        assert "A" not in runtime._coordinator.contexts
        assert worker.pool.lookup("B") is peer_cache
        runtime.cancel("B")
        runtime.acknowledge("B")
        assert peer.request.kv_cache_ref is None
        assert runtime.retention() == {"contexts": 0, "claims": 0, "released_ids": 0, "finished_ids": 0}
        observed["request_id"] = id(original_request)
        observed["tokens"] = tokens
    _, rows, _ = collect(tmp_path, action)
    result = check_retirement(rows)
    assert result["status"] == "PASS", result
    releases = [r for r in rows if r["kind"] == "release_end" and r["request_id"] == "A"]
    assert len(releases) == count + 1
    assert {r["request_object_identity"] for r in releases} == {observed["request_id"]}
    assert releases[-1]["request_cache_ref"] is False
    assert releases[-1]["snapshot"]["retirements"] == {}
    assert replay(tmp_path)["status"] == "PASS"


def test_close_during_reference_cleanup_retry(tmp_path, monkeypatch):
    def action(runtime, worker):
        context = submit(runtime, "A", count=7)
        request = context.request
        runtime.step()
        clear = worker._clear_request_cache
        monkeypatch.setattr(worker, "_clear_request_cache", lambda req: (_ for _ in ()).throw(RuntimeError("clear unavailable")))
        for operation in (lambda: runtime.cancel("A"), runtime.close, runtime.close):
            with pytest.raises(Exception):
                operation()
            assert request.is_terminal and request.kv_cache_ref is not None
            assert worker.pool.snapshot()["allocated_slots"] == 0
            assert worker._retiring["A"].request is request
            assert "A" in runtime._coordinator.contexts
        monkeypatch.setattr(worker, "_clear_request_cache", clear)
        runtime.close(); runtime.close()
        assert request.kv_cache_ref is None
        assert worker._retiring == {} and worker._sessions == {}
        assert runtime.retention()["contexts"] == 0
    _, rows, _ = collect(tmp_path, action)
    assert check_retirement(rows)["status"] == "PASS"


def test_identity_and_missing_post_object_observation_rejected(tmp_path):
    def action(runtime, worker):
        submit(runtime, "A", count=7)
        runtime.step()
        runtime.cancel("A")
        runtime.acknowledge("A")
    _, rows, _ = collect(tmp_path, action)
    bad = copy.deepcopy(rows)
    next(r for r in bad if r["kind"] == "release_end")["request_object_identity"] = -1
    assert check_retirement(bad)["status"] == "FAIL"
    missing = copy.deepcopy(rows)
    for row in missing:
        if row["kind"] == "release_end":
            row.pop("request_cache_ref", None)
        if row["kind"] == "retirement_evidence":
            for request in row["requests"].values():
                request.pop("has_cache_ref", None)
        snapshot = row.get("snapshot", {})
        for request in snapshot.get("requests", {}).values():
            request.pop("has_cache_ref", None)
    assert check_retirement(missing)["status"] == "UNKNOWN"


def test_partial_allocation_exception_retains_identity(monkeypatch):
    from test_runtime_continuous import backend, runtime
    worker = backend()
    facade = runtime(worker)
    original_add = worker.pool.add_request_cache
    original_release = worker.pool.release
    def add_then_raise(rid, cache):
        original_add(rid, cache)
        raise RuntimeError("allocation committed then raised")
    monkeypatch.setattr(worker.pool, "add_request_cache", add_then_raise)
    monkeypatch.setattr(worker.pool, "release", lambda rid: (_ for _ in ()).throw(RuntimeError("release unavailable")))
    request = facade.submit("short", request_id="A").request
    try:
        with pytest.raises(Exception):
            facade.step()
        assert request.is_terminal
        assert worker._retiring["A"].request is request
        assert worker.pool.slot_id("A") == 0
        monkeypatch.setattr(worker.pool, "release", original_release)
        facade.acknowledge("A")
        assert request.kv_cache_ref is None
        assert worker._retiring == {} and worker._sessions == {}
        assert worker.pool.snapshot()["allocated_slots"] == 0
    finally:
        facade.close(); worker.close()


def test_orphan_cache_cannot_claim_original_reference_cleared():
    from test_runtime_continuous import backend
    worker = backend()
    worker.pool.allocate("orphan", object())
    with pytest.raises(RuntimeError, match="identity.*UNKNOWN"):
        worker.release_request("orphan")
    assert worker.pool.slot_id("orphan") == 0
    assert worker._retiring["orphan"].request is None
    assert worker._retiring["orphan"].reference_cleared is False
    # Explicit test teardown of a manually injected orphan, not production recovery.
    worker.pool.release("orphan")
    worker.close()
