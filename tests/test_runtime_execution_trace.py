"""Actual Runtime + fake tensor worker; no CUDA/model downloads/performance."""
import hashlib
import json

import pytest

from benchmarks.runtime.online_http.runtime_trace import RuntimeTrace
from benchmarks.runtime.online_http.trace_check import check_directory, check_rows, seal
from minillm.runtime import RuntimeFacade
from minillm.engine import GenerationConfig
from minillm.workers import BatchedTransformersWorker, LoadedModel
from minillm.memory.slot_kv_pool import SlotKVPool
from test_transformers_batch_worker import FakeModel, FakeTokenizer, FakeCache


def collect(directory, action, *, model=None, source_identity=None):
    identity = {"run_id": directory.name, "config_sha256": "a" * 64,
        "model_revision": "fake", "dtype": "float32", "device": "cpu",
        "git": {"sha": "b" * 40, "dirty": True, "source_tree_sha256": "c" * 64},
        "geometry": {"layers": 1, "kv_heads": 1, "head_dim": 1}}
    if source_identity is not None:
        identity["git"] = source_identity
    trace = RuntimeTrace(directory, identity).install()
    worker = BatchedTransformersWorker(LoadedModel(model or FakeModel(), FakeTokenizer()),
        slot_capacity=2, cache_factory=lambda _: FakeCache())
    runtime = RuntimeFacade(worker, execution_mode="continuous", max_batch_size=2)
    try:
        action(runtime, worker)
    finally:
        runtime.close()
        worker.close()
        trace.close()
    result = seal(directory)
    rows = [json.loads(line) for line in (directory / "runtime-trace.jsonl").read_text(encoding="utf-8").splitlines()]
    ending = json.loads((directory / "trace-end.json").read_text(encoding="utf-8"))
    return result, rows, ending


def submit(runtime, rid, prompt="short", count=3):
    return runtime.submit(prompt, GenerationConfig(max_tokens=count, temperature=0), request_id=rid)


def drain(runtime):
    for _ in range(12):
        runtime.step()
        if runtime.snapshot().active_requests == 0:
            break


def assert_no_counterexample(result):
    assert result["status"] == "UNKNOWN", result
    assert result["abstract_model"]["counterexamples"] == {}, result
    assert not result["admitted"]


def test_normal_trace_and_independent_recompute(tmp_path):
    def action(runtime, _worker):
        submit(runtime, "A")
        submit(runtime, "B", "long", 5)
        drain(runtime)
    result, rows, ending = collect(tmp_path, action)
    assert_no_counterexample(result)
    assert check_rows(rows, ending) == check_directory(tmp_path) == result
    totals = result["association"]
    assert totals["forwards"] == totals["associated_forwards"] > 0
    assert totals["iterations"] == totals["associated_iterations"] > 0
    assert totals["slot_kv_rows"] == totals["associated_slot_kv_rows"] > 0


def test_arrival_survivor_cancel_and_slot_reuse(tmp_path):
    def action(runtime, worker):
        submit(runtime, "A", count=7)
        submit(runtime, "B", "long", 3)
        runtime.step()
        slot = worker.pool.slot_id("B")
        runtime.step()
        runtime.cancel("B")
        runtime.cancel("B")
        submit(runtime, "C", "long", 2)
        runtime.step()
        assert worker.pool.slot_id("C") == slot
        drain(runtime)
    result, rows, _ = collect(tmp_path, action)
    assert_no_counterexample(result)
    assert any(r["kind"] == "cancel_end" and r["request_id"] == "B" for r in rows)


def test_single_input_failure_isolated(tmp_path):
    def action(runtime, _worker):
        submit(runtime, "bad", "bad")
        submit(runtime, "good", "long")
        drain(runtime)
        assert runtime.get_result("bad").status.value == "failed"
        assert runtime.get_result("good").status.value == "finished"
    assert_no_counterexample(collect(tmp_path, action)[0])


def test_whole_forward_failure(tmp_path):
    class Fails(FakeModel):
        def forward(self, **kwargs):
            raise RuntimeError("injected batch forward failure")
    def action(runtime, _worker):
        submit(runtime, "A")
        submit(runtime, "B", "long")
        drain(runtime)
    result, rows, _ = collect(tmp_path, action, model=Fails())
    assert_no_counterexample(result)
    assert any(r["kind"] == "forward_end" and r["error"] for r in rows)


def test_protocol_error_is_diagnosed_and_cleaned(tmp_path, monkeypatch):
    original = BatchedTransformersWorker.prefill_batch
    def duplicate(worker, contexts):
        results = original(worker, contexts)
        return [results[0], results[0]]
    monkeypatch.setattr(BatchedTransformersWorker, "prefill_batch", duplicate)
    def action(runtime, _worker):
        submit(runtime, "A")
        submit(runtime, "B", "long")
        with pytest.raises(RuntimeError, match="duplicate"):
            runtime.step()
    result, rows, _ = collect(tmp_path, action)
    assert_no_counterexample(result)
    assert any(r["kind"] == "executor_phase_end" and r["error"] for r in rows)


def test_cleanup_failure_is_not_release_success(tmp_path, monkeypatch):
    original = SlotKVPool.release
    attempts = []
    def fail_once(pool, rid):
        if rid == "A" and not attempts:
            attempts.append(rid)
            raise RuntimeError("injected release failure")
        return original(pool, rid)
    monkeypatch.setattr(SlotKVPool, "release", fail_once)
    def action(runtime, _worker):
        submit(runtime, "A", count=7)
        runtime.step()
        with pytest.raises(Exception):
            runtime.cancel("A")
        runtime.cancel("A")
    result, rows, _ = collect(tmp_path, action)
    assert_no_counterexample(result)
    assert "V2_CLEANUP_RETRY_REPRESENTATION" in result["reason_codes"]
    failures = [r for r in rows if r["kind"] == "release_end" and r["error"]]
    assert failures and "A" in failures[0]["snapshot"]["slots"]


@pytest.mark.parametrize("mutation,code", [
    ("executor", "PLAN_EXECUTOR_MEMBERS"), ("worker", "EXECUTOR_WORKER_MEMBERS"),
    ("forward", "FORWARD_MEMBERS"), ("slot", "SLOT_UNIQUENESS"),
    ("cache", "CACHE_OWNERSHIP"), ("identity", "TRACE_IDENTITY"),
])
def test_modified_semantics_rejected(tmp_path, mutation, code):
    def action(runtime, _worker):
        submit(runtime, "A")
        submit(runtime, "B", "long")
        drain(runtime)
    _, rows, ending = collect(tmp_path, action)
    if mutation == "executor":
        next(r for r in rows if r["kind"] == "executor_plan")["prefill"] = ["unknown"]
    elif mutation == "worker":
        next(r for r in rows if r["kind"] == "worker_begin")["members"] = ["unknown"]
    elif mutation == "forward":
        next(r for r in rows if r["kind"] == "forward_begin")["members"] = ["unknown", "B"]
    elif mutation in {"slot", "cache"}:
        slots = next(r for r in rows if r["kind"] == "worker_end" and len(r["snapshot"]["slots"]) == 2)["snapshot"]["slots"]
        if mutation == "slot":
            slots["A"]["slot"] = slots["B"]["slot"]
        else:
            slots["A"]["context_matches_cache"] = False
    else:
        rows[0]["run_id"] = "other-run"
    result = check_rows(rows, ending)
    assert code in result["reason_codes"]
    assert result["status"] == "FAIL"


def test_hash_tampering_rejected(tmp_path):
    result, _, _ = collect(tmp_path, lambda r, _w: (submit(r, "A"), drain(r)))
    assert_no_counterexample(result)
    path = tmp_path / "runtime-trace.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    assert check_directory(tmp_path)["reason_codes"] == ["TRACE_ARTIFACT_INTEGRITY"]


def test_missing_observation_is_not_complete(tmp_path):
    _, rows, ending = collect(tmp_path, lambda r, _w: (submit(r, "A"), drain(r)))
    row = next(r for r in rows if r["kind"] == "forward_begin")
    row["members"] = None
    result = check_rows(rows, ending)
    assert "FORWARD_MEMBERS" in result["reason_codes"]
    assert not result["admitted"]


def test_waiting_cancel_and_idempotent_close(tmp_path):
    def action(runtime, _worker):
        submit(runtime, "waiting")
        runtime.cancel("waiting")
        runtime.cancel("waiting")
        submit(runtime, "active")
        runtime.step()
        runtime.close()
        runtime.close()
    assert_no_counterexample(collect(tmp_path, action)[0])


def test_external_source_binding_mismatch(tmp_path):
    collect(tmp_path, lambda r, _w: (submit(r, "A"), drain(r)))
    result = check_directory(tmp_path, expected_identity={"config_sha256": "wrong"})
    assert result["status"] == "FAIL" and "identity mismatch" in result["error"]


def test_unconfirmed_cleanup_cannot_claim_drain(tmp_path):
    _, rows, ending = collect(tmp_path, lambda r, _w: (submit(r, "A"), drain(r)))
    release = next(r for r in reversed(rows) if r["kind"] == "release_end")
    release["error"] = "permanent cleanup failure"
    result = check_rows(rows, ending)
    assert result["status"] == "FAIL"
    assert "FINAL_DRAIN" in result["reason_codes"]


def test_non_sequence_protocol_result_does_not_break_observer(tmp_path, monkeypatch):
    original = BatchedTransformersWorker.prefill_batch
    def invalid(worker, contexts):
        original(worker, contexts)
        return 3
    monkeypatch.setattr(BatchedTransformersWorker, "prefill_batch", invalid)
    def action(runtime, _worker):
        submit(runtime, "A")
        with pytest.raises(TypeError):
            runtime.step()
    result, _, ending = collect(tmp_path, action)
    assert_no_counterexample(result)
    assert ending["errors"] == []


def test_pool_event_wrong_request_mapping_rejected(tmp_path):
    _, rows, ending = collect(tmp_path, lambda r, _w: (submit(r, "A"), drain(r)))
    next(r for r in rows if r["kind"] == "pool_allocate")["after_slot"] = 999
    assert "SLOT_EVENT_CONSISTENCY" in check_rows(rows, ending)["reason_codes"]
