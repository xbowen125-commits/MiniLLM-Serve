"""Direct rule examples, counterexamples, real CPU spans and proof boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.runtime.online_http.transition_relation import map_events
from benchmarks.runtime.online_http.transition_audit import audit, frozen_equivalence, search_report
from benchmarks.runtime.online_http.config import load_config
from test_resource_retirement_fix import collect
from test_runtime_execution_trace import submit, drain
from test_runtime_retirement_integration import (
    trace_directory, test_owner_admission_finalization_and_retry as _run_owner_case,
)


def sample():
    waiting = {"A": {"request_object_identity": 111, "status": "waiting", "phase": "waiting", "generated": 0, "has_cache_ref": False}}
    running = {"A": {**waiting["A"], "status": "running", "phase": "decode", "has_cache_ref": True}}
    terminal = {"A": {**running["A"], "status": "cancelled", "phase": "finished"}}
    empty = {"slots": {}, "retirements": {}, "session_request_ids": []}
    snapshot = lambda r: {"requests": deepcopy(r), "worker": deepcopy(empty)}
    rows = [
        dict(kind="lease_granted", request_id="A", request_object_identity=111, lease_id=1),
        dict(kind="submit_end", request_id="A", error=None, snapshot=snapshot(waiting)),
        dict(kind="scheduler_plan", iteration_id=1, prefill=["A"], decode=[], selection_input={
            "max_active": 2, "active": [], "waiting": [{"request_id": "A", "request_object_identity": 111,
                "created_time": 0, "status": "waiting", "phase": "waiting"}]}),
        dict(kind="executor_plan", iteration_id=1, prefill=["A"], decode=[], free_slots=2),
        dict(kind="executor_submit", iteration_id=1, phase_id="p", phase="prefill", members=["A"], snapshot=snapshot(waiting)),
        dict(kind="worker_begin", phase_id="p", phase="prefill", members=["A"]),
        dict(kind="forward_begin", phase_id="p", forward_id="f", members=["A"], input_shape=[1, 2], mask_shape=[1, 2], cache_before={"length": 0}),
        dict(kind="forward_end", forward_id="f", cache_after={"length": 2}, error=None),
        dict(kind="worker_end", phase_id="p", phase="prefill", results=[{"request_id": "A", "phase": "prefill", "has_token": False, "error": None}]),
        dict(kind="executor_phase_end", phase_id="p", phase="prefill", snapshot=snapshot(running)),
        dict(kind="cancel_begin", request_id="A", operation_id="c", snapshot=snapshot(running)),
        dict(kind="cancel_end", request_id="A", operation_id="c", snapshot=snapshot(terminal)),
        dict(kind="release_begin", request_id="A", request_object_identity=111, release_id="r"),
        dict(kind="release_end", request_id="A", request_object_identity=111, release_id="r", request_cache_ref=False, error=None, snapshot=deepcopy(empty)),
        dict(kind="acknowledge_begin", request_id="A", operation_id="a", snapshot=snapshot(terminal)),
        dict(kind="acknowledge_end", request_id="A", operation_id="a", error=None, snapshot=snapshot({})),
        dict(kind="lease_release_begin", request_id="A", request_object_identity=111, released=False),
        dict(kind="lease_release_end", request_id="A", request_object_identity=111, released=True, error=None),
        dict(kind="owner_publish", closing=True),
        dict(kind="worker_close_begin", runtime_identity=12, owns_worker=True),
        dict(kind="worker_close_end", runtime_identity=12, owns_worker=True, error=None, snapshot=deepcopy(empty)),
        dict(kind="close_begin", operation_id="z", snapshot=snapshot({})),
        dict(kind="close_end", operation_id="z", error=None, snapshot=snapshot({})),
        dict(kind="collector_end"),
    ]
    return [{"seq": i, "event_id": str(i), **r} for i, r in enumerate(rows, 1)]


def mapped(rows):
    return map_events(rows, input_sha256="a" * 64)


def test_positive_examples_cover_observed_rule_catalog():
    result = mapped(sample())
    assert result["status"] == "PASS", result
    assert set(result["rule_counts"]) == {"BIND", "PLAN_SELECT", "PLAN_TRIM", "EXECUTOR_DISPATCH", "WORKER_DISPATCH",
        "WORKER_RESULT", "FORWARD", "PHASE_APPLY", "CANCEL", "RETIRE_BEGIN", "RETIRE_END", "ACK", "LEASE", "OWNER", "CLOSE", "OBSERVE"}
    assert all(r["input_sha256"] == "a" * 64 and r["event_id"] for r in result["mapping"])
    assert result["frozen_schedule_equivalence"] == "UNKNOWN"


@pytest.mark.parametrize("kind,field,value,rule", [
    ("executor_plan", "decode", ["foreign"], "PLAN_TRIM"),
    ("executor_submit", "members", ["foreign"], "EXECUTOR_DISPATCH"),
    ("worker_begin", "members", ["foreign"], "WORKER_DISPATCH"),
    ("worker_end", "results", [{"request_id": "A", "phase": "prefill"}] * 2, "WORKER_RESULT"),
    ("forward_begin", "members", ["foreign"], "FORWARD"),
    ("release_end", "request_cache_ref", True, "RETIRE_END"),
    ("worker_close_begin", "owns_worker", False, "CLOSE"),
    ("release_end", "request_object_identity", 222, "RETIRE_END"),
])
def test_explicit_invariant_violations_fail(kind, field, value, rule):
    rows = sample()
    next(r for r in rows if r["kind"] == kind)[field] = value
    result = mapped(rows)
    assert any(r["rule_id"] == rule and r["status"] == "FAIL" for r in result["mapping"])


@pytest.mark.parametrize("kind,field", [("scheduler_plan", "selection_input"), ("release_end", "release_id"),
    ("release_end", "request_cache_ref"), ("release_begin", "request_object_identity"), ("forward_begin", "members")])
def test_missing_evidence_unknown_not_guessed(kind, field):
    rows = sample()
    next(r for r in rows if r["kind"] == kind).pop(field)
    result = mapped(rows)
    assert any(r["kind"] == kind and r["status"] == "UNKNOWN" for r in result["mapping"])


def test_early_lease_and_cancel_regeneration_are_counterexamples():
    rows = sample()
    lease = next(r for r in rows if r["kind"] == "lease_release_begin")
    rows.remove(lease); rows.insert(12, lease)
    assert any(r["rule_id"] == "LEASE" and r["status"] == "UNKNOWN" for r in mapped(rows)["mapping"])
    cancel = next(r for r in rows if r["kind"] == "cancel_end")
    cancel["snapshot"]["requests"]["A"]["generated"] = 1
    assert any(r["rule_id"] == "CANCEL" and r["status"] == "FAIL" for r in mapped(rows)["mapping"])


def test_duplicate_event_is_idempotent_but_changed_payload_fails():
    rows = sample()
    result = mapped(rows + [deepcopy(rows[0])])
    assert result["mapping"][-1]["rule_id"] == "IDEMPOTENT" and result["status"] == "PASS"
    duplicate = deepcopy(rows[0]); duplicate["lease_id"] = 2
    assert mapped(rows + [duplicate])["status"] == "FAIL"
    assert mapped(rows + [{"kind": "unsupported-new-event", "event_id": "extra"}])["status"] == "UNKNOWN"


def test_slot_allocation_update_release_do_not_clear_original_reference():
    rows = [
        {"kind": "submit_end", "request_id": "A", "error": None, "snapshot": {"requests": {
            "A": {"request_object_identity": 111, "status": "waiting", "phase": "waiting", "generated": 0, "has_cache_ref": False}}}},
        {"kind": "pool_allocate", "request_id": "A", "before_slot": None, "after_slot": 0, "slots": {"A": 0}, "error": None},
        {"kind": "retirement_evidence", "requests": {"A": {"request_object_identity": 111, "has_cache_ref": True}}},
        {"kind": "pool_update_request_cache", "request_id": "A", "before_slot": 0, "after_slot": 0, "slots": {"A": 0}, "error": None},
        {"kind": "pool_release", "request_id": "A", "before_slot": 0, "after_slot": None, "slots": {}, "error": None},
    ]
    for i, row in enumerate(rows):
        row.update(event_id=str(i), seq=i + 1)
    result = mapped(rows)
    assert result["status"] == "PASS"
    assert result["mapping"][-1]["after"]["A"]["slot_present"]["value"] is False
    assert result["mapping"][-1]["after"]["A"]["cache_ref"]["value"] is True
    rows[-1]["after_slot"] = 0
    assert mapped(rows)["status"] == "FAIL"
    rows[-1].pop("slots")
    assert any(r["rule_id"] == "SLOT_UPDATE" and r["status"] == "UNKNOWN" for r in mapped(rows)["mapping"])


def test_failed_cleanup_then_reference_only_retry_and_recovery():
    rows = sample()
    begin = next(r for r in rows if r["kind"] == "release_begin")
    end = next(r for r in rows if r["kind"] == "release_end")
    index = rows.index(end)
    failure = {**end, "event_id": "failure", "error": "release failed", "request_cache_ref": True,
        "snapshot": {"slots": {}, "session_request_ids": [], "retirements": {"A": {
            "request_object_identity": 111, "attempts": 1, "slot_released": True, "reference_cleared": False}}}}
    rows[index:index] = [failure, {**begin, "event_id": "retry", "release_id": "r2"}]
    end["release_id"] = "r2"
    result = mapped(rows)
    assert result["status"] == "PASS", result
    middle = next(r for r in result["mapping"] if r["event_id"] == "failure")
    assert middle["after"]["A"]["cache_ref"]["value"] is True
    assert middle["after"]["A"]["slot_present"]["value"] is False
    assert middle["after"]["A"]["release_confirmed"]["value"] is False
    late = deepcopy(next(r for r in rows if r["kind"] == "lease_release_begin")); late["event_id"] = "tooearly"
    rows.insert(rows.index(failure) + 1, late)
    assert any(r["event_id"] == "tooearly" and r["status"] == "FAIL" for r in mapped(rows)["mapping"])


def config():
    return load_config(Path(__file__).parents[1] / "benchmarks/runtime/online_http/configs/qwen_admission_v2.json")


def test_historical_probe_and_equal_head_do_not_prove_frozen_equivalence():
    frozen = config()
    rows = [{"kind": "submit_end", "request_id": "engine-1", "pid": 42, "config_sha256": frozen.sha256,
             "git": {"sha": "same-head", "source_tree_sha256": "old"}}]
    clients = [{"request_id": "short-01", "phase": "warmup", "pid": 42,
                "authoritative_usage": {"engine_request_id": "engine-1"}}]
    result = frozen_equivalence(frozen, rows, clients, {"sha": "same-head", "source_tree_sha256": "new"})
    assert result["historical"] and result["status"] == "UNKNOWN" and result["measured_records"] == 0
    assert result["request_links"][0]["status"] == "PASS"
    rows.append({**rows[0], "request_id": "unlinked-runtime"})
    coverage = frozen_equivalence(frozen, rows, clients, {"source_tree_sha256": "new"})
    assert coverage["unlinked_runtime_requests"] == ["unlinked-runtime"]
    assert next(c for c in coverage["checks"] if c["code"] == "CLIENT_RUNTIME_COVERAGE")["status"] == "UNKNOWN"


def test_frozen_arrival_mismatch_not_excused_as_timing_noise():
    frozen = config()
    rows = [{"kind": "submit_end", "request_id": str(i), "pid": 42, "git": {"source_tree_sha256": "now"},
             "config_sha256": frozen.sha256} for i in range(6)]
    clients = [{"request_id": r["request_id"], "phase": "measured", "pid": 42,
                "scheduled_at_s": r["arrival_s"], "max_tokens": r["max_tokens"], "send_lateness_ms": 0,
                "authoritative_usage": {"engine_request_id": str(i)}} for i, r in enumerate(frozen.data["requests"])]
    result = frozen_equivalence(frozen, rows, clients, {"source_tree_sha256": "now"})
    assert result["status"] == "UNKNOWN"  # observed replay never proves every path
    clients[2]["scheduled_at_s"] += .1
    assert frozen_equivalence(frozen, rows, clients, {"source_tree_sha256": "now"})["status"] == "FAIL"


def test_actual_cpu_spans_and_exclusive_authenticated_audit(tmp_path):
    source = tmp_path / "trace"; source.mkdir()
    def action(runtime, worker):
        submit(runtime, "A"); submit(runtime, "B", "long")
        runtime.step(); runtime.cancel("A"); runtime.acknowledge("A"); drain(runtime)
    _, rows, _ = collect(source, action)
    result = map_events(rows, input_sha256=hashlib.sha256((source / "runtime-trace.jsonl").read_bytes()).hexdigest())
    assert result["status"] == "PASS", result
    output = tmp_path / "new-audit"
    result = audit(source, output, Path(__file__).parents[1] / "benchmarks/runtime/online_http/configs/qwen_admission_v2.json")
    assert result["admitted"] is False
    admission = json.loads((output / "admission.json").read_text())
    assert not admission["admitted"] and not admission["current_gpu_preflight_executed"]
    assert admission["unclosed_checks"]
    assert json.loads((output / "frozen-equivalence.json").read_text())["historical"] is False
    with pytest.raises(FileExistsError):
        audit(source, output, Path(__file__).parents[1] / "benchmarks/runtime/online_http/configs/qwen_admission_v2.json")
    raw = source / "runtime-trace.jsonl"; raw.write_bytes(raw.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="SHA256"):
        audit(source, tmp_path / "tampered", Path(__file__).parents[1] / "benchmarks/runtime/online_http/configs/qwen_admission_v2.json")


def test_bounded_search_scc_counts_and_symbolic_identity_relations():
    result = search_report()
    assert result["initial_total"] == 2 and result["exhausted"] and result["unexplored_total"] == 0
    assert result["duplicate_successor_visits"] == result["transition_total"] - result["state_total"] + 2
    assert result["cyclic_scc_total"] > 0 and result["retry_cycle_witnesses"]
    assert all(len(r["slot_owners"]) <= 1 for r in result["symbolic_relations"])
    assert result["fixed_workload_exhaustion"] == result["whole_runtime_exhaustion"] == "UNKNOWN"


@pytest.mark.parametrize("mode", ["cancel", "unknown", "shutdown", "decode_error", "reference", "shutdown_reference"])
def test_actual_owner_event_to_rule_relation(trace_directory, monkeypatch, mode):
    directory, _ = trace_directory
    _run_owner_case(trace_directory, mode, monkeypatch)
    raw = (directory / "runtime-trace.jsonl").read_bytes()
    result = map_events([json.loads(line) for line in raw.splitlines()], input_sha256=hashlib.sha256(raw).hexdigest())
    assert result["status"] == "PASS", [r for r in result["mapping"] if r["status"] != "PASS"]
    assert result["rule_counts"]["RETIRE_END"] > 1
    assert result["rule_counts"]["LEASE"] >= 6
    assert result["frozen_schedule_equivalence"] == "UNKNOWN"


def test_prepared_trace_identity_is_distinct_and_authenticated(tmp_path):
    from benchmarks.runtime.online_http.workload import build_trace, trace_sha256
    source = tmp_path / "trace"; source.mkdir()
    collect(source, lambda runtime, worker: (submit(runtime, "A"), drain(runtime)))
    frozen = config()
    prepared = [{**row, "observed_prompt_tokens": 3} for row in build_trace(frozen)]
    # Synthetic tokenizer counts test hashing only, not real GPU/model correctness.
    root = tmp_path / "synthetic-bundle"; root.mkdir()
    manifest = root / "expected-shapes.json"
    manifest.write_text(json.dumps({"trace": prepared, "trace_sha256": trace_sha256(prepared)}), encoding="utf-8")
    client = root / "requests.jsonl"; client.write_text("", encoding="utf-8")
    (root / "artifacts.json").write_text(json.dumps({"sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest, client)}}), encoding="utf-8")
    config_path = Path(__file__).parents[1] / "benchmarks/runtime/online_http/configs/qwen_admission_v2.json"
    result = audit(source, tmp_path / "result", config_path, clients_path=client, artifact_root=root)
    assert result["prepared_identity"]["status"] == "PASS"
    assert result["prepared_identity"]["prepared_trace_sha256"] != result["frozen_trace_sha256"]
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="prepared trace manifest hash"):
        audit(source, tmp_path / "bad", config_path, clients_path=client, artifact_root=root)
