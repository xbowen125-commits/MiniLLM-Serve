"""Identity proof tests use synthetic evidence; no model download/GPU required."""
import asyncio
from copy import deepcopy
import json
import hashlib
from pathlib import Path

import pytest

from benchmarks.runtime.online_http.config import load_config
from benchmarks.runtime.online_http.identity_check import reconstruct, historical_difference, audit
from benchmarks.runtime.online_http.identity_trace import IdentityObserver, HTTPIdentityBoundary
from benchmarks.runtime.online_http.runtime_trace import RuntimeTrace
from benchmarks.runtime.online_http.artifacts import git_snapshot, write_new
from benchmarks.runtime.online_http.trace_check import seal
from test_http_continuous import application, asgi_request, wait_idle
from test_runtime_retirement_integration import trace_directory, test_owner_admission_finalization_and_retry as _owner_case

CONFIG = Path("benchmarks/runtime/online_http/configs/qwen_admission_v2.json")


def fixture_rows():
    config = load_config(CONFIG)
    events, clients = [], []
    def event(kind, **fields):
        n = len(events) + 1
        row = dict(kind=kind, event_id=f"e{n}", seq=n, pid=123, run_id="run", config_sha256=config.sha256,
                   git={"source_tree_sha256": "source"},
                   **{k: config.data[k] for k in ("model_revision", "dtype", "device")}, **fields)
        events.append((row, {"file": "runtime-trace.jsonl", "line": n, "sha256": "raw", "event_id": row["event_id"]}))
    for i, item in enumerate(config.data["requests"]):
        p, cid, exchange, rid, command, obj = item["request_id"], f"c{i}", f"h{i}", str(i), f"q{i}", i + 100
        link = dict(planned_request_id=p, client_record_id=cid, http_exchange_id=exchange)
        state = dict(request_id=rid, request_object_identity=obj)
        event("identity_http_begin", **link)
        event("identity_request_created", **link, **state)
        event("identity_admission_accept", **link, **state, lease_id=rid)
        event("identity_command_enqueue", **link, **state, command_id=command)
        event("identity_command_start", **link, **state, command_id=command)
        event("submit_end", request_id=rid, error=None)
        event("identity_runtime_handoff", **state, runtime_context_id=rid)
        event("worker_begin", members=[rid])
        event("identity_command_finish", **state, command_id=command, terminal_result_id=command,
              has_cache_ref=False, status="finished")
        event("identity_response_identity", **link, response_id=f"response{i}")
        event("identity_finalizer_complete", **state, released=True, has_cache_ref=False)
        clients.append((dict(request_id=p, planned_request_id=p, client_record_id=cid, lease_id=rid,
            response_id=f"response{i}", generated_count_source="terminal_request_state_sse",
            authoritative_usage={"engine_request_id": rid, "count_source": "terminal_request_state", "status": "finished",
                "object": "minillm.benchmark.usage", "id": f"response{i}", "request_id": rid,
                "includes_eos": True, "includes_stop_and_hidden_tokens": True,
                "usage": {"completion_tokens": item["max_tokens"]}}, scheduled_at_s=item["arrival_s"],
            max_tokens=item["max_tokens"], generated_output_tokens=item["max_tokens"], finish_reason="length", success=True, saw_done=True,
            send_lateness_ms=1, server_pid=123, request_body_sha256=hashlib.sha256(json.dumps({
                "model": config.data["model"], "messages": item["messages"], "max_tokens": item["max_tokens"],
                "stream": config.data["stream"], **{k: config.data[k] for k in
                ("temperature", "top_p", "top_k", "seed", "stop")}}, ensure_ascii=False).encode()).hexdigest()),
            {"file": "requests.jsonl", "line": i + 1, "sha256": "client"}))
    return config, events, clients


def check(events=None, clients=None):
    config, default_events, default_clients = fixture_rows()
    return reconstruct(config, default_events if events is None else events,
                       default_clients if clients is None else clients, current_source={"source_tree_sha256": "source"})


def test_six_request_chain_uses_direct_references():
    result = check()
    assert result["status"] == "PASS", result
    assert len(result["chains"]) == 6
    assert all(c["terminal_result_id"] and c["evidence"]["worker"] for c in result["chains"])
    assert result["full_runtime_equivalence"] == "UNKNOWN"


@pytest.mark.parametrize("kind", ["identity_command_finish", "identity_admission_accept", "identity_command_start",
                                  "identity_runtime_handoff", "worker_begin"])
def test_missing_edge_is_unknown(kind):
    _, events, clients = fixture_rows()
    events = [(r, f) for r, f in events if not (r["kind"] == kind and (r.get("request_id") == "0" or r.get("members") == ["0"]))]
    assert check(events, clients)["status"] == "UNKNOWN"


@pytest.mark.parametrize("field,value", [("request_object_identity", 999), ("runtime_context_id", "foreign")])
def test_conflicting_original_identity_fails(field, value):
    _, events, clients = fixture_rows()
    next(r for r, _ in events if r["kind"] == "identity_runtime_handoff")[field] = value
    assert check(events, clients)["status"] == "FAIL"


def test_duplicate_client_not_silently_deduplicated():
    _, events, clients = fixture_rows()
    assert check(events, clients + [deepcopy(clients[0])])["status"] == "FAIL"


def test_identical_event_reapplication_idempotent_changed_event_fails():
    _, events, clients = fixture_rows()
    assert check(events + [deepcopy(events[0])], clients)["status"] == "PASS"
    altered = deepcopy(events[0]); altered[0]["pid"] = 999
    assert check(events + [altered], clients)["status"] == "FAIL"


def test_extra_runtime_request_not_dissolved_by_counts():
    _, events, clients = fixture_rows()
    events.append(({"kind": "submit_end", "request_id": "extra", "event_id": "extra", "git": {"source_tree_sha256": "source"}},
                   {"file": "raw", "line": 999, "sha256": "raw"}))
    result = check(events, clients)
    assert result["status"] == "UNKNOWN" and result["unlinked_runtime"] == ["extra"]


def test_historical_source_unknown_even_same_head():
    _, events, clients = fixture_rows()
    events[0][0]["git"]["source_tree_sha256"] = "old"
    assert check(events, clients)["status"] == "UNKNOWN"


@pytest.mark.parametrize("field", ["model_revision", "dtype", "device"])
def test_frozen_model_identity_mismatch_fails(field):
    _, events, clients = fixture_rows()
    events[0][0][field] = "foreign"
    assert check(events, clients)["status"] == "FAIL"


def test_authoritative_count_is_rechecked_not_trusted_from_client_success():
    _, events, clients = fixture_rows()
    clients[0][0]["authoritative_usage"]["count_source"] = "retokenized_proxy"
    assert check(events, clients)["status"] == "UNKNOWN"


def test_lateness_and_early_lease_are_not_excused():
    _, events, clients = fixture_rows()
    clients[0][0]["send_lateness_ms"] = 25.001
    final = next(f for r, f in events if r["kind"] == "identity_finalizer_complete")
    final["line"] = 1
    result = check(events, clients)
    assert {i["code"] for i in result["issues"]} >= {"SEND_LATE_25MS", "LEASE_BEFORE_OWNER_CONFIRMATION"}


def test_frozen_offsets_cannot_be_changed():
    _, events, clients = fixture_rows()
    clients[0][0]["scheduled_at_s"] = .01
    assert check(events, clients)["status"] == "FAIL"


def test_stop_before_max_requires_direct_terminal_count_length_still_exact():
    config, events, clients = fixture_rows()
    clients[0][0].update(finish_reason="stop", generated_output_tokens=12)
    clients[0][0]["authoritative_usage"]["usage"]["completion_tokens"] = 12
    assert check(events, clients)["status"] == "UNKNOWN"
    result = reconstruct(config, events, clients, current_source={"source_tree_sha256": "source"}, terminals=[
        ({"kind": "terminal", "request_id": str(i), "pid": 123, "generated_token_ids": list(range(c["generated_output_tokens"]))},
         {"file": "observer.jsonl", "line": i + 1}) for i, (c, _) in enumerate(clients)])
    assert result["status"] == "PASS", result
    clients[0][0]["finish_reason"] = "length"
    assert check(events, clients)["status"] == "FAIL"


@pytest.mark.parametrize("field,value", [("server_pid", 999), ("request_body_sha256", "bad"), ("response_id", "foreign")])
def test_explicit_transport_identity_conflicts(field, value):
    _, events, clients = fixture_rows()
    clients[0][0][field] = value
    assert check(events, clients)["status"] == "FAIL"


def test_separate_disconnect_probe_explains_scope_not_complete_identity(tmp_path):
    events = [{"kind": "submit_end", "request_id": str(i), "error": None} for i in range(1, 28)]
    clients = [{"authoritative_usage": {"engine_request_id": str(i)}} for i in range(1, 28) if i != 26]
    (tmp_path / "runtime-trace.jsonl").write_text("\n".join(json.dumps(r) for r in events))
    (tmp_path / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in clients))
    write_new(tmp_path / "probe.json", {"faults": {"cancel": {"lease_id": "26", "response_id": "real-response",
        "terminal_counters": [{"engine_request_id": "26", "status": "cancelled"}]}}})
    result = historical_difference(tmp_path)
    assert result["status"] == "PASS" and result["runtime_submits"] == 27 and result["client_jsonl_rows"] == 26
    assert result["seven_layer_chain"] == "UNKNOWN"
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 26


def test_queue_order_counterexample_and_missing_entry():
    _, events, clients = fixture_rows()
    for n, (kind, rid) in enumerate([("identity_queue_enter", "0"), ("identity_queue_enter", "1"),
                                    ("identity_queue_leave", "1")]):
        events.append(({"kind": kind, "request_id": rid, "event_id": f"queue{n}", "queue_identity": 55,
            "operation": "get_batch", "git": {"source_tree_sha256": "source"}}, {"line": 100 + n}))
    assert any(i["code"] == "QUEUE_FIFO_VIOLATION" for i in check(events, clients)["issues"])


def test_raw_hash_error_refuses_audit_before_output(tmp_path):
    write_new(tmp_path / "trace-index.json", {"sha256": {"runtime-trace.jsonl": "bad", "trace-end.json": "bad"}})
    (tmp_path / "runtime-trace.jsonl").write_text("{}\n")
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="SHA-256"):
        audit(tmp_path, output, load_config(CONFIG), {})
    assert not output.exists()


@pytest.mark.parametrize("observer_fault", [False, True])
def test_real_cpu_owner_boundary_and_fault_noninterference(tmp_path, monkeypatch, observer_fault):
    source = git_snapshot()
    trace = RuntimeTrace(tmp_path, {"run_id": tmp_path.name, "config_sha256": "cpu-fake",
        "git": source, "model_revision": "fake", "dtype": "float32", "device": "cpu", "geometry": {"fake": True}}).install()
    observer = IdentityObserver(trace).install()
    if observer_fault:
        def failed_write(row):
            raise OSError("intentional observer persistence failure")
        monkeypatch.setattr(trace.journal, "append", failed_write)
    async def run():
        app, evidence = application()
        app.add_middleware(HTTPIdentityBoundary, observer=observer)
        async with app.router.lifespan_context(app):
            status, body, _ = await asgi_request(app, max_tokens=3)
            assert status == 200 and body["usage"]["completion_tokens"] == 3
            await wait_idle(app)
            assert evidence["worker"].pool.snapshot()["allocated_slots"] == 0
        assert not evidence["owner"].thread.is_alive()
    try:
        asyncio.run(run())
    finally:
        trace.close()
    assert bool(trace.errors) is observer_fault
    if not observer_fault:
        rows = [json.loads(x) for x in (tmp_path / "runtime-trace.jsonl").read_text().splitlines()]
        kinds = {r["kind"] for r in rows}
        assert kinds >= {"identity_admission_accept", "identity_command_enqueue", "identity_command_start",
                         "identity_command_finish", "identity_runtime_handoff", "identity_owner_stopped"}
        enq = next(r for r in rows if r["kind"] == "identity_command_enqueue")
        finish = next(r for r in rows if r["kind"] == "identity_command_finish")
        assert enq["request_object_identity"] == finish["request_object_identity"]
        assert finish["has_cache_ref"] is False


@pytest.mark.parametrize("mode", ["cancel", "unknown", "shutdown", "decode_error", "reference", "shutdown_reference"])
def test_cancel_retry_shutdown_boundary_composition(trace_directory, monkeypatch, mode):
    directory, trace = trace_directory
    IdentityObserver(trace).install()
    _owner_case(trace_directory, mode, monkeypatch)
    rows = [json.loads(x) for x in (directory / "runtime-trace.jsonl").read_text().splitlines()]
    finishes = [r for r in rows if r["kind"] == "identity_command_finish"]
    assert finishes and all(r["has_cache_ref"] is False for r in finishes)
    for finish in finishes:
        release = [r for r in rows if r["kind"] == "identity_finalizer_complete" and r["request_id"] == finish["request_id"]]
        assert release and finish["seq"] < release[0]["seq"]
    if mode in {"cancel", "shutdown", "shutdown_reference"}:
        assert any(r["kind"] == "identity_cancel_intent" for r in rows)
        assert any(r["kind"] == "cancel_end" for r in rows)  # effective is separate
