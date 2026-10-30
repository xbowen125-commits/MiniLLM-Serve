"""CPU/Fake real HTTP protocol checks; never execute a model or GPU trial."""

import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmarks.runtime.online_http.artifacts import Journal
from benchmarks.runtime.online_http.client import observe_request
from benchmarks.runtime.online_http.config import ProtocolConfig, load_config
from benchmarks.runtime.online_http.runner import ServerProcess, is_idle, replay, run_experiment

CONFIG = Path("benchmarks/runtime/online_http/configs/smoke.json")


@pytest.fixture
def tiny_http():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            case = body["messages"][0]["content"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if body["stream"] else "application/json")
            self.send_header("X-MiniLLM-Request-ID", "3")
            self.end_headers()
            if not body["stream"]:
                self.wfile.write(json.dumps({"id": "chatcmpl-json", "choices": [{"message": {"content": "one two"},
                    "finish_reason": "length"}], "usage": {"prompt_tokens": 3, "completion_tokens": 3}}).encode())
                return
            def chunk(delta, finish=None, rid="chatcmpl-test"):
                value = {"id": rid, "object": "chat.completion.chunk",
                    "choices": [{"delta": delta, "finish_reason": finish}]}
                self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.flush()
            try:
                chunk({"role": "assistant"})
                if case == "dribble":
                    for _ in range(200):
                        self.wfile.write(b"d"); self.wfile.flush(); time.sleep(.005)
                    return
                chunk({"content": "one two"}, rid="wrong" if case == "wrong-id" else "chatcmpl-test")
                if case in {"error", "cancel"}:
                    error = {"code": "request_cancelled"} if case == "cancel" else {}
                    self.wfile.write(("data: " + json.dumps({"error": error}) + "\n\n").encode())
                else:
                    chunk({}, "length")
                if case in {"usage", "usage-bad"}:
                    value = {"id": "chatcmpl-test", "request_id": "wrong" if case == "usage-bad" else "3",
                        "engine_request_id": "req-tiny", "count_source": "terminal_request_state", "status": "finished",
                        "object": "minillm.benchmark.usage", "includes_eos": True,
                        "includes_stop_and_hidden_tokens": True, "usage": {"prompt_tokens": 3,
                            "completion_tokens": 3, "total_tokens": 6}}
                    self.wfile.write(b"event: minillm.usage\n" + ("data: " + json.dumps(value) + "\n\n").encode())
                if case != "missing-done":
                    self.wfile.write(b"data: [DONE]\n\n")
            except (OSError, BrokenPipeError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown(); server.server_close(); thread.join(2)
        assert not thread.is_alive()


@pytest.mark.parametrize("case,expected", [("normal", "completed"), ("missing-done", "error"),
    ("wrong-id", "error"), ("error", "error"), ("cancel", "cancelled")])
def test_client_preserves_protocol_failures_and_never_counts_chunks_as_tokens(tiny_http, case, expected):
    config = load_config(CONFIG)
    item = {**config.data["requests"][0], "messages": [{"role": "user", "content": case}]}
    row = observe_request(tiny_http, config, item, time.perf_counter(), lambda text: len(text.split()))
    assert row["outcome"] == expected
    assert row["content_events"] == 1 and row["visible_output_tokens"] == 2
    assert row["generated_output_tokens"] is None
    assert row["prompt_count_source"] == "pinned_tokenizer_chat_template_preflight"
    assert row["ttft_ms"] is not None
    assert row["runtime_metrics"] is None
    if case == "missing-done":
        assert not row["saw_done"] and "missing DONE" in row["protocol_errors"]
    if case == "error":
        assert row["error_code"] == "server_error" and not row["success"]


def test_send_lateness_diagnostics_do_not_change_frozen_clock_boundaries(tiny_http):
    config = load_config(CONFIG)
    origin = time.perf_counter()
    generated = time.perf_counter() - origin
    worker_started = time.perf_counter() - origin
    row = observe_request(tiny_http, config, config.data["requests"][0], origin,
        lambda text: len(text.split()), generator_scheduled_at_s=generated, worker_started_at_s=worker_started)
    stages = [row[key] for key in ("scheduled_at_s", "generator_scheduled_at_s", "worker_started_at_s",
        "body_started_at_s", "body_ready_at_s", "sent_at_s", "http_call_at_s", "first_content_at_s", "terminal_at_s")]
    assert stages == sorted(stages)
    assert row["send_lateness_ms"] == pytest.approx(
        row["generator_lateness_ms"] + row["pool_wait_ms"] + row["client_preparation_ms"])
    assert row["http_setup_ms"] >= 0 and row["body_build_ms"] >= 0
    assert row["queue_wait_ms"] is None  # No admission header in this tiny server.


def test_nonstream_actual_usage_is_separate_from_visible_tokens(tiny_http):
    data = load_config(CONFIG).data
    data["stream"] = False
    config = ProtocolConfig.from_dict(data)
    row = observe_request(tiny_http, config, data["requests"][0], time.perf_counter(), lambda text: len(text.split()))
    assert row["success"]
    assert row["visible_output_tokens"] == 2 and row["generated_output_tokens"] == 3
    assert row["prompt_count_source"] == "server_usage"
    assert row["ttft_ms"] is row["tpot_ms"] is None


@pytest.mark.parametrize("case,success", [("usage", True), ("usage-bad", False)])
def test_named_usage_is_identity_checked_and_never_counted_as_content(tiny_http, case, success):
    data = load_config(CONFIG).data
    data["authoritative_sse_usage"] = True
    config = ProtocolConfig.from_dict(data)
    item = {**data["requests"][0], "messages": [{"role": "user", "content": case}]}
    row = observe_request(tiny_http, config, item, time.perf_counter(), lambda s: len(s.split()))
    assert row["success"] is success
    assert row["content_events"] == 1 and row["visible_output_tokens"] == 2
    if success:
        assert row["generated_output_tokens"] == 3
        assert row["generated_count_source"] == "terminal_request_state_sse"
    else:
        assert row["protocol_errors"]


def test_absolute_deadline_retains_partial_failed_record(tiny_http):
    data = load_config(CONFIG).data
    data["request_timeout_s"] = .04
    config = ProtocolConfig.from_dict(data)
    item = {**data["requests"][0], "messages": [{"role": "user", "content": "dribble"}]}
    started = time.perf_counter()
    row = observe_request(tiny_http, config, item, started, lambda text: len(text.split()))
    assert time.perf_counter() - started < 1
    assert row["outcome"] == "timeout" and row["error_code"] == "client_timeout"
    assert row["terminal_at_s"] is not None


def test_whole_phase_deadline_caps_late_request_budget(tiny_http):
    config = load_config(CONFIG)
    item = {**config.data["requests"][0], "messages": [{"role": "user", "content": "dribble"}]}
    started = time.perf_counter()
    row = observe_request(tiny_http, config, item, started, lambda text: len(text.split()),
                          run_deadline=started + .04)
    assert time.perf_counter() - started < 1
    assert 0 < row["effective_timeout_s"] <= .04
    assert row["outcome"] == "timeout" and row["error_code"] == "client_timeout"


def test_client_tokenizer_error_cannot_drop_a_request(tiny_http):
    config = load_config(CONFIG)
    def fail(_text):
        raise ValueError("tokenizer test failure")
    row = observe_request(tiny_http, config, config.data["requests"][0], time.perf_counter(), fail)
    assert row["outcome"] == "error" and row["error_code"] == "client_tokenization_error"


def test_nominal_arrival_offsets_affect_replay_and_are_not_moved(tmp_path, monkeypatch):
    import benchmarks.runtime.online_http.runner as module
    config = load_config(CONFIG)
    trace = [dict(config.data["requests"][0], request_id="a", arrival_s=0),
             dict(config.data["requests"][1], request_id="b", arrival_s=.04)]
    def observe(_url, _config, item, origin, _count, **kwargs):
        assert kwargs["run_deadline"] == origin + config.data["max_run_seconds"]
        assert kwargs["worker_started_at_s"] >= kwargs["generator_scheduled_at_s"] >= item["arrival_s"]
        actual = time.perf_counter() - origin
        return {"request_id": item["request_id"], "scheduled_at_s": item["arrival_s"],
            "sent_at_s": actual, "terminal_at_s": actual, "success": True, "outcome": "completed"}
    monkeypatch.setattr(module, "observe_request", observe)
    journal = Journal(tmp_path / "arrival.jsonl")
    try:
        rows, _ = replay("unused", config, trace, None, journal, {"config_sha256": config.sha256})
    finally:
        journal.close()
    assert [row["scheduled_at_s"] for row in rows] == [0, .04]
    assert rows[1]["sent_at_s"] - rows[0]["sent_at_s"] >= .02


def test_generator_submission_and_client_pool_wait_are_distinct(tmp_path, monkeypatch):
    import benchmarks.runtime.online_http.runner as module
    data = load_config(CONFIG).data
    data["max_client_concurrency"] = 1
    config = ProtocolConfig.from_dict(data)
    trace = [dict(data["requests"][0], request_id="a", arrival_s=0),
             dict(data["requests"][0], request_id="b", arrival_s=.01)]
    def observe(_url, _config, item, origin, _count, **kwargs):
        if item["request_id"] == "a":
            time.sleep(.06)
        sent = time.perf_counter() - origin
        return {"request_id": item["request_id"], "sent_at_s": sent, "terminal_at_s": sent,
                "outcome": "completed", "success": True, **kwargs}
    monkeypatch.setattr(module, "observe_request", observe)
    journal = Journal(tmp_path / "pool.jsonl")
    try:
        rows, _ = replay("unused", config, trace, None, journal, {})
    finally:
        journal.close()
    assert rows[1]["generator_scheduled_at_s"] >= .01
    assert rows[1]["worker_started_at_s"] - rows[1]["generator_scheduled_at_s"] > .02


def test_formal_gpu_gate_blocks_without_preflight_before_any_server(tmp_path, monkeypatch):
    import benchmarks.runtime.online_http.runner as module
    config = load_config("benchmarks/runtime/online_http/configs/qwen_protocol.json")
    monkeypatch.setattr(module, "prepare_trace", lambda _c: (config.data["requests"], lambda _: 0))
    monkeypatch.setattr(module, "gpu_environment_snapshot", lambda _c: {"cuda_available": False})
    monkeypatch.setattr(module, "ServerProcess", lambda *_a, **_k: pytest.fail("formal GPU server must not start"))
    result = run_experiment(config, tmp_path / "blocked")
    assert result["trials"] == [] and result["admission"]["admitted"] is False
    assert {"PREFLIGHT_MISSING", "AUTHORITATIVE_OUTPUT_TOKENS_UNAVAILABLE"} <= set(result["admission"]["reason_codes"])
    assert (tmp_path / "blocked" / "admission.json").exists()
    assert "no server or formal request launched" in result["aborted"]
    assert "Formal admission gate" in (tmp_path / "blocked" / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("mode,source", [("continuous", "same"), ("legacy", "different")])
def test_ready_process_must_match_mode_and_source(tmp_path, monkeypatch, mode, source):
    import benchmarks.runtime.online_http.runner as module
    config = load_config(CONFIG)
    metadata = {"port": 12345, "config_sha256": config.sha256, "mode": mode,
                "git": {"source_tree_sha256": source}}
    (tmp_path / "server-ready.json").write_text(json.dumps(metadata), encoding="utf-8")
    server = ServerProcess.__new__(ServerProcess)
    server.config, server.directory = config, tmp_path
    server.mode, server.git = "legacy", {"source_tree_sha256": "same"}
    server.process = SimpleNamespace(poll=lambda: None, pid=123)
    monkeypatch.setattr(module, "get_json", lambda *a, **k: {
        "status": "ok", "loaded": True, "model": config.data["model"]})
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        server.wait_ready()


def test_invalid_metric_observation_prevents_valid_trial(tmp_path, monkeypatch):
    import benchmarks.runtime.online_http.runner as module
    config = load_config(CONFIG)
    class FakeProcess:
        def __init__(self, *_args):
            self.metadata = {"pid": 123}
            self.base_url = "unused"
        def wait_ready(self): return 0
        def close(self):
            return {"seconds": 0, "stopped": True, "final": {
                "pending_finalizers": 0, "stream_consumers": 0, "owner_thread_alive": False}}
    class FakeSampler:
        rows = []
        def __init__(self, *_args): pass
        def start(self): pass
        def close(self): pass
    monkeypatch.setattr(module, "ServerProcess", FakeProcess)
    monkeypatch.setattr(module, "ResourceSampler", FakeSampler)
    monkeypatch.setattr(module, "hardware_snapshot", lambda: {})
    monkeypatch.setattr(module, "drain", lambda *_a, **_k: {"seconds": 0, "idle_after_test": True,
        "status": {"running_requests": 0, "queued_requests": 0}})
    # Deliberately completed wire request with impossible usage. Raw evidence
    # must remain inspectable but may not establish a valid performance trial.
    bad = {"request_id": "a", "success": True, "outcome": "completed", "sent_at_s": 0,
        "first_content_at_s": .01, "last_content_at_s": .02, "terminal_at_s": .03,
        "visible_output_tokens": 2, "generated_output_tokens": -1, "send_lateness_ms": 0}
    monkeypatch.setattr(module, "replay", lambda *_a, **_k: ([bad], .03))
    result = module.run_trial(config, config.data["requests"], None, mode="legacy", pair=0,
        directory=tmp_path / "invalid-metric", config_path=CONFIG,
        git={"sha": "test", "dirty": True})
    assert result["summary"]["metric_validation_error_requests"] == 1
    assert not result["comparison_valid"]
    assert "retained samples with invalid metric observations" in result["issues"]


def test_startup_failure_retains_each_planned_request_with_nullable_schema(tmp_path, monkeypatch):
    import benchmarks.runtime.online_http.runner as module
    config = load_config(CONFIG)
    def fail_start(*_args):
        raise RuntimeError("intentional startup failure")
    monkeypatch.setattr(module, "ServerProcess", fail_start)
    monkeypatch.setattr(module, "hardware_snapshot", lambda: {})
    result = module.run_trial(config, config.data["requests"], None, mode="legacy", pair=0,
        directory=tmp_path / "startup-failed", config_path=CONFIG,
        git={"sha": "test", "dirty": True})
    rows = [json.loads(line) for line in (tmp_path / "startup-failed" / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == len(config.data["requests"])
    assert [row["trace_ordinal"] for row in rows] == list(range(len(rows)))
    assert all(row["outcome"] == "not_sent" and row["error_code"] == "trial_setup_failed" for row in rows)
    for row in rows:
        assert row["sent_at_s"] is row["generated_output_tokens"] is row["runtime_metrics"] is None
        assert row["ttft_ms"] is row["tpot_ms"] is row["e2e_ms"] is None
        assert row["max_tokens"] > 0 and row["git_dirty"] is True
    assert not result["comparison_valid"]
    assert result["summary"]["planned_requests"] == len(rows)


def test_missing_continuous_telemetry_cannot_establish_idle():
    empty_legacy = {"running_requests": 0, "queued_requests": 0}
    assert is_idle(empty_legacy, expected_mode="legacy")
    assert not is_idle(empty_legacy, expected_mode="continuous")
    assert not is_idle({**empty_legacy, "continuous": {}}, expected_mode="continuous")


def test_cpu_fake_four_fresh_processes_record_identical_trace_and_keep_failures(tmp_path):
    config = load_config(CONFIG)
    output = tmp_path / "cpu-http-smoke"
    result = run_experiment(config, output)
    assert [(r["pair_index"], r["mode"]) for r in result["trials"]] == [
        (0, "legacy"), (0, "continuous"), (1, "continuous"), (1, "legacy")]
    assert result["evidence_kind"] == "cpu_fake"
    assert len({r["server_metadata"]["pid"] for r in result["trials"]}) == 4
    traces = []
    for trial in result["trials"]:
        assert trial["server_metadata"]["actual_model_dtype"] == "fake"
        assert trial["server_metadata"]["actual_model_device"] == "cpu"
        assert len(trial["server_metadata"]["chat_template_sha256"]) == 64
        raw = [json.loads(line) for line in (output / trial["raw_jsonl"]).read_text(encoding="utf-8").splitlines()]
        formal = [r for r in raw if r["phase"] == "measured"]
        assert len(raw) == 5 and len(formal) == 4
        assert sum(r["outcome"] == "error" for r in formal) == 1
        assert len([r for r in raw if r["phase"] == "warmup"]) == 1
        assert all(r["config_sha256"] == config.sha256 for r in raw)
        assert all(isinstance(r["git_dirty"], bool) for r in raw)
        assert all(r["runtime_metrics"] is None for r in formal)
        assert trial["resources"]["after"]["idle_after_test"] is True
        assert trial["shutdown"]["stopped"] is True
        assert trial["shutdown"]["final"]["owner_thread_alive"] is False
        assert trial["summary"]["completed_generated_output_tokens_per_s"] is None
        assert trial["summary"]["completed_requests"] == 3
        assert not trial["comparison_valid"]  # intentional failures are retained, never speedup samples
        assert trial["formal_statistics"] is None
        assert trial["validity_gate"]["eligibility"]["formal_statistics"] is False
        local = json.loads((output / trial["trial_id"] / "summary.json").read_text())
        verdict = json.loads((output / trial["trial_id"] / "final-verdict.json").read_text())
        assert "trial_local_only" in local["validity_scope"]
        assert verdict["comparison_valid"] == trial["comparison_valid"]
        # JSONL is flushed in completion order, not falsely rewritten into
        # arrival order. The explicit frozen ordinal reconstructs the trace.
        assert sorted(r["trace_ordinal"] for r in formal) == list(range(4))
        traces.append([(r["request_id"], r["scheduled_at_s"], r["max_tokens"])
            for r in sorted(formal, key=lambda row: row["trace_ordinal"])])
    assert all(trace == traces[0] for trace in traces)
    manifest = json.loads((output / "artifacts.json").read_text())
    assert manifest["config_sha256"] == config.sha256
    prior = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(config, output)
    assert (output / "summary.json").read_bytes() == prior
    assert "CPU/Fake" in (output / "report.md").read_text(encoding="utf-8")


def test_default_legacy_path_and_real_gpu_opt_in_guard(monkeypatch):
    from minillm.config import Settings
    import benchmarks.runtime.online_http.runner as module
    assert Settings().execution_mode == "legacy"
    monkeypatch.setattr("sys.argv", ["runner", "--config",
        "benchmarks/runtime/online_http/configs/qwen_protocol.json", "--execute"])
    monkeypatch.setattr(module, "run_experiment", lambda *_args: pytest.fail("GPU run must not start"))
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
