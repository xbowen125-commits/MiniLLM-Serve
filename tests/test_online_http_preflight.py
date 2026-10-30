"""Correctness gate tests use CPU/Fake only; no model download or GPU work."""

import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import pytest

from benchmarks.runtime.online_http.config import ProtocolConfig, load_config
from benchmarks.runtime.online_http import preflight


CONFIG = Path("benchmarks/runtime/online_http/configs/smoke.json")


def fake_config():
    value = load_config(CONFIG).data
    value["requests"] = value["requests"][:2]
    return ProtocolConfig.from_dict(value)


def test_preflight_plan_does_not_launch_server(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "run_preflight", lambda *_a, **_k: pytest.fail("plan launched probe"))
    monkeypatch.setattr("sys.argv", ["preflight", "--config", str(CONFIG)])
    preflight.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["plan_only"] and plan["performance_samples"] == 0
    assert plan["request_count_per_mode"] == 1


def test_real_gpu_requires_explicit_opt_in_before_creating_output(tmp_path):
    config = load_config("benchmarks/runtime/online_http/configs/qwen_protocol.json")
    with pytest.raises(ValueError, match="allow_gpu"):
        preflight.run_preflight(config, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_overall_status_never_treats_missing_checks_as_pass():
    assert preflight.overall_status([]) == "UNKNOWN"
    checks = [{"code": code, "status": "PASS"} for code in preflight.REQUIRED_CHECKS]
    assert preflight.overall_status(checks) == "PASS"
    checks[-1]["status"] = "UNKNOWN"
    assert preflight.overall_status(checks) == "UNKNOWN"
    checks[-1]["status"] = "FAIL"
    assert preflight.overall_status(checks) == "FAIL"


def test_formal_warmup_cannot_be_proven_by_diagnostic_batch_shapes():
    config = fake_config()
    events = [{"kind": "forward_completed", "success": True, "mode": shape["mode"], "phase": shape["phase"],
        "input_shape": [shape["batch_size"], 1 if shape["phase"] == "decode" else 9]}
        for shape in preflight.required_shapes(config)]
    coverage = preflight.warmup_coverage(config, events)
    assert coverage["missing_probe_shapes"] == []
    assert coverage["status"] == "FAIL"
    assert not coverage["formal_warmup_coverage"]
    assert coverage["tensor_shape_coverage"]["status"] == "UNKNOWN"
    assert coverage["tensor_shape_coverage"]["required_shapes"] == []
    assert len(coverage["tensor_shape_coverage"]["observed_shapes"]) == len(events)
    assert {shape["batch_size"] for shape in coverage["required_shapes"] if shape["mode"] == "continuous"} == {1, 2}


def test_failed_forward_attempts_cannot_claim_observed_shape_coverage():
    config = fake_config()
    shape = {"mode": "continuous", "phase": "prefill", "input_shape": [2, 20]}
    coverage = preflight.warmup_coverage(config, [
        {**shape, "kind": "forward_attempt", "call_id": "failed"},
        {**shape, "kind": "forward_completed", "call_id": "failed", "success": False,
         "error": {"type": "RuntimeError", "message": "injected"}},
    ])
    assert coverage["observed_probe_shapes"] == coverage["observed_shapes"] == []
    assert coverage["tensor_shape_coverage"]["observed_shapes"] == []
    assert {"mode": "continuous", "phase": "prefill", "batch_size": 2} in coverage["missing_probe_shapes"]
    assert coverage["status"] == "FAIL"


def test_cpu_fake_never_claims_gpu_correctness():
    checks, coverage, outputs = preflight.assess(fake_config(), [],
        {"cuda_available": False}, scope="single")
    status = {item["code"]: item["status"] for item in checks}
    assert status["CUDA_AVAILABLE"] == "UNKNOWN"
    assert status["CONCURRENCY_2_ISOLATED"] == "UNKNOWN"
    assert status["ERROR_CLEANUP"] == "UNKNOWN"
    assert status["CANCELLATION_CLEANUP"] == "UNKNOWN"
    assert preflight.overall_status(checks) != "PASS"
    assert not outputs and coverage["status"] == "FAIL"


def test_preflight_source_change_invalidates_report(monkeypatch, tmp_path):
    snapshots = iter([{"sha": "a", "dirty": True, "source_tree_sha256": "before", "files": {}},
                      {"sha": "a", "dirty": True, "source_tree_sha256": "after", "files": {}}])
    monkeypatch.setattr(preflight, "git_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(preflight, "cuda_environment", lambda _config: {"cuda_available": False})
    monkeypatch.setattr(preflight, "prepare_trace", lambda _config: (_ for _ in ()).throw(RuntimeError("unavailable")))
    report = preflight.run_preflight(fake_config(), tmp_path / "report")
    assert report["status"] == "FAIL" and not report["source_unchanged"]
    assert report["error"]["message"] == "unavailable"
    assert (tmp_path / "report" / "preflight.json").exists()


def test_exact_token_signature_does_not_use_visible_text_length(monkeypatch, tmp_path):
    config = fake_config()
    item = config.data["requests"][0]
    from benchmarks.runtime.online_http.server import FakeTokenizer
    prompt = FakeTokenizer().apply_chat_template(item["messages"])
    event = {"kind": "terminal", "request_id": "engine-one", "prompt_sha256": preflight.prompt_hash(prompt),
             "status": "finished", "generation_config": {"max_tokens": item["max_tokens"]},
             "generated_token_ids": [101, 999999]}
    assert preflight._terminal_tokens([event], item, config) == [101, 999999]
    event["status"] = "cancelled"
    assert preflight._terminal_tokens([event], item, config) is None


def test_terminal_annotation_rejects_stale_missing_and_reused_request_states():
    config = fake_config()
    item = config.data["requests"][0]
    from benchmarks.runtime.online_http.server import FakeTokenizer
    prompt = FakeTokenizer().apply_chat_template(item["messages"])
    stale = {"kind": "terminal", "request_id": "stale", "prompt_sha256": preflight.prompt_hash(prompt),
             "status": "finished", "generation_config": {"max_tokens": item["max_tokens"]},
             "generated_token_ids": [999]}
    assert preflight._terminal_tokens([stale], item, config, since=1) is None
    fresh = [{**stale, "request_id": rid, "generated_token_ids": [index]}
             for index, rid in enumerate(("fresh-a", "fresh-b"))]
    events = [stale, *fresh]
    used = set()
    row_a, row_b, row_missing = {}, {}, {}
    assert preflight.annotate_tokens(row_a, events, item, config, "continuous", since=1, used_request_ids=used) == [0]
    assert preflight.annotate_tokens(row_b, events, item, config, "continuous", since=1, used_request_ids=used) == [1]
    assert preflight.annotate_tokens(row_missing, events, item, config, "continuous", since=1, used_request_ids=used) is None
    assert row_a["observed_engine_request_id"] != row_b["observed_engine_request_id"]
    assert row_missing["observed_generated_token_ids"] is None


def test_legacy_release_proof_requires_matching_cleared_cache_reference():
    config = fake_config()
    item = config.data["requests"][0]
    from benchmarks.runtime.online_http.server import FakeTokenizer
    prompt = FakeTokenizer().apply_chat_template(item["messages"])
    event = {"kind": "terminal", "request_id": "legacy-exact", "prompt_sha256": preflight.prompt_hash(prompt),
             "status": "finished", "generation_config": {"max_tokens": item["max_tokens"]},
             "generated_token_ids": [1]}
    row = {}
    preflight.annotate_tokens(row, [event], item, config, "legacy", since=0, used_request_ids=set())
    assert row["kv_release_confirmed"] is False
    released = {"kind": "legacy_runner_released", "request_id": "legacy-exact", "kv_reference_cleared": True}
    preflight.annotate_tokens(row, [event, released], item, config, "legacy", since=0, used_request_ids=set())
    assert row["kv_release_confirmed"] is True


def test_partial_preflight_cli_returns_failure_exit_status(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", ["preflight", "--config", str(CONFIG), "--execute",
                                     "--output-dir", str(tmp_path / "fake-result")])
    monkeypatch.setattr(preflight, "run_preflight", lambda *_a, **_k: {"status": "UNKNOWN"})
    with pytest.raises(SystemExit) as error:
        preflight.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "UNKNOWN"


def test_disconnect_probe_absolute_watchdog_handles_connection_close_dribbling():
    class Dribble(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for _ in range(200):
                    self.wfile.write(b"d")
                    self.wfile.flush()
                    time.sleep(.005)
            except (OSError, BrokenPipeError):
                pass

    data = fake_config().data
    data["request_timeout_s"] = .05
    config = ProtocolConfig.from_dict(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Dribble)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        started = time.perf_counter()
        result = preflight.disconnect_after_content(f"http://127.0.0.1:{server.server_port}",
                                                    config, data["requests"][0])
        assert time.perf_counter() - started < 1
        assert result["timed_out"] and result["error"]["code"] == "client_timeout"
        assert not result["partial_content_received"] and result["timeout_s"] == .05
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def test_single_cpu_real_tcp_preflight_preserves_unknowns_and_shutdown(tmp_path):
    directory = tmp_path / "preflight"
    result = preflight.run_preflight(fake_config(), directory, scope="single")
    assert len(result["modes"]) == 2
    assert result["performance_samples"] == 0 and result["status"] != "PASS"
    checks = {item["code"]: item for item in result["checks"]}
    assert checks["NORMAL_CLEANUP"]["status"] == "PASS", result
    assert checks["SHUTDOWN_CLEANUP"]["status"] == "UNKNOWN"
    assert checks["CONCURRENCY_2_ISOLATED"]["status"] == "UNKNOWN"
    for mode in result["modes"]:
        assert len(mode["reference"]) == 1
        assert mode["shutdown"]["stopped"], mode
        assert mode["shutdown"]["final"]["pending_finalizers"] == 0
        events = mode["events"]
        assert events[-1]["kind"] == "shutdown_complete"
        assert not events[-1]["owner_thread_alive"]
        assert events[-1]["scheduler"]["running_requests"] == 0
        assert events[-1]["scheduler"]["queued_requests"] == 0
        rows = [json.loads(line) for line in (directory / mode["mode"] / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows == list(mode["reference"].values()), "raw and summarized reference annotations must agree"
    artifact = json.loads((directory / "artifacts.json").read_text(encoding="utf-8"))
    assert "legacy/observer.jsonl" in artifact["sha256"]
    assert "continuous/requests.jsonl" in artifact["sha256"]
    before = (directory / "preflight.json").read_bytes()
    with pytest.raises(FileExistsError):
        preflight.run_preflight(fake_config(), directory)
    assert (directory / "preflight.json").read_bytes() == before


def test_full_cpu_probe_keeps_faults_outside_performance_and_observes_active_shutdown(tmp_path):
    result = preflight.run_preflight(fake_config(), tmp_path / "full", scope="full")
    assert result["status"] != "PASS" and result["performance_samples"] == 0
    for mode in result["modes"]:
        assert len(mode["reference"]) == 2 and len(mode["concurrent"]) == 2, mode
        assert mode["faults"]["shutdown_active_observed"], mode
        assert mode["faults"]["cancel"]["partial_content_received"], mode
        assert mode["faults"]["cancel"]["drain"]["idle_after_test"], mode
        assert mode["faults"]["error"]["injection_consumed"] is False  # Fake never invokes a model.
        assert mode["shutdown"]["stopped"]
        rows = [json.loads(line) for line in (tmp_path / "full" / mode["mode"] / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(row["performance_sample"] is False for row in rows)
        assert {row["probe"] for row in rows} == {"independent_reference", "concurrency_two",
                                                   "injected_forward_error", "active_shutdown"}
