"""Pure Fake evidence tests; no HTTP service, model, CUDA or GPU is executed."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.runtime.online_http.config import ProtocolConfig, load_config
from benchmarks.runtime.online_http.gate import (
    ENVIRONMENT_FIELDS, PREFLIGHT_CHECKS, evaluate_trial, load_preflight, required_batch_shapes,
    validate_admission, validate_preflight, verify_preflight_artifacts,
)
from benchmarks.runtime.online_http.workload import build_trace, trace_sha256


@pytest.fixture
def evidence():
    # A real_gpu label exercises the gate, but every observation below is Fake.
    data = load_config(Path("benchmarks/runtime/online_http/configs/qwen_protocol.json")).data
    data.update(stream=False, pairs=1)
    data["requests"] = data["requests"][:2]
    config = ProtocolConfig.from_dict(data)
    trace = build_trace(config)
    git = {"sha": "a" * 40, "dirty": True, "source_tree_sha256": "b" * 64}
    environment = {key: None for key in ENVIRONMENT_FIELDS}
    environment.update(python="3.12.9", versions={"torch": "2.11.0+cu128", "transformers": "5.19.0"},
        cuda_runtime="12.8", gpu="Fake GPU evidence only", model=data["model"],
        model_revision=data["model_revision"], tokenizer_revision=data["tokenizer_revision"],
        tokenizer_files_sha256={"tokenizer.json": "c" * 64}, tokenizer_class="FakeTokenizer",
        chat_template_sha256="d" * 64, dtype="float32", device="cuda",
        actual_model_dtype="torch.float32", actual_model_device="cuda:0")
    # Synthetic explicit positive proof exercises admission. It is not a claim
    # that real preflight currently observes every mask/cache tensor shape.
    tensor_shapes = [
        {"mode": shape["mode"], "phase": shape["phase"], "input_ids": [shape["batch_size"], 8],
         "attention_mask": [shape["batch_size"], 8], "cache_sequence_length": 0}
        if shape["phase"] == "prefill" else
        {"mode": shape["mode"], "phase": shape["phase"], "input_ids": [shape["batch_size"], 1],
         "attention_mask": [shape["batch_size"], 9], "cache_sequence_length": 8}
        for shape in required_batch_shapes(config)]
    coverage = {"status": "PASS", "required_shapes": required_batch_shapes(config),
                "observed_shapes": required_batch_shapes(config),
                "formal_warmup_coverage": True,
                "tensor_shape_coverage": {"status": "PASS", "required_shapes": deepcopy(tensor_shapes),
                    "observed_shapes": deepcopy(tensor_shapes), "evidence": ["formal-tensor-shapes.jsonl"]},
                "evidence": ["forward-shapes.jsonl"]}
    refs, rows = {}, []
    for ordinal, item in enumerate(trace):
        ref = {"output_sha256": hashlib.sha256(item["request_id"].encode()).hexdigest(),
            "visible_output_tokens": item["max_tokens"] - 1, "generated_output_tokens": item["max_tokens"],
            "prompt_tokens": 7 + ordinal, "finish_reason": "length"}
        refs[item["request_id"]] = ref
        arrival = item["arrival_s"]
        rows.append({**ref, "request_id": item["request_id"], "phase": "measured", "trace_ordinal": ordinal,
            "scheduled_at_s": arrival, "generator_scheduled_at_s": arrival + .001,
            "worker_started_at_s": arrival + .002, "body_started_at_s": arrival + .003,
            "body_ready_at_s": arrival + .004, "sent_at_s": arrival + .005,
            "first_content_at_s": None, "last_content_at_s": None, "terminal_at_s": arrival + .1,
            "success": True, "outcome": "completed", "http_status": 200,
            "error_code": None, "protocol_errors": [], "metric_validation_errors": [],
            "max_tokens": item["max_tokens"], "send_lateness_ms": 5.,
            "generated_count_source": "server_usage", "config_sha256": config.sha256,
            "trace_sha256": trace_sha256(trace), "git_sha": git["sha"], "git_dirty": True,
            "server_pid": 1234, "mode": "continuous", "saw_done": False})
    preflight = {"status": "PASS", "config_sha256": config.sha256, "trace_sha256": trace_sha256(trace),
        "git": deepcopy(git), "environment": deepcopy(environment),
        "checks": [{"code": code, "status": "PASS", "evidence": [f"checks/{code}.json"]}
                   for code in PREFLIGHT_CHECKS],
        "warmup_coverage": deepcopy(coverage), "expected_outputs": refs}
    runtime = {key: 0 for key in ("pending", "allocated_slots", "contexts", "claims", "released_ids",
        "finished_ids", "runtime_active_requests", "pending_commands", "stream_consumers", "pending_finalizers")}
    local_coverage = {**deepcopy(coverage), "required_shapes": required_batch_shapes(config, mode="continuous")}
    trial = {"mode": "continuous", "git": git, "warmup_coverage": local_coverage,
        "server_metadata": {**deepcopy(environment), "mode": "continuous", "config_sha256": config.sha256,
                            "git": deepcopy(git), "pid": 1234},
        "resources": {"after": {"idle_after_test": True, "status": {
            "running_requests": 0, "queued_requests": 0, "continuous": runtime}}},
        "shutdown": {"stopped": True, "returncode": 0,
            "final": {"owner_thread_alive": False, "pending_finalizers": 0, "stream_consumers": 0}}}
    artifacts = {"exclusive_directory": True, "history_preserved": True, "raw_journal_complete": True,
        "source_unchanged": True, "config_sha256": config.sha256, "trace_sha256": trace_sha256(trace),
        "source_tree_sha256": git["source_tree_sha256"], "evidence": ["source-manifest.json", "requests.jsonl"]}
    return {"config": config, "trace": trace, "rows": rows, "trial": trial,
            "preflight": preflight, "git": git, "environment": environment, "artifacts": artifacts}


def evaluate(evidence):
    return evaluate_trial(evidence["config"], evidence["trace"], evidence["rows"], evidence["trial"],
        preflight=evidence["preflight"], git=evidence["git"], environment=evidence["environment"],
        artifact_evidence=evidence["artifacts"])


def preflight(evidence):
    return validate_preflight(evidence["preflight"], config=evidence["config"],
        trace_sha256=trace_sha256(evidence["trace"]), git=evidence["git"], environment=evidence["environment"])


def test_complete_bound_fake_evidence_demonstrates_gate_and_is_not_a_gpu_run(evidence):
    result = evaluate(evidence)
    assert result["status"] == "PASS"
    assert result["reason_codes"] == []
    assert result["eligibility"]["formal_statistics"]
    # Pure function: no mutation, filtering, normalization or dropped records.
    prior = deepcopy(evidence)
    evaluate(evidence)
    assert evidence == prior


@pytest.mark.parametrize("field,value", [("config_sha256", "x" * 64), ("trace_sha256", "e" * 64),
    ("git.sha", "f" * 40), ("git.dirty", False), ("git.source_tree_sha256", "f" * 64)])
def test_preflight_must_bind_exact_config_trace_and_dirty_tree(evidence, field, value):
    if field.startswith("git."):
        evidence["preflight"]["git"][field.split(".")[1]] = value
    else:
        evidence["preflight"][field] = value
    assert "PREFLIGHT_IDENTITY_BOUND" in preflight(evidence)["reason_codes"]


@pytest.mark.parametrize("change", ["missing", "unknown", "failed", "no-evidence", "duplicate"])
def test_overall_pass_cannot_override_incomplete_per_check_proof(evidence, change):
    entry = evidence["preflight"]["checks"][0]
    if change == "missing":
        evidence["preflight"]["checks"].pop(0)
    elif change == "unknown":
        entry["status"] = "UNKNOWN"
    elif change == "failed":
        entry["status"] = "FAIL"
    elif change == "no-evidence":
        entry["evidence"] = []
    else:
        evidence["preflight"]["checks"].append(dict(entry))
    assert preflight(evidence)["status"] != "PASS"
    assert "PREFLIGHT_CUDA_AVAILABLE" in preflight(evidence)["reason_codes"]


def test_partial_preflight_never_admits_formal_concurrency(evidence):
    evidence["preflight"]["checks"] = evidence["preflight"]["checks"][:5]
    evidence["preflight"]["expected_outputs"].pop(evidence["trace"][1]["request_id"])
    result = validate_admission(evidence["config"], evidence["trace"], evidence["preflight"],
        evidence["git"], evidence["environment"])
    assert not result["admitted"]
    assert "PREFLIGHT_CONCURRENCY_2_ISOLATED" in result["reason_codes"]
    assert "ALL_OUTPUT_REFERENCES_FROZEN" in result["reason_codes"]


def test_preflight_environment_version_change_is_rejected(evidence):
    evidence["environment"]["versions"]["torch"] = "different"
    assert "PREFLIGHT_ENVIRONMENT_BOUND" in preflight(evidence)["reason_codes"]


def test_preflight_warmup_count_is_not_observed_shape_coverage(evidence):
    evidence["preflight"]["warmup_coverage"]["observed_shapes"].remove(
        {"mode": "continuous", "phase": "decode", "batch_size": 2})
    assert "PREFLIGHT_WARMUP_COVERAGE" in preflight(evidence)["reason_codes"]


def test_current_sse_protocol_cannot_claim_actual_generated_tokens(evidence):
    data = evidence["config"].data
    data["stream"] = True
    config = ProtocolConfig.from_dict(data)
    evidence["config"] = config
    evidence["preflight"]["config_sha256"] = config.sha256
    evidence["artifacts"]["config_sha256"] = config.sha256
    evidence["trial"]["server_metadata"]["config_sha256"] = config.sha256
    for row in evidence["rows"]:
        row.update(config_sha256=config.sha256, saw_done=True,
            first_content_at_s=row["sent_at_s"] + .01, last_content_at_s=row["terminal_at_s"] - .01,
            generated_output_tokens=None, generated_count_source="unavailable_sse")
    result = evaluate(evidence)
    assert result["status"] == "UNKNOWN"
    assert result["eligibility"]["http_visible_proxy"]
    assert not result["eligibility"]["formal_statistics"]
    assert "AUTHORITATIVE_OUTPUT_TOKENS_UNAVAILABLE" in result["reason_codes"]
    admission = validate_admission(config, evidence["trace"], evidence["preflight"],
        evidence["git"], evidence["environment"])
    assert not admission["admitted"]
    assert "AUTHORITATIVE_OUTPUT_TOKENS_UNAVAILABLE" in admission["reason_codes"]


def test_completion_order_not_confused_with_frozen_arrival_order(evidence):
    evidence["rows"].reverse()
    assert evaluate(evidence)["status"] == "PASS"
    evidence["rows"][0]["trace_ordinal"] = 0
    assert "RAW_IDENTITY_BOUND" in evaluate(evidence)["reason_codes"]


@pytest.mark.parametrize("field,value,code", [
    ("sent_at_s", -.001, "RAW_TIMELINE_VALID"),
    ("terminal_at_s", float("nan"), "RAW_TIMELINE_VALID"),
    ("body_ready_at_s", 99., "RAW_TIMELINE_VALID"),
    ("send_lateness_ms", 0, "RAW_TIMELINE_VALID"),
    ("generated_output_tokens", -1, "AUTHORITATIVE_OUTPUT_TOKENS_MISMATCH"),
    ("generated_output_tokens", 15, "AUTHORITATIVE_OUTPUT_TOKENS_MISMATCH"),
    ("visible_output_tokens", 100, "OUTPUT_REFERENCE_MATCH"),
    ("output_sha256", "f" * 64, "OUTPUT_REFERENCE_MATCH"),
    ("outcome", "timeout", "ALL_REQUESTS_COMPLETED"),
    ("protocol_errors", ["missing DONE"], "WIRE_PROTOCOL_VALID"),
])
def test_bad_raw_observations_are_retained_but_not_admitted(evidence, field, value, code):
    evidence["rows"][0][field] = value
    count = len(evidence["rows"])
    assert code in evaluate(evidence)["reason_codes"]
    assert len(evidence["rows"]) == count


def test_arrival_tolerance_is_exactly_25ms_and_cannot_be_relaxed(evidence):
    row = evidence["rows"][0]
    row.update(sent_at_s=.025001, send_lateness_ms=25.001)
    assert "CLIENT_SEND_ON_TIME" in evaluate(evidence)["reason_codes"]
    data = evidence["config"].data
    data["arrival_tolerance_ms"] = 30
    evidence["config"] = ProtocolConfig.from_dict(data)
    assert "FROZEN_ARRIVAL_TOLERANCE" in evaluate(evidence)["reason_codes"]


def test_missing_rows_fields_and_duplicate_ids_cannot_disappear(evidence):
    evidence["rows"][0].pop("output_sha256")
    evidence["rows"][1]["request_id"] = evidence["rows"][0]["request_id"]
    result = evaluate(evidence)
    assert "RAW_FIELDS_COMPLETE" in result["reason_codes"]
    assert "REQUEST_SET_COMPLETE" in result["reason_codes"]


@pytest.mark.parametrize("change", ["slot", "context", "owner", "finalizer", "shutdown", "warmup", "artifacts", "fingerprint"])
def test_cleanup_warmup_and_artifact_proof_are_required(evidence, change):
    trial = evidence["trial"]
    if change in {"slot", "context"}:
        key = "allocated_slots" if change == "slot" else "contexts"
        trial["resources"]["after"]["status"]["continuous"][key] = 1
    elif change == "owner":
        trial["shutdown"]["final"]["owner_thread_alive"] = True
    elif change == "finalizer":
        trial["shutdown"]["final"]["pending_finalizers"] = 1
    elif change == "shutdown":
        trial["shutdown"]["stopped"] = False
    elif change == "warmup":
        trial["warmup_coverage"]["observed_shapes"] = []
    elif change == "artifacts":
        evidence["artifacts"].pop("evidence")
    else:
        evidence["artifacts"]["source_tree_sha256"] = "d" * 64
    assert not evaluate(evidence)["eligibility"]["formal_statistics"]


def test_missing_preflight_cannot_be_replaced_by_boolean_flag(evidence):
    evidence["preflight"] = {"status": "PASS"}
    assert evaluate(evidence)["status"] != "PASS"
    evidence["preflight"] = None
    assert "PREFLIGHT_MISSING" in evaluate(evidence)["reason_codes"]


def test_load_report_checks_hash_and_preserves_original_bytes(tmp_path):
    path = tmp_path / "preflight.json"
    raw = json.dumps({"status": "UNKNOWN", "checks": []}).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert load_preflight(path, expected_sha256=digest)["sha256"] == digest
    with pytest.raises(ValueError, match="SHA-256"):
        load_preflight(path, expected_sha256="0" * 64)
    assert path.read_bytes() == raw


def test_load_report_rejects_nonfinite_json(tmp_path):
    path = tmp_path / "preflight.json"
    path.write_text('{"latency":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_preflight(path)


@pytest.fixture
def preflight_artifacts(tmp_path):
    config = load_config(Path("benchmarks/runtime/online_http/configs/qwen_protocol.json"))
    report = {"status": "UNKNOWN", "config_sha256": config.sha256,
              "git": {"sha": "b" * 40, "dirty": True, "source_tree_sha256": "c" * 64}}
    (tmp_path / "preflight.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps(config.as_dict()), encoding="utf-8")
    for mode in ("legacy", "continuous"):
        directory = tmp_path / mode
        directory.mkdir()
        for name in ("observer.jsonl", "requests.jsonl", "server-ready.json", "probe.json"):
            (directory / name).write_text('{}\n', encoding="utf-8")
    manifest = {"config_sha256": report["config_sha256"], "git": report["git"],
        "sha256": {file.relative_to(tmp_path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
                   for file in tmp_path.rglob("*") if file.is_file()}}
    (tmp_path / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_preflight_artifact_manifest_readback_valid(preflight_artifacts):
    result = verify_preflight_artifacts(preflight_artifacts / "preflight.json")
    assert result["code"] == "PREFLIGHT_ARTIFACT_INTEGRITY" and result["status"] == "PASS"


@pytest.mark.parametrize("change", ["missing-sidecar", "modified-sidecar", "missing-manifest", "missing-self", "wrong-source"])
def test_preflight_artifact_manifest_rejects_missing_or_changed_evidence(preflight_artifacts, change):
    directory = preflight_artifacts
    manifest_path = directory / "artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "missing-sidecar":
        (directory / "legacy" / "probe.json").unlink()
    elif change == "modified-sidecar":
        (directory / "continuous" / "observer.jsonl").write_text('changed\n', encoding="utf-8")
    elif change == "missing-manifest":
        manifest_path.unlink()
    elif change == "missing-self":
        manifest["sha256"].pop("preflight.json")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        manifest["git"]["source_tree_sha256"] = "d" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_preflight_artifacts(directory / "preflight.json")["status"] == "FAIL"


@pytest.mark.parametrize("path", ["../outside.json", "D:/outside.json", "/outside.json", "C:\\outside.json", "..\\outside.json"])
def test_preflight_manifest_paths_cannot_escape_report_directory(preflight_artifacts, path):
    manifest_path = preflight_artifacts / "artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"][path] = "a" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_preflight_artifacts(preflight_artifacts / "preflight.json")
    assert result["status"] == "FAIL" and "unsafe_manifest_path" in result["detail"]


def test_batch_shape_requirement_cannot_be_shrunk_to_unwarmed_subset(evidence):
    coverage = evidence["preflight"]["warmup_coverage"]
    coverage["required_shapes"] = coverage["observed_shapes"] = coverage["required_shapes"][:2]
    assert "PREFLIGHT_WARMUP_COVERAGE" in preflight(evidence)["reason_codes"]


def test_frozen_config_file_must_match_report_identity_even_with_fresh_manifest(preflight_artifacts):
    directory = preflight_artifacts
    path = directory / "config.json"
    data = json.loads(path.read_text())
    data["seed"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest_path = directory / "artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["config.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_preflight_artifacts(directory / "preflight.json")
    assert result["status"] == "FAIL" and "frozen_config_identity_mismatch" in result["detail"]


def test_raw_pid_cannot_be_borrowed_from_another_trial(evidence):
    evidence["rows"][0]["server_pid"] = 999
    assert "RAW_IDENTITY_BOUND" in evaluate(evidence)["reason_codes"]


@pytest.mark.parametrize("change", ["logical-only", "diagnostic-not-formal", "tensor-missing", "tensor-incomplete", "tensor-no-evidence"])
def test_logical_batch_cardinality_alone_cannot_attest_full_formal_warmup(evidence, change):
    coverage = evidence["preflight"]["warmup_coverage"]
    if change == "logical-only":
        coverage.pop("formal_warmup_coverage")
        coverage.pop("tensor_shape_coverage")
    elif change == "diagnostic-not-formal":
        coverage["formal_warmup_coverage"] = False
    elif change == "tensor-missing":
        coverage.pop("tensor_shape_coverage")
    elif change == "tensor-incomplete":
        coverage["tensor_shape_coverage"]["observed_shapes"].pop()
    else:
        coverage["tensor_shape_coverage"]["evidence"] = []
    result = preflight(evidence)
    assert "PREFLIGHT_WARMUP_COVERAGE" in result["reason_codes"]
    assert result["status"] != "PASS"


def test_each_formal_process_needs_its_own_tensor_shape_coverage(evidence):
    evidence["trial"]["warmup_coverage"].pop("tensor_shape_coverage")
    result = evaluate(evidence)
    assert "TRIAL_WARMUP_COVERAGE" in result["reason_codes"]
    assert not result["eligibility"]["formal_statistics"]
