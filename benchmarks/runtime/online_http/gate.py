"""Fail-closed, model-free admission rules for frozen online experiments.

This module does not execute requests, edit observations or choose samples. A
passing flag is never sufficient: preflight evidence must cover every required
check and bind to the same configuration, prepared trace, sources and machine.
Unknown generated-token counts remain unknown; visible text is not usage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

from .config import ProtocolConfig, canonical_json
from .workload import trace_sha256


PREFLIGHT_CHECKS = (
    "CUDA_AVAILABLE", "MODEL_ASSETS_PINNED", "MODEL_MODE_PARITY",
    "GREEDY_OUTPUT_CORRECT", "CONCURRENCY_1_CORRECT", "CONCURRENCY_2_ISOLATED",
    "NORMAL_CLEANUP", "ERROR_CLEANUP", "CANCELLATION_CLEANUP",
    "SHUTDOWN_CLEANUP", "WARMUP_SHAPES_COVERED",
)
ENVIRONMENT_FIELDS = (
    "python", "versions", "cuda_runtime", "gpu", "model", "model_revision",
    "tokenizer_revision", "tokenizer_files_sha256", "tokenizer_class",
    "chat_template_sha256", "dtype", "device", "actual_model_dtype",
    "actual_model_device",
)
RAW_REQUIRED_FIELDS = (
    "request_id", "phase", "trace_ordinal", "scheduled_at_s", "sent_at_s",
    "first_content_at_s", "last_content_at_s", "terminal_at_s", "success",
    "outcome", "http_status", "error_code", "protocol_errors", "finish_reason",
    "visible_output_tokens", "generated_output_tokens", "generated_count_source",
    "prompt_tokens", "max_tokens", "output_sha256", "send_lateness_ms",
    "config_sha256", "trace_sha256", "git_sha", "git_dirty", "server_pid",
    "mode", "saw_done",
)
STAGE_FIELDS = ("generator_scheduled_at_s", "worker_started_at_s", "body_started_at_s", "body_ready_at_s")
_STATUS = {"PASS", "FAIL", "UNKNOWN"}


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def authoritative_count(row, data):
    if not data["stream"]:
        return row.get("generated_count_source") == "server_usage" and _count(row.get("generated_output_tokens"))
    usage = row.get("authoritative_usage") or {}
    return (data.get("authoritative_sse_usage") is True and
        bool(row.get("response_id")) and bool(row.get("lease_id")) and bool(usage.get("engine_request_id")) and
        usage.get("count_source") == "terminal_request_state" and usage.get("status") in {"finished", "failed", "cancelled"} and
        row.get("generated_count_source") == "terminal_request_state_sse" and
        usage.get("object") == "minillm.benchmark.usage" and usage.get("id") == row.get("response_id") and
        usage.get("request_id") == row.get("lease_id") and usage.get("includes_eos") is True and
        usage.get("includes_stop_and_hidden_tokens") is True and
        _count(row.get("generated_output_tokens")) and
        usage.get("usage", {}).get("completion_tokens") == row.get("generated_output_tokens"))


def _digest(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _ordered(values: Sequence[Any]) -> bool:
    return all(_finite(value) for value in values) and all(
        left <= right + 1e-9 for left, right in zip(values, values[1:]))


def _evidence(value: Any) -> bool:
    """Evidence may be a relative artifact reference or structured observations."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_evidence(item) for item in value)
    return False


def _check(code: str, status: str, evidence: Any, detail: str = "") -> dict[str, Any]:
    return {"code": code, "status": status, "evidence": evidence, "detail": detail}


def _verdict(checks: list[dict[str, Any]]) -> dict[str, Any]:
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else (
        "UNKNOWN" if any(c["status"] != "PASS" for c in checks) else "PASS")
    return {"status": status, "checks": checks,
            "reason_codes": [c["code"] for c in checks if c["status"] != "PASS"]}


def environment_identity(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Stable runtime identity; deliberately excludes PID, timestamps and clocks."""
    return {key: environment.get(key) for key in ENVIRONMENT_FIELDS}


def load_preflight(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Read an immutable report, optionally checking its externally stored hash.

    File parsing is not admission. Call :func:`validate_preflight` with the
    experiment's expected identities before starting a real formal run.
    """
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("preflight report SHA-256 mismatch")
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("preflight report must be an object")
    return {"report": value, "sha256": digest, "path": str(path)}


def verify_preflight_artifacts(path: str | Path) -> dict[str, Any]:
    """Read back all hash-linked preflight artifacts before formal admission.

    This detects accidental omission, stale reports and modified sidecars. It
    is not a signed or trusted attestation: someone able to rewrite both files
    and their manifest can forge evidence. Paths must stay strictly inside the
    new report directory, including after resolving symlinks.
    """
    report_path = Path(path).resolve()
    root = report_path.parent
    manifest_path = root / "artifacts.json"
    problems = []
    try:
        report = load_preflight(report_path)["report"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or not isinstance(manifest.get("sha256"), Mapping):
            raise ValueError("artifact manifest lacks sha256 mapping")
        entries = manifest["sha256"]
        if not entries or len(entries) > 4096:
            raise ValueError("artifact manifest must contain 1 to 4096 files")
        for key in ("config_sha256", "git"):
            if manifest.get(key) != report.get(key):
                problems.append("artifacts.json." + key)
        required = {report_path.name, "config.json"}
        for mode in ("legacy", "continuous"):
            required.update(f"{mode}/{name}" for name in
                ("observer.jsonl", "requests.jsonl", "server-ready.json", "probe.json"))
        problems += ["missing_manifest_entry:" + name for name in sorted(required - set(entries))]
        for name, expected_digest in entries.items():
            if not isinstance(name, str):
                problems.append("nonstring_manifest_path")
                continue
            pure = PurePosixPath(name)
            if (not name or "\\" in name or pure.is_absolute() or PureWindowsPath(name).is_absolute()
                    or PureWindowsPath(name).drive or ".." in pure.parts or "." == name):
                problems.append("unsafe_manifest_path")
                continue
            candidate = (root / name).resolve()
            if not candidate.is_relative_to(root) or candidate == root:
                problems.append("escaped_manifest_path")
                continue
            if not candidate.is_file() or not _digest(expected_digest):
                problems.append("missing_or_invalid_digest:" + name)
                continue
            digest = hashlib.sha256()
            with candidate.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 64), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                problems.append("digest_mismatch:" + name)
        if "config.json" in entries:
            frozen_config = ProtocolConfig.from_dict(json.loads((root / "config.json").read_text(encoding="utf-8")))
            if frozen_config.sha256 != report.get("config_sha256"):
                problems.append("frozen_config_identity_mismatch")
        if (report.get("warmup_coverage") or {}).get("shape_proof") is not None:
            from .shapes import verify_bundle, IDENTITY
            expected = {"config_sha256": report["config_sha256"], "git": report["git"],
                **{k: frozen_config.data[k] for k in ("model_revision", "dtype", "device")}}
            replay = verify_bundle(root, expected_identity={k: expected[k] for k in IDENTITY})
            shape_manifest = json.loads((root / "expected-shapes.json").read_text(encoding="utf-8"))
            if shape_manifest.get("trace_sha256") != report.get("trace_sha256"):
                problems.append("shape_manifest_trace_mismatch")
            for name in ("expected-shapes.json", "shape-proof-index.json", "shape-coverage.json"):
                if name not in entries:
                    problems.append("missing_shape_artifact:" + name)
            if replay != report["warmup_coverage"]["shape_proof"]:
                problems.append("shape_proof_replay_mismatch")
    except (OSError, ValueError, TypeError) as exc:
        problems.append(type(exc).__name__ + ": " + str(exc))
    return _check("PREFLIGHT_ARTIFACT_INTEGRITY", "FAIL" if problems else "PASS",
                  ["preflight.json", "artifacts.json"],
                  "; ".join(problems) if problems else "Every required raw artifact is present and matches its recorded SHA-256.")


def _config(config):
    data = config.as_dict() if hasattr(config, "as_dict") else dict(config)
    digest = (config.sha256 if hasattr(config, "sha256") else
              hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest())
    return data, digest


def required_batch_shapes(config, *, mode: str | None = None) -> list[dict[str, Any]]:
    """Necessary logical row counts for frozen admission/batch capacity.

    Shapes describe actual model-forward batch dimensions, not the number of
    concurrent HTTP callers. These row counts alone are not complete tensor
    shapes: prompt padding, masks and decode-cache lengths also matter. Legacy
    forwards one request at a time per call.
    """
    data, _ = _config(config)
    shapes = []
    for current_mode in ("legacy", "continuous"):
        if mode is not None and current_mode != mode:
            continue
        largest = 1 if current_mode == "legacy" else data["admission_limit"]
        for batch_size in range(1, largest + 1):
            for phase in ("prefill", "decode"):
                shapes.append({"mode": current_mode, "phase": phase, "batch_size": batch_size})
    return shapes


def _coverage(value: Any, *, code: str, evidence: str, expected_shapes: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _check(code, "UNKNOWN", evidence, "No observed shape-coverage evidence.")
    proof = value.get("shape_proof")
    if isinstance(proof, Mapping) and proof.get("status") != "PASS":
        return _check(code, proof.get("status") if proof.get("status") in _STATUS else "UNKNOWN",
                      [evidence, "shape-coverage.json"], str(proof.get("reason_codes")))
    required, observed = value.get("required_shapes"), value.get("observed_shapes")
    if not isinstance(required, list) or not required or not isinstance(observed, list):
        return _check(code, "UNKNOWN", evidence, "Required/observed batch shapes are absent.")
    try:
        required_set = {canonical_json(shape) for shape in required}
        observed_set = {canonical_json(shape) for shape in observed}
    except (ValueError, TypeError):
        return _check(code, "FAIL", evidence, "Malformed shape records.")
    expected_set = {canonical_json(shape) for shape in expected_shapes}
    complete = required_set == expected_set and required_set <= observed_set
    statuses = ["PASS" if complete and value.get("status") == "PASS" and _evidence(value.get("evidence")) else "FAIL"]
    formal = value.get("formal_warmup_coverage")
    statuses.append("PASS" if formal is True else ("FAIL" if formal is False else "UNKNOWN"))
    tensor = value.get("tensor_shape_coverage")
    tensor_status = "UNKNOWN"
    if isinstance(tensor, Mapping):
        tensor_required, tensor_observed = tensor.get("required_shapes"), tensor.get("observed_shapes")
        if tensor.get("status") == "FAIL":
            tensor_status = "FAIL"
        elif (isinstance(tensor_required, list) and tensor_required and isinstance(tensor_observed, list)
                and tensor_observed and _evidence(tensor.get("evidence"))):
            try:
                tensor_complete = {canonical_json(shape) for shape in tensor_required} <= {
                    canonical_json(shape) for shape in tensor_observed}
                tensor_status = ("PASS" if tensor_complete and tensor.get("status") == "PASS" else "FAIL")
            except (ValueError, TypeError):
                tensor_status = "FAIL"
    statuses.append(tensor_status)
    status = "FAIL" if "FAIL" in statuses else ("UNKNOWN" if "UNKNOWN" in statuses else "PASS")
    return _check(code, status, evidence,
        "Logical mode/phase/batch cardinality is necessary, not full tensor-shape coverage. "
        "Require process-local formal warmup plus observed input/mask/cache-length tensor-shape coverage; "
        f"formal_attestation={formal!r}, tensor_coverage={tensor_status}.")


def validate_preflight(preflight: Any, *, config, trace_sha256: str,
                       git: Mapping[str, Any], environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Bind complete GPU preflight evidence to this exact formal experiment.

    An incomplete low-cost preflight is valid diagnostic evidence but is not
    permission to benchmark concurrency two or unobserved backend batch shapes.
    Dirty worktrees are permitted only when the dirty flag and source manifest
    fingerprint agree exactly; HEAD equality alone is insufficient.
    """
    data, config_digest = _config(config)
    if not isinstance(preflight, Mapping):
        return _verdict([_check("PREFLIGHT_MISSING", "UNKNOWN", "preflight report")])
    checks = []
    identity = {"config_sha256": config_digest, "trace_sha256": trace_sha256}
    mismatch = [key for key, value in identity.items() if preflight.get(key) != value]
    recorded_git = preflight.get("git") or {}
    if not isinstance(recorded_git, Mapping):
        recorded_git = {}
    for key in ("sha", "dirty", "source_tree_sha256"):
        if key not in git or key not in recorded_git or recorded_git.get(key) != git.get(key):
            mismatch.append("git." + key)
    if not (_digest(git.get("sha"), 40) and type(git.get("dirty")) is bool
            and _digest(git.get("source_tree_sha256")) and _digest(trace_sha256)):
        mismatch.append("invalid_expected_source_identity")
    if type(recorded_git.get("dirty")) is not bool:
        mismatch.append("invalid_recorded_dirty_flag")
    checks.append(_check("PREFLIGHT_IDENTITY_BOUND", "FAIL" if mismatch else "PASS",
                         ["preflight." + key for key in mismatch] or list(identity) + ["git"],
                         "Expected configuration, prepared trace and complete dirty source identity must agree."))
    checks.append(_check("PREFLIGHT_DECLARED_PASS", "PASS" if preflight.get("status") == "PASS" else "FAIL",
                         "preflight.status"))
    raw_checks = preflight.get("checks", [])
    if not isinstance(raw_checks, list):
        raw_checks = []
    for code in PREFLIGHT_CHECKS:
        entries = [item for item in raw_checks if isinstance(item, Mapping) and item.get("code") == code]
        if not entries:
            status = "UNKNOWN"
        elif len(entries) != 1:
            status = "FAIL"
        else:
            entry = entries[0]
            status = entry.get("status") if entry.get("status") in _STATUS else "FAIL"
            if status == "PASS" and not _evidence(entry.get("evidence")):
                status = "UNKNOWN"
        checks.append(_check("PREFLIGHT_" + code, status, "preflight.checks." + code,
                             "Each required check needs its own retained evidence."))
    extra_failures = [item.get("code") for item in raw_checks if isinstance(item, Mapping)
                      and item.get("status") in {"FAIL", "UNKNOWN"} and item.get("code") not in PREFLIGHT_CHECKS]
    checks.append(_check("PREFLIGHT_NO_OTHER_FAILURES", "FAIL" if extra_failures else "PASS",
                         extra_failures or "preflight.checks"))
    checks.append(_coverage(preflight.get("warmup_coverage"), code="PREFLIGHT_WARMUP_COVERAGE",
                            evidence="preflight.warmup_coverage", expected_shapes=required_batch_shapes(config)))
    if data.get("authoritative_sse_usage") is True:
        shape_proof = (preflight.get("warmup_coverage") or {}).get("shape_proof")
        current = isinstance(shape_proof, Mapping) and shape_proof.get("schema_version") == 1
        checks.append(_check("PREFLIGHT_SHAPE_PROOF_RECORDED",
            shape_proof.get("status", "UNKNOWN") if current else "UNKNOWN", "shape-proof-index.json",
            "Versioned raw shape proof must be independently replayed by artifact verification; legacy counters are insufficient."))
    recorded_env = preflight.get("environment")
    if not isinstance(recorded_env, Mapping):
        checks.append(_check("PREFLIGHT_ENVIRONMENT_BOUND", "UNKNOWN", "preflight.environment"))
    else:
        needed = ("python", "versions", "cuda_runtime", "gpu", "model_revision", "tokenizer_revision", "dtype", "device")
        missing = [key for key in needed if recorded_env.get(key) in (None, "", {})]
        versions = recorded_env.get("versions") or {}
        if not isinstance(versions, Mapping):
            versions = {}
        missing += ["versions." + key for key in ("torch", "transformers") if not versions.get(key)]
        wrong = [key for key in ("model_revision", "tokenizer_revision", "dtype", "device")
                 if recorded_env.get(key) != data[key]]
        if environment is None:
            status = "UNKNOWN"
            detail = "Current process environment must be compared, not just a preflight snapshot."
        else:
            wrong += [key for key in ENVIRONMENT_FIELDS
                      if environment_identity(recorded_env)[key] != environment_identity(environment)[key]]
            status = "FAIL" if wrong else ("UNKNOWN" if missing else "PASS")
            detail = f"Missing={missing}; differing={sorted(set(wrong))}."
        checks.append(_check("PREFLIGHT_ENVIRONMENT_BOUND", status, "preflight.environment", detail))
    return _verdict(checks)


def validate_admission(config, trace: Sequence[Mapping[str, Any]], preflight: Any,
                       git: Mapping[str, Any], environment: Mapping[str, Any] | None) -> dict[str, Any]:
    """Decide before launching formal requests, not after observing their speed.

    The existing SSE observer has no authoritative usage field. Consequently a
    correct GPU preflight alone cannot admit the current streaming protocol to
    actual generated-token performance statistics. This is a documented API
    observability gap, not an invitation to retokenize text as generated usage.
    """
    data, _ = _config(config)
    verdict = validate_preflight(preflight, config=config, trace_sha256=trace_sha256(list(trace)),
                                 git=git, environment=environment)
    checks = list(verdict["checks"])
    checks.append(_check("REAL_GPU_EVIDENCE", "PASS" if data["evidence_kind"] == "real_gpu" else "FAIL",
                         "config.evidence_kind"))
    checks.append(_check("FROZEN_ARRIVAL_TOLERANCE", "PASS" if data.get("arrival_tolerance_ms") == 25 else "FAIL",
                         "config.arrival_tolerance_ms", "The frozen threshold is exactly 25 ms."))
    refs = preflight.get("expected_outputs") if isinstance(preflight, Mapping) else None
    valid = isinstance(refs, Mapping) and set(refs) == {item["request_id"] for item in trace}
    if valid:
        for item in trace:
            reference = refs[item["request_id"]]
            if not isinstance(reference, Mapping):
                valid = False
                break
            count = reference.get("generated_output_tokens")
            if (not _count(count) or count > item["max_tokens"]
                    or not _count(reference.get("prompt_tokens"))
                    or not _count(reference.get("visible_output_tokens"))
                    or not _digest(reference.get("output_sha256"))
                    or reference.get("finish_reason") not in {"stop", "length"}
                    or (reference.get("finish_reason") == "length" and count != item["max_tokens"])):
                valid = False
                break
    checks.append(_check("ALL_OUTPUT_REFERENCES_FROZEN", "PASS" if valid else "UNKNOWN",
                         "preflight.expected_outputs", "Correctness references are required for every workload request."))
    capable = not data["stream"]
    if data["stream"] and isinstance(preflight, Mapping):
        modes = preflight.get("modes", [])
        capable = len(modes) == 2 and all(set(mode.get("reference", {})) == {i["request_id"] for i in trace}
            and all(authoritative_count(row, data) for row in mode["reference"].values()) for mode in modes)
    checks.append(_check("AUTHORITATIVE_OUTPUT_TOKENS_CAPABILITY" if capable else "AUTHORITATIVE_OUTPUT_TOKENS_UNAVAILABLE",
                         "PASS" if capable else "FAIL", "preflight.modes.reference.authoritative_usage",
                         "Every streamed reference needs identity-bound terminal-counter metadata, not visible retokenization."))
    result = _verdict(checks)
    result["admitted"] = result["status"] == "PASS"
    result["scope"] = "formal real-GPU online performance; CPU/Fake gate tests are not admitted samples"
    return result


def _zero_resources(trial: Mapping[str, Any]) -> bool:
    after = (trial.get("resources") or {}).get("after") or {}
    status = after.get("status") or {}
    if after.get("idle_after_test") is not True or status.get("running_requests") != 0 or status.get("queued_requests") != 0:
        return False
    if trial.get("mode") == "continuous":
        runtime = status.get("continuous") or {}
        names = ("pending", "allocated_slots", "contexts", "claims", "released_ids", "finished_ids",
                 "runtime_active_requests", "pending_commands", "stream_consumers", "pending_finalizers")
        if any(runtime.get(name) != 0 for name in names):
            return False
    shutdown = trial.get("shutdown") or {}
    final = shutdown.get("final") or {}
    return (shutdown.get("stopped") is True and shutdown.get("returncode", 0) == 0
            and final.get("owner_thread_alive") is False and final.get("pending_finalizers") == 0
            and final.get("stream_consumers") == 0)


def _server_provenance(config, trial, git, environment):
    data, digest = _config(config)
    meta = trial.get("server_metadata")
    if not isinstance(meta, Mapping):
        return _check("SERVER_MODEL_PROVENANCE", "UNKNOWN", "trial.server_metadata")
    wrong = []
    for key in ("model", "model_revision", "tokenizer_revision", "dtype", "device"):
        if meta.get(key) != data[key]:
            wrong.append(key)
    if meta.get("config_sha256") != digest or meta.get("mode") != trial.get("mode"):
        wrong.append("config_or_mode")
    meta_git = meta.get("git") if isinstance(meta.get("git"), Mapping) else {}
    if any(meta_git.get(key) != git.get(key) for key in ("sha", "dirty", "source_tree_sha256")):
        wrong.append("source_identity")
    if meta.get("actual_model_dtype") != "torch." + data["dtype"]:
        wrong.append("actual_model_dtype")
    actual_device = str(meta.get("actual_model_device", "")).split(":")[0]
    if actual_device != data["device"]:
        wrong.append("actual_model_device")
    files = meta.get("tokenizer_files_sha256")
    if not isinstance(files, Mapping) or not files or not all(_digest(value) for value in files.values()):
        wrong.append("tokenizer_files_sha256")
    if not _digest(meta.get("chat_template_sha256")) or not meta.get("tokenizer_class"):
        wrong.append("tokenizer_identity")
    if not meta.get("gpu") or not meta.get("cuda_runtime"):
        wrong.append("actual_cuda")
    if environment is None:
        return _check("SERVER_MODEL_PROVENANCE", "UNKNOWN", "trial.server_metadata",
                      "Control-process environment is unavailable.")
    for key in ("python", "versions", "cuda_runtime", "gpu"):
        if meta.get(key) != environment.get(key):
            wrong.append(key)
    return _check("SERVER_MODEL_PROVENANCE", "FAIL" if wrong else "PASS",
                  ["trial.server_metadata." + key for key in wrong] or "trial.server_metadata")


def evaluate_trial(config, trace: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
                   trial: Mapping[str, Any], *, preflight: Any = None,
                   git: Mapping[str, Any] | None = None, environment: Mapping[str, Any] | None = None,
                   artifact_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return diagnostic and separate formal/proxy eligibility without filtering.

    ``status`` describes strict formal admission, including authoritative output
    token accounting. A proxy-eligible stream is not thereby an eligible actual
    generated-token throughput experiment. CPU/Fake runs never establish GPU
    admission even if all tooling checks pass.
    """
    data, config_digest = _config(config)
    trace_digest = trace_sha256(list(trace))
    git = git or trial.get("git") or {}
    checks = []
    real = data["evidence_kind"] == "real_gpu"
    checks.append(_check("REAL_GPU_EVIDENCE", "PASS" if real else "FAIL", "config.evidence_kind",
                         "CPU/Fake records test the gate; they are not formal GPU samples."))
    checks.append(_check("FROZEN_ARRIVAL_TOLERANCE", "PASS" if data.get("arrival_tolerance_ms") == 25 else "FAIL",
                         "config.arrival_tolerance_ms", "Mission 24.4 freezes 25 ms; results cannot change it."))
    preflight_verdict = validate_preflight(preflight, config=config, trace_sha256=trace_digest,
        git=git, environment=environment)
    checks.extend(preflight_verdict["checks"])
    checks.append(_server_provenance(config, trial, git, environment))

    fields_missing, identity_errors, timing_errors, late, failed, protocol = [], [], [], [], [], []
    expected_ids = [item["request_id"] for item in trace]
    actual_ids = [row.get("request_id") for row in rows]
    coverage_ok = (all(isinstance(value, str) for value in actual_ids) and len(rows) == len(trace)
                   and len(set(str(value) for value in actual_ids)) == len(rows)
                   and set(str(value) for value in actual_ids) == set(expected_ids))
    checks.append(_check("REQUEST_SET_COMPLETE", "PASS" if coverage_ok else "FAIL", "requests.jsonl",
                         "Every planned ID appears exactly once; JSONL completion order is not arrival order."))
    items = {item["request_id"]: (ordinal, item) for ordinal, item in enumerate(trace)}
    refs = preflight.get("expected_outputs", {}) if isinstance(preflight, Mapping) else {}
    reference_duplicate = False
    if isinstance(refs, list):
        reference_ids = [item.get("request_id") for item in refs if isinstance(item, Mapping)]
        reference_duplicate = len(set(str(value) for value in reference_ids)) != len(reference_ids)
        refs = {item.get("request_id"): item for item in refs if isinstance(item, Mapping)}
    reference_ids = set(refs) if isinstance(refs, Mapping) else set()
    checks.append(_check("OUTPUT_REFERENCE_SET_COMPLETE",
        "FAIL" if reference_duplicate else ("PASS" if reference_ids == set(expected_ids) else "UNKNOWN"),
        "preflight.expected_outputs", "Every frozen request needs a retained correctness reference."))
    output_unknown, output_mismatch, actual_unknown, actual_mismatch = [], [], [], []
    for index, row in enumerate(rows):
        label = f"requests.jsonl[{index}]"
        fields_missing += [label + "." + field for field in RAW_REQUIRED_FIELDS if field not in row]
        rid = row.get("request_id")
        expected = items.get(rid) if isinstance(rid, str) else None
        if expected is None:
            continue
        ordinal, item = expected
        expected_values = {"phase": "measured", "trace_ordinal": ordinal, "scheduled_at_s": item["arrival_s"],
            "max_tokens": item["max_tokens"], "config_sha256": config_digest, "trace_sha256": trace_digest,
            "git_sha": git.get("sha"), "git_dirty": git.get("dirty"), "mode": trial.get("mode")}
        for key, value in expected_values.items():
            if value is None or row.get(key) != value:
                identity_errors.append(label + "." + key)
        if type(row.get("git_dirty")) is not bool or type(row.get("trace_ordinal")) is not int:
            identity_errors.append(label + ".identity_type")
        if type(row.get("server_pid")) is not int or row.get("server_pid", 0) <= 0:
            identity_errors.append(label + ".server_pid")
        elif isinstance(trial.get("server_metadata"), Mapping) and row["server_pid"] != trial["server_metadata"].get("pid"):
            identity_errors.append(label + ".server_pid")
        if not (row.get("success") is True and row.get("outcome") == "completed" and row.get("http_status") == 200
                and row.get("error_code") is None):
            failed.append(label)
        if row.get("protocol_errors") != [] or row.get("metric_validation_errors"):
            protocol.append(label)
        if data["stream"] and row.get("saw_done") is not True:
            protocol.append(label + ".saw_done")
        if row.get("finish_reason") not in {"stop", "length"}:
            protocol.append(label + ".finish_reason")
        stamps = [row.get("sent_at_s"), row.get("terminal_at_s")]
        if not _ordered(stamps) or (stamps[0] is not None and _finite(stamps[0]) and stamps[0] + 1e-9 < item["arrival_s"]):
            timing_errors.append(label + ".send_terminal")
        if data["stream"] and _count(row.get("visible_output_tokens")) and row["visible_output_tokens"] > 0:
            stamps = [row.get(key) for key in ("sent_at_s", "first_content_at_s", "last_content_at_s", "terminal_at_s")]
            if not _ordered(stamps):
                timing_errors.append(label + ".content_timeline")
        stages = [row.get(key) for key in STAGE_FIELDS] + [row.get("sent_at_s")]
        if not _ordered(stages):
            timing_errors.append(label + ".generator_timeline")
        if _finite(row.get("sent_at_s")) and _finite(row.get("send_lateness_ms")):
            observed_late = (row["sent_at_s"] - item["arrival_s"]) * 1000
            if not math.isclose(row["send_lateness_ms"], observed_late, abs_tol=1e-6):
                timing_errors.append(label + ".send_lateness_ms")
            if observed_late > 25:
                late.append(label)
        else:
            timing_errors.append(label + ".send_lateness_ms")
        reference = refs.get(rid) if isinstance(refs, Mapping) else None
        if (not isinstance(reference, Mapping) or not _digest(reference.get("output_sha256"))
                or not _count(reference.get("visible_output_tokens")) or not _count(reference.get("prompt_tokens"))
                or reference.get("finish_reason") not in {"stop", "length"}):
            output_unknown.append(label)
        elif any(row.get(key) != reference[key] for key in
                 ("output_sha256", "visible_output_tokens", "prompt_tokens", "finish_reason")):
            output_mismatch.append(label)
        count = row.get("generated_output_tokens")
        expected_count = reference.get("generated_output_tokens") if isinstance(reference, Mapping) else None
        if count is not None and not _count(count):
            actual_mismatch.append(label)
        elif count is None or not _count(expected_count) or not authoritative_count(row, data):
            actual_unknown.append(label)
        elif count != expected_count or count > item["max_tokens"] or (row.get("finish_reason") == "length" and count != item["max_tokens"]):
            actual_mismatch.append(label)
    for code, failures in (("RAW_FIELDS_COMPLETE", fields_missing), ("RAW_IDENTITY_BOUND", identity_errors),
                           ("RAW_TIMELINE_VALID", timing_errors), ("CLIENT_SEND_ON_TIME", late),
                           ("ALL_REQUESTS_COMPLETED", failed), ("WIRE_PROTOCOL_VALID", protocol)):
        checks.append(_check(code, "FAIL" if failures else "PASS", failures or "requests.jsonl"))
    checks.append(_check("OUTPUT_REFERENCE_MATCH", "FAIL" if output_mismatch else ("UNKNOWN" if output_unknown else "PASS"),
                         output_mismatch + output_unknown or "preflight.expected_outputs"))
    checks.append(_coverage(trial.get("warmup_coverage"), code="TRIAL_WARMUP_COVERAGE", evidence="trial.warmup_coverage",
                            expected_shapes=required_batch_shapes(config, mode=trial.get("mode"))))
    checks.append(_check("FINAL_RESOURCES_ZERO", "PASS" if _zero_resources(trial) else "FAIL",
                         ["trial.resources.after", "trial.shutdown"]))
    artifact_evidence = artifact_evidence or {}
    exclusive = (artifact_evidence.get("exclusive_directory") is True
        and artifact_evidence.get("raw_journal_complete") is True
        and artifact_evidence.get("history_preserved") is True
        and artifact_evidence.get("source_unchanged") is True
        and artifact_evidence.get("config_sha256") == config_digest
        and artifact_evidence.get("trace_sha256") == trace_digest
        and artifact_evidence.get("source_tree_sha256") == git.get("source_tree_sha256")
        and _evidence(artifact_evidence.get("evidence")))
    checks.append(_check("APPEND_ONLY_ARTIFACTS", "PASS" if exclusive else "UNKNOWN",
                         artifact_evidence or "artifact creation and source manifest evidence required"))
    base_pass = all(check["status"] == "PASS" for check in checks)
    token_code = ("AUTHORITATIVE_OUTPUT_TOKENS_MISMATCH" if actual_mismatch else
                  "AUTHORITATIVE_OUTPUT_TOKENS_UNAVAILABLE" if actual_unknown else "AUTHORITATIVE_OUTPUT_TOKENS")
    checks.append(_check(token_code, "FAIL" if actual_mismatch else ("UNKNOWN" if actual_unknown else "PASS"),
                         actual_mismatch + actual_unknown or "requests.jsonl.generated_output_tokens",
                         "Retokenized visible text and chunk counts never replace server generated-token usage."))
    verdict = _verdict(checks)
    verdict["eligibility"] = {"formal_statistics": verdict["status"] == "PASS",
        "actual_generated_throughput": verdict["status"] == "PASS",
        "http_visible_proxy": base_pass,
        "scope": "GPU correctness/proxy observations only; actual generated throughput requires authoritative per-request usage"}
    return verdict
