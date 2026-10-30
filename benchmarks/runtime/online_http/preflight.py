"""Bounded correctness probes, never a formal HTTP performance experiment.

Default: print a plan. ``--execute --allow-gpu --scope single`` performs only
one frozen request in each fresh service process. Unexecuted isolation/fault
checks remain UNKNOWN and cannot unlock the formal benchmark validity gate.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit

from .artifacts import Journal, environment_snapshot, git_snapshot, gpu_environment_snapshot, hardware_snapshot, utc_now, write_new
from .client import get_json, observe_request
from .config import load_config
from .preflight_server import prompt_hash
from .runner import ServerProcess, drain, prepare_trace
from .workload import trace_sha256


REQUIRED_CHECKS = (
    "CUDA_AVAILABLE", "MODEL_ASSETS_PINNED", "MODEL_MODE_PARITY", "GREEDY_OUTPUT_CORRECT",
    "CONCURRENCY_1_CORRECT", "CONCURRENCY_2_ISOLATED", "NORMAL_CLEANUP", "ERROR_CLEANUP",
    "CANCELLATION_CLEANUP", "SHUTDOWN_CLEANUP", "WARMUP_SHAPES_COVERED",
)


def overall_status(checks):
    """No silent skips: every required check must explicitly PASS."""
    indexed = {check["code"]: check["status"] for check in checks}
    if any(value == "FAIL" for value in indexed.values()):
        return "FAIL"
    return "PASS" if all(indexed.get(code) == "PASS" for code in REQUIRED_CHECKS) else "UNKNOWN"


def read_events(directory):
    path = Path(directory) / "observer.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def cuda_environment(config):
    """Query capability only; no model loading, benchmark or fallback."""
    result = {**environment_snapshot(), **{key: config.data[key] for key in
        ("device", "dtype", "model_revision", "tokenizer_revision")},
        "cuda_runtime": None, "gpu": None, "cuda_available": False}
    if config.data["evidence_kind"] == "real_gpu":
        result = gpu_environment_snapshot(config)
    return result


def required_shapes(config):
    """Logical model row counts, not an assertion about observed forward calls."""
    limit = config.data["admission_limit"]
    return [{"mode": mode, "phase": phase, "batch_size": size}
            for mode in ("legacy", "continuous") for phase in ("prefill", "decode")
            for size in range(1, (limit if mode == "continuous" else 1) + 1)]


def warmup_coverage(config, evidence):
    """The frozen formal warmup is sequential; probes are not formal warmup.

    Successful diagnostic B=2 forwards must never be substituted for each new
    formal process's actual warmup. The existing runner's sequential warmup
    cannot cover Continuous B=2, regardless of two warmup requests being sent.
    Exact padded widths and decode-cache lengths remain separately observable.
    """
    required = required_shapes(config)
    if config.data.get("warmup_strategy") == "sequential_and_pairs":
        evidence = [row for row in evidence if row.get("execution_stage") == "warmup"]
    observed = sorted({(row["mode"], row["phase"], row["input_shape"][0])
        for row in evidence if row.get("kind") == "forward_completed" and
        row.get("success") is True and row.get("input_shape")})
    observed = [{"mode": mode, "phase": phase, "batch_size": size} for mode, phase, size in observed]
    tensor_shapes = [json.loads(value) for value in sorted({json.dumps({
        "mode": row["mode"], "phase": row["phase"], "input_shape": row["input_shape"],
        "attention_mask_shape": row.get("attention_mask_shape"), "kv_length_before": row.get("kv_length_before"),
        "kv_tensor_shapes_before": row.get("kv_tensor_shapes_before")}, sort_keys=True)
        for row in evidence if row.get("kind") == "forward_completed" and
        row.get("success") is True and row.get("input_shape")})]
    missing = [shape for shape in required if shape not in observed]
    # Observed diagnostic requests are never relabelled as warmup of a future
    # fresh process. B=2 is a proven omission in the current formal strategy;
    # B=1 still needs future process-local attestation, not a guessed PASS.
    strategy = config.data.get("warmup_strategy", "sequential")
    known_gap = bool(missing) or (strategy == "sequential" and
        (config.data["admission_limit"] > 1 or config.data["warmup_requests"] == 0))
    return {"status": "FAIL" if known_gap else "UNKNOWN", "required_shapes": required,
        "observed_shapes": observed, "missing_probe_shapes": missing,
        "observed_probe_shapes": observed,
        "formal_warmup_coverage": False, "formal_warmup_strategy": strategy,
        "tensor_shape_coverage": {"status": "UNKNOWN", "required_shapes": [],
            "observed_shapes": tensor_shapes,
            "evidence": ["Formal padded widths / per-request KV lengths not exhaustively enumerated or process-locally warmed."]},
        "evidence": ["Observer shapes include diagnostic probes, not formal warmup.",
            "Fresh formal processes must independently attest warmup coverage.",
            "The selected recipe must be attested again in each future formal process.",
            "Only successful Python forward returns count; failed attempts do not.",
            "Input-width phase inference is ambiguous for one-token prompts; frozen chat prompts exceed one token."]}


def _signature(row):
    return {key: row.get(key) for key in ("output_sha256", "visible_output_tokens",
        "generated_output_tokens", "prompt_tokens", "finish_reason")}


def _terminal_record(events, item, config, *, since=0, used_request_ids=None):
    from .server import FakeTokenizer, pinned_assets
    if config.data["evidence_kind"] == "cpu_fake":
        tokenizer = FakeTokenizer()
    else:
        from transformers import AutoTokenizer
        _, path, _ = pinned_assets(config.data)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    prompt = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
    used_request_ids = used_request_ids if used_request_ids is not None else set()
    matches = [event for event in events[since:] if event.get("kind") == "terminal" and
        event.get("prompt_sha256") == prompt_hash(prompt) and event.get("status") == "finished" and
        event.get("generation_config", {}).get("max_tokens") == item["max_tokens"] and
        event.get("request_id") and event["request_id"] not in used_request_ids]
    return matches[0] if matches else None


def _terminal_tokens(events, item, config, *, since=0, used_request_ids=None):
    record = _terminal_record(events, item, config, since=since, used_request_ids=used_request_ids)
    return record["generated_token_ids"] if record is not None else None


def annotate_tokens(row, events, item, config, mode, *, since, used_request_ids):
    """Bind raw row to one fresh terminal state; never reuse an earlier probe."""
    record = _terminal_record(events, item, config, since=since, used_request_ids=used_request_ids)
    if record is None:
        row.update(observed_generated_token_ids=None, observed_engine_request_id=None,
                   kv_release_confirmed=None)
        return None
    rid = record["request_id"]
    used_request_ids.add(rid)
    ids = record["generated_token_ids"]
    source = row.get("generated_count_source")
    if source == "terminal_request_state_sse" and row.get("generated_output_tokens") != len(ids):
        row["protocol_errors"].append("wire usage differs from terminal token IDs")
        row.update(success=False, outcome="error")
    row.update(observed_generated_token_ids=ids, observed_engine_request_id=rid,
        generated_output_tokens=len(ids), generated_count_source=source if source == "terminal_request_state_sse" else "instrumented_terminal_request_state",
        kv_release_confirmed=any(event.get("kind") == "legacy_runner_released" and
            event.get("request_id") == rid and event.get("kv_reference_cleared") is True
            for event in events[since:]) if mode == "legacy" else None)
    return ids


def disconnect_after_content(base_url, config, item):
    """Close after content, with one absolute budget even for byte dribbling."""
    address = urlsplit(base_url)
    data = config.data
    budget = min(30, data["request_timeout_s"])
    connection = http.client.HTTPConnection(address.hostname, address.port,
                                          timeout=budget)
    body = {"model": data["model"], "messages": item["messages"], "max_tokens": 256, "stream": True,
            **{key: data[key] for key in ("temperature", "top_p", "top_k", "seed", "stop")}}
    received = False
    response_id = lease_id = None
    response = transport = None
    expired = threading.Event()
    started = time.perf_counter()
    deadline = started + budget
    error = None

    def remaining():
        seconds = deadline - time.perf_counter()
        if seconds <= 0 or expired.is_set():
            raise TimeoutError("absolute disconnect probe deadline exceeded")
        sock = connection.sock or transport
        if sock is not None:
            sock.settimeout(seconds)

    def abort():
        expired.set()
        sock = connection.sock or transport
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    watchdog = threading.Timer(budget, abort)
    watchdog.daemon = True
    watchdog.start()
    try:
        remaining()
        connection.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                           {"Content-Type": "application/json"})
        # getresponse() may clear connection.sock for Connection: close even
        # while response.readline still owns and uses the transport socket.
        transport = connection.sock
        remaining()
        response = connection.getresponse()
        lease_id = response.getheader("X-MiniLLM-Request-ID")
        for _ in range(512):
            remaining()
            line = response.readline(1048576)
            remaining()
            if not line:
                break
            if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                event = json.loads(line[6:])
                response_id = event.get("id", response_id)
                if event.get("choices", [{}])[0].get("delta", {}).get("content"):
                    received = True
                    break
    except Exception as exc:
        error = {"code": "client_timeout" if expired.is_set() or isinstance(exc, (TimeoutError, socket.timeout))
                 else type(exc).__name__, "message": str(exc)}
    finally:
        watchdog.cancel()
        sock = connection.sock or transport
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if response is not None:
            response.close()
        connection.close()
    return {"partial_content_received": received, "generation_limit_for_fault_probe": 256,
            "response_id": response_id, "lease_id": lease_id,
            "timeout_s": budget, "elapsed_s": time.perf_counter() - started,
            "timed_out": bool(error and error["code"] == "client_timeout"), "error": error}


def probe_mode(config, trace, token_count, mode, directory, config_path, git, *, scope="single"):
    """Fresh production HTTP service; no formal timing/arrival classification."""
    directory.mkdir(exist_ok=False)
    server = ServerProcess(config, mode, directory, config_path, git,
        host_module="benchmarks.runtime.online_http.preflight_server")
    result = {"mode": mode, "reference": {}, "concurrent": [], "cleanup": [], "faults": {},
              "metadata": None, "shutdown": None, "scope": scope}
    journal = Journal(directory / "requests.jsonl")
    try:
        server.wait_ready()
        result["metadata"] = server.metadata
        if config.data.get("warmup_strategy") == "sequential_and_pairs":
            from .warmup import run_warmup
            result["process_warmup"] = run_warmup(config, trace, token_count, server, directory, journal, drain)
        for item in trace[:1] if scope == "single" else trace:
            before_events = len(read_events(directory))
            row = observe_request(server.base_url, config, {**item, "arrival_s": 0}, time.perf_counter(), token_count)
            row.update(probe="independent_reference", performance_sample=False, mode=mode)
            ended = drain(server.base_url, config, mode=mode)
            result["cleanup"].append(ended)
            events = read_events(directory)
            annotate_tokens(row, events, item, config, mode, since=before_events, used_request_ids=set())
            if mode == "continuous":
                row["kv_release_confirmed"] = ended["idle_after_test"]
            journal.append(row)
            result["reference"][item["request_id"]] = row
        if scope == "full":
            if config.data["admission_limit"] < 2:
                result["faults"]["concurrency"] = {"status": "UNKNOWN", "reason": "admission_limit_less_than_two"}
            else:
                for offset in range(0, len(trace), 2):
                    group = trace[offset:offset + 2]
                    if len(group) != 2:
                        group = [group[0], {**group[0], "request_id": group[0]["request_id"] + "-isolation-copy"}]
                    before_events = len(read_events(directory))
                    origin = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        rows = list(pool.map(lambda item: observe_request(server.base_url, config,
                            {**item, "arrival_s": 0}, origin, token_count), group))
                    ended = drain(server.base_url, config, mode=mode)
                    result["cleanup"].append(ended)
                    events = read_events(directory)
                    used_ids = set()
                    for item, row in zip(group, rows):
                        ids = annotate_tokens(row, events, item, config, mode, since=before_events,
                                              used_request_ids=used_ids)
                        if mode == "continuous":
                            row["kv_release_confirmed"] = ended["idle_after_test"]
                        reference = result["reference"][trace[offset]["request_id"]] if item["request_id"].endswith("-isolation-copy") else result["reference"][item["request_id"]]
                        row.update(probe="concurrency_two", performance_sample=False, mode=mode,
                            matches_independent_reference=_signature(row) == _signature(reference) and
                                ids is not None and ids == reference.get("observed_generated_token_ids"))
                        journal.append(row)
                    result["concurrent"].extend(rows)
            # Deliberate faults are outside reference samples and never count
            # as performance observations or expected successful requests.
            with (directory / "fault-active").open("x") as phase:
                phase.write("fault probes; never formal or warmup\n")
            before = get_json(server.base_url, "/scheduler/status")
            flag = directory / "inject-next-forward"
            with flag.open("x", encoding="utf-8") as output:
                output.write("preflight fault injection only\n")
            row = observe_request(server.base_url, config, {**trace[0], "arrival_s": 0},
                                  time.perf_counter(), token_count)
            row.update(probe="injected_forward_error", performance_sample=False, mode=mode)
            journal.append(row)
            ended = drain(server.base_url, config, mode=mode)
            result["faults"]["error"] = {"row": row, "drain": ended,
                "injection_consumed": not flag.exists(), "before": before}
            before = get_json(server.base_url, "/scheduler/status")
            cancel_start = len(read_events(directory))
            cancelled = disconnect_after_content(server.base_url, config, trace[0])
            ended = drain(server.base_url, config, mode=mode)
            result["faults"]["cancel"] = {**cancelled, "drain": ended,
                "terminal_counters": [{"engine_request_id": e["request_id"],
                    "generated_tokens": len(e["generated_token_ids"]), "status": e["status"],
                    "delivery": "out_of_band_after_client_disconnect"}
                    for e in read_events(directory)[cancel_start:] if e.get("kind") == "terminal"],
                "cancelled_delta": (ended.get("status") or {}).get("cancelled_requests", 0) - before["cancelled_requests"]}
            # Active shutdown is separate from passive normal closure. Wait for
            # observed running work rather than assuming a background Future
            # means the server admitted it.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(observe_request, server.base_url, config,
                    {**trace[0], "arrival_s": 0, "max_tokens": 256}, time.perf_counter(), token_count)
                deadline = time.monotonic() + min(10, config.data["request_timeout_s"])
                active = None
                while time.monotonic() < deadline and not future.done():
                    active = get_json(server.base_url, "/scheduler/status")
                    if active.get("running_requests", 0) > 0:
                        break
                    time.sleep(.01)
                result["faults"]["shutdown_active_observed"] = bool(active and active.get("running_requests", 0))
                result["shutdown"] = server.close()
                row = future.result()
                row.update(probe="active_shutdown", performance_sample=False, mode=mode)
                journal.append(row)
                result["faults"]["shutdown_response"] = row
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if result["shutdown"] is None:
            result["shutdown"] = server.close()
        journal.close()
        result["events"] = read_events(directory)
        write_new(directory / "probe.json", result)
    return result


def assess(config, modes, environment, *, scope):
    """Derive named checks from explicit observations, retaining unknowns."""
    checks = []
    def add(code, condition, evidence, *, known=True):
        checks.append({"code": code, "status": ("PASS" if condition else "FAIL") if known else "UNKNOWN",
                       "evidence": evidence})
    real = config.data["evidence_kind"] == "real_gpu"
    add("CUDA_AVAILABLE", environment.get("cuda_available") is True,
        [{key: environment.get(key) for key in ("versions", "cuda_runtime", "gpu", "cuda_available")}], known=real)
    metadata = [mode.get("metadata") or {} for mode in modes]
    add("MODEL_ASSETS_PINNED", len(metadata) == 2 and all(meta.get("model_revision") == config.data["model_revision"] and
        meta.get("tokenizer_revision") == config.data["tokenizer_revision"] and meta.get("tokenizer_files_sha256")
        for meta in metadata), ["Local-only pinned snapshot loading; no download.", metadata], known=real)
    keys = ("model", "model_revision", "tokenizer_revision", "tokenizer_files_sha256", "dtype", "device",
            "actual_model_dtype", "actual_model_device", "chat_template_sha256", "versions", "cuda_runtime", "gpu")
    parity = [{key: meta.get(key) for key in keys} for meta in metadata]
    add("MODEL_MODE_PARITY", len(parity) == 2 and parity[0] == parity[1], parity)
    references = [mode.get("reference", {}) for mode in modes]
    covered = sorted(set(references[0]) & set(references[1])) if len(references) == 2 else []
    correct = bool(covered) and all(references[0][rid].get("success") and references[1][rid].get("success") and
        _signature(references[0][rid]) == _signature(references[1][rid]) and
        references[0][rid].get("observed_generated_token_ids") is not None and
        references[0][rid]["observed_generated_token_ids"] == references[1][rid].get("observed_generated_token_ids")
        for rid in covered)
    add("GREEDY_OUTPUT_CORRECT", correct and config.data["temperature"] == 0,
        [{"covered_request_ids": covered, "required_request_ids": [item["request_id"] for item in config.data["requests"]],
          "all_frozen_requests_covered": len(covered) == len(config.data["requests"]),
          "exact_terminal_token_ids_observed": correct}], known=real)
    add("CONCURRENCY_1_CORRECT", correct, [{"covered_request_ids": covered}], known=real)
    concurrent = [row for mode in modes for row in mode.get("concurrent", [])]
    add("CONCURRENCY_2_ISOLATED", bool(concurrent) and all(row.get("matches_independent_reference") for row in concurrent),
        [{"requests_checked": len(concurrent)}], known=scope == "full" and real)
    id_sets = [[*mode.get("reference", {}).values(), *mode.get("concurrent", [])] for mode in modes]
    isolated = len(modes) == 2 and all(rows and all(row.get("response_id") and row.get("lease_id") and
        row.get("observed_engine_request_id") for row in rows) and all(
        len({row.get(key) for row in rows}) == len(rows) for key in
        ("response_id", "lease_id", "observed_engine_request_id")) for rows in id_sets)
    add("REQUEST_IDENTITIES_ISOLATED", isolated,
        [{"mode": mode["mode"], "request_count": len(rows)} for mode, rows in zip(modes, id_sets)], known=real)
    add("NORMAL_CLEANUP", len(modes) == 2 and all(mode.get("reference") and mode.get("cleanup") and
        all(item.get("idle_after_test") for item in mode["cleanup"]) and
        (not real or all(row.get("kv_release_confirmed") is True for row in
            [*mode.get("reference", {}).values(), *mode.get("concurrent", [])])) for mode in modes),
        [{"mode": mode["mode"], "drains": mode.get("cleanup", [])} for mode in modes])
    errors = [mode.get("faults", {}).get("error", {}) for mode in modes]
    add("ERROR_CLEANUP", all(item.get("injection_consumed") and not item.get("row", {}).get("success") and
        item.get("drain", {}).get("idle_after_test") for item in errors), errors, known=scope == "full" and real)
    cancels = [mode.get("faults", {}).get("cancel", {}) for mode in modes]
    add("CANCELLATION_CLEANUP", all(item.get("partial_content_received") and item.get("cancelled_delta", 0) > 0 and
        item.get("drain", {}).get("idle_after_test") for item in cancels), cancels, known=scope == "full")
    stopped = [mode.get("shutdown", {}) for mode in modes]
    shutdown_events = [event for mode in modes for event in mode.get("events", []) if event.get("kind") == "shutdown_complete"]
    shutdown_ok = len(shutdown_events) == 2 and all(item.get("stopped") for item in stopped) and all(
        not item["owner_thread_alive"] and item["pending_finalizers"] == item["stream_consumers"] == 0 and
        item["scheduler"]["running_requests"] == item["scheduler"]["queued_requests"] == 0 and
        (item.get("continuous") is None or item["continuous"].get("allocated_slots") == 0) for item in shutdown_events)
    add("SHUTDOWN_CLEANUP", shutdown_ok and all(mode.get("faults", {}).get("shutdown_active_observed") for mode in modes),
        [{"passive_shutdown_clean": shutdown_ok}, *shutdown_events], known=scope == "full")
    add("PASSIVE_SHUTDOWN_CLEANUP", shutdown_ok, shutdown_events)
    events = [{**event, "mode": mode["mode"]} for mode in modes for event in mode.get("events", [])]
    coverage = warmup_coverage(config, events)
    coverage["processes"] = [{"mode": mode["mode"], "pid": (mode.get("metadata") or {}).get("pid"),
        "recipe": mode.get("process_warmup"),
        "successful_warmup_forwards": sum(e.get("kind") == "forward_completed" and e.get("success") is True and
            e.get("execution_stage") == "warmup" for e in mode.get("events", []))} for mode in modes]
    checks.append({"code": "WARMUP_SHAPES_COVERED", "status": coverage["status"], "evidence": [coverage]})
    outputs = {rid: _signature(references[0][rid]) for rid in covered if references[0][rid].get("success") and
               references[0][rid].get("observed_generated_token_ids") is not None}
    return checks, coverage, outputs


def run_preflight(config, output_dir, *, scope="single", allow_gpu=False):
    if scope not in {"single", "full"}:
        raise ValueError("scope must be single or full")
    if config.data["evidence_kind"] == "real_gpu" and not allow_gpu:
        raise ValueError("real GPU correctness probes require explicit allow_gpu")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new(output_dir / "config.json", config.data)
    source = git_snapshot()
    git = {key: value for key, value in source.items() if key != "files"}
    report = {"schema_version": 1, "kind": "correctness_preflight_not_performance", "scope": scope,
        "timestamp": utc_now(), "config_sha256": config.sha256, "git": git,
        "trace_sha256": None, "environment": {}, "modes": [], "checks": [], "status": "UNKNOWN",
        "expected_outputs": {}, "performance_samples": 0}
    try:
        environment = cuda_environment(config)
        report["environment"] = environment
        if config.data["evidence_kind"] == "real_gpu" and not environment["cuda_available"]:
            raise RuntimeError("CUDA unavailable; no fallback and no server started")
        trace, count_tokens = prepare_trace(config)
        report["trace_sha256"] = trace_sha256(trace)
        report["trace"] = trace
        from .shapes import build_manifest, model_geometry, save_proof
        manifest = build_manifest(config, trace, git, model_geometry(config))
        write_new(output_dir / "expected-shapes.json", manifest)
        if config.data["evidence_kind"] == "real_gpu":
            from .server import pinned_assets
            model_path, _, _ = pinned_assets(config.data)
            with (output_dir / "model-config.json").open("xb") as output:
                output.write((model_path / "config.json").read_bytes())
        report["hardware_before"] = hardware_snapshot()
        for mode in ("legacy", "continuous"):
            result = probe_mode(config, trace, count_tokens, mode, output_dir / mode,
                output_dir / "config.json", git, scope=scope)
            report["modes"].append(result)
            if not result.get("shutdown", {}).get("stopped"):
                report["aborted"] = "previous process may own resources; no next process launched"
                break
        checks, coverage, outputs = assess(config, report["modes"], environment, scope=scope)
        proof = save_proof(output_dir, manifest, report["modes"])
        coverage["shape_proof"] = proof
        coverage["status"] = proof["status"]
        # This is process-local preflight evidence, never warmup of a future
        # formal process. Keep that independent attestation requirement intact.
        for check in checks:
            if check["code"] == "WARMUP_SHAPES_COVERED":
                check["status"] = proof["status"]
                check["evidence"] = ["shape-coverage.json", proof]
        report.update(checks=checks, warmup_coverage=coverage, expected_outputs=outputs)
        report["hardware_after"] = hardware_snapshot()
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["checks"] = [{"code": code, "status": "FAIL" if code == "CUDA_AVAILABLE" and
            report["environment"].get("cuda_available") is False else "UNKNOWN",
            "evidence": [report["error"]]} for code in REQUIRED_CHECKS]
    report["source_unchanged"] = git_snapshot()["source_tree_sha256"] == source["source_tree_sha256"]
    report["checks"].append({"code": "SOURCE_UNCHANGED", "status": "PASS" if report["source_unchanged"] else "FAIL",
                             "evidence": [git]})
    report["status"] = overall_status(report["checks"])
    write_new(output_dir / "preflight.json", report)
    write_new(output_dir / "artifacts.json", {"config_sha256": config.sha256, "git": git,
        "sha256": {path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output_dir.rglob("*") if path.is_file()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scope", choices=("single", "full"), default="single")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.execute:
        print(json.dumps({"plan_only": True, "config_sha256": config.sha256, "scope": args.scope,
            "request_count_per_mode": 1 if args.scope == "single" else len(config.data["requests"]),
            "model": config.data["model"], "performance_samples": 0,
            "unexecuted_checks_remain_unknown": True}, indent=2))
        return
    if config.data["evidence_kind"] == "real_gpu" and not args.allow_gpu:
        parser.error("--allow-gpu is required; nothing executed")
    output = args.output_dir or Path("benchmarks/runtime/results/online_http") / (
        time.strftime("preflight-%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:12])
    result = run_preflight(config, output, scope=args.scope, allow_gpu=args.allow_gpu)
    print(json.dumps({"output_dir": str(output), "status": result["status"],
                      "scope": args.scope, "performance_samples": 0}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
