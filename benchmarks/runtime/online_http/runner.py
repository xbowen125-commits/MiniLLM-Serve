"""Plan by default; explicitly execute bounded fresh-process HTTP comparisons."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .artifacts import ROOT, Journal, environment_snapshot, git_snapshot, gpu_environment_snapshot, hardware_snapshot, utc_now, write_new
from .client import get_json, observe_request, request_record
from .config import load_config
from .metrics import calculate_request_metrics, summarize
from .report import render_report
from .workload import build_trace, execution_order, trace_sha256
from .gate import evaluate_trial, load_preflight, validate_admission, verify_preflight_artifacts


def is_idle(status, *, expected_mode=None):
    if status.get("running_requests") != 0 or status.get("queued_requests") != 0:
        return False
    runtime = status.get("continuous")
    if expected_mode == "continuous" and runtime is None:
        return False  # Missing resource telemetry is not proof of zero slots.
    keys = ("pending", "allocated_slots", "contexts", "claims", "released_ids", "finished_ids",
            "runtime_active_requests", "pending_commands", "stream_consumers", "pending_finalizers")
    return runtime is None or all(runtime.get(key) == 0 for key in keys)


def drain(base_url, config, *, mode=None):
    start = time.perf_counter()
    status, error = None, None
    while time.perf_counter() - start < config.data["drain_timeout_s"]:
        try:
            status = get_json(base_url, "/scheduler/status", timeout=1)
            if is_idle(status, expected_mode=mode):
                return {"seconds": time.perf_counter() - start, "idle_after_test": True, "status": status}
        except Exception as exc:
            error = type(exc).__name__
        time.sleep(config.data["poll_interval_s"])
    return {"seconds": time.perf_counter() - start, "idle_after_test": False, "status": status, "error": error}


class ResourceSampler:
    """HTTP sampling lower bounds, not an exact resettable CUDA peak meter."""
    def __init__(self, base_url, config, path, metadata):
        self.url, self.config = base_url, config
        self.journal = Journal(path)
        self.rows = []
        self.metadata = metadata
        self.phase = "warmup"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="benchmark-status-sampler", daemon=True)
    def _run(self):
        while not self.stop_event.is_set():
            row = {**self.metadata, "phase": self.phase, "timestamp": utc_now(), "monotonic_s": time.perf_counter()}
            try:
                row["status"] = get_json(self.url, "/scheduler/status", timeout=1)
            except Exception as exc:
                row["error"] = type(exc).__name__
            self.rows.append(row)
            self.journal.append(row)
            self.stop_event.wait(self.config.data["telemetry_interval_s"])
    def start(self):
        self.thread.start()
    def close(self):
        self.stop_event.set()
        self.thread.join(3)
        self.journal.close()


class ServerProcess:
    def __init__(self, config, mode, directory, config_path, git, *,
                 host_module="benchmarks.runtime.online_http.server", extra_args=()):
        self.config, self.directory = config, directory
        self.mode, self.git = mode, git
        self.log = (directory / "server.log").open("x", encoding="utf-8")
        self.log.write(json.dumps({"config_sha256": config.sha256, "git": git, "mode": mode}) + "\n")
        self.log.flush()
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
                   MINILLM_EXECUTION_MODE="legacy")
        command = [sys.executable, "-m", host_module, "--config", str(config_path),
                   "--mode", mode, "--port", "0", "--trial-dir", str(directory), *extra_args]
        self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
            stdout=self.log, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        self.metadata = None
        self.base_url = None
    def wait_ready(self):
        start = time.perf_counter()
        while time.perf_counter() - start < self.config.data["startup_timeout_s"]:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited during startup: {self.process.returncode}; retained server.log")
            try:
                self.metadata = json.loads((self.directory / "server-ready.json").read_text(encoding="utf-8"))
                self.base_url = f"http://127.0.0.1:{self.metadata['port']}"
                health = get_json(self.base_url, "/health", timeout=.5)
                if health.get("status") == "ok" and health.get("loaded"):
                    if health["model"] != self.config.data["model"] or self.metadata["config_sha256"] != self.config.sha256:
                        raise RuntimeError("server provenance mismatch")
                    if (self.metadata["mode"] != self.mode or
                            self.metadata["git"]["source_tree_sha256"] != self.git["source_tree_sha256"]):
                        raise RuntimeError("server mode or source provenance mismatch")
                    self.metadata["health"] = health
                    self.metadata["launcher_pid"] = self.process.pid
                    return time.perf_counter() - start
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(self.config.data["poll_interval_s"])
        raise TimeoutError("server startup timeout; no request measurements attempted")
    def close(self):
        started = time.perf_counter()
        if self.process.poll() is None:
            try:
                self.process.stdin.write("STOP\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            except (OSError, BrokenPipeError):
                pass
        try:
            self.process.wait(timeout=self.config.data["drain_timeout_s"] + 5)
        except subprocess.TimeoutExpired:
            # Never fake a clean drain or start another GPU process while a
            # previous one might still own resources. No forced kill here.
            return {"seconds": time.perf_counter() - started, "stopped": False,
                    "launcher_pid": self.process.pid, "manual_intervention_required": True}
        finally:
            self.log.close()
        path = self.directory / "server-stopped.json"
        return {"seconds": time.perf_counter() - started, "stopped": self.process.returncode == 0,
                "returncode": self.process.returncode,
                "final": json.loads(path.read_text()) if path.exists() else None}


def prepare_trace(config):
    """One pinned tokenizer and one prepared trace for every mode and process."""
    if config.data["evidence_kind"] == "cpu_fake":
        from .server import FakeTokenizer
        tokenizer = FakeTokenizer()
    else:
        from .server import pinned_assets
        from transformers import AutoTokenizer
        _, path, _ = pinned_assets(config.data)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    trace = build_trace(config)
    for item in trace:
        prompt = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
        try:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            item["observed_prompt_tokens"] = len(ids)
            item["prompt_ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        except Exception as exc:
            if config.data["evidence_kind"] != "cpu_fake":
                raise
            item["observed_prompt_tokens"] = None
            item["prompt_preflight_error"] = str(exc)
        if item.get("expected_prompt_tokens") is not None and item["observed_prompt_tokens"] != item["expected_prompt_tokens"]:
            raise ValueError("frozen expected prompt length does not match tokenizer")
    lock = threading.Lock()
    def count_tokens(text):
        with lock:
            return len(tokenizer.encode(text, add_special_tokens=False))
    return trace, count_tokens


def replay(base_url, config, trace, token_count, journal, metadata, *, phase="measured", identity_trace=False):
    """Replay nominal offsets with a bounded client pool, recording real lateness."""
    origin = time.perf_counter() + .02
    rows, errors, futures = [], [], []
    order = {item["request_id"]: i for i, item in enumerate(trace)}
    lock = threading.Lock()
    def perform(item, generated):
        worker_started = time.perf_counter() - origin
        # Queueing behind the bounded pool is visible in sent_at/lateness, not
        # hidden by moving the recorded scheduled arrival.
        if time.perf_counter() - origin >= config.data["max_run_seconds"]:
            return calculate_request_metrics({**request_record(config, item), "outcome": "not_sent",
                "generator_scheduled_at_s": generated, "worker_started_at_s": worker_started,
                "terminal_at_s": time.perf_counter() - origin, "error_code": "client_run_deadline"})
        return observe_request(base_url, config, item, origin, token_count,
                               run_deadline=origin + config.data["max_run_seconds"],
                               generator_scheduled_at_s=generated, worker_started_at_s=worker_started,
                               **({"identity_record_id": uuid.uuid4().hex} if identity_trace else {}))
    def completed(future, item):
        try:
            row = future.result()
        except Exception as exc:
            row = calculate_request_metrics({**request_record(config, item), "outcome": "error",
                   "error_code": "client_runner_exception", "error_message": str(exc),
                   "terminal_at_s": time.perf_counter() - origin})
        row.update(metadata, phase=phase, timestamp=utc_now(), trace_ordinal=order[item["request_id"]])
        with lock:
            rows.append(row)
            try:
                journal.append(row)
            except Exception as exc:
                errors.append(str(exc))
    with ThreadPoolExecutor(max_workers=1 if phase == "warmup" else config.data["max_client_concurrency"]) as pool:
        for item in trace:
            delay = origin + item["arrival_s"] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            generated = time.perf_counter() - origin
            future = pool.submit(perform, item, generated)
            futures.append(future)
            future.add_done_callback(lambda f, item=item: completed(f, item))
    if errors:
        raise RuntimeError("raw journal persistence failed: " + "; ".join(errors))
    rows.sort(key=lambda row: order[row["request_id"]])
    window = max([row["terminal_at_s"] for row in rows] or [.000001])
    return rows, max(window, .000001)


def run_trial(config, trace, token_count, *, mode, pair, directory, config_path, git):
    directory.mkdir(exist_ok=False)
    trial = {"trial_id": directory.name, "pair_index": pair, "mode": mode,
        "config_sha256": config.sha256, "trace_sha256": trace_sha256(trace),
        "git": git,
        "comparison_valid": False, "issues": [], "raw_jsonl": f"{directory.name}/requests.jsonl"}
    journal = Journal(directory / "requests.jsonl")
    metadata = {"trial_id": trial["trial_id"], "pair_index": pair, "mode": mode,
                "config_sha256": config.sha256, "trace_sha256": trial["trace_sha256"],
                "git_sha": git["sha"], "git_dirty": git["dirty"]}
    server = sampler = None
    measured_started = False
    trial["hardware_before"] = hardware_snapshot()
    try:
        if config.data.get("warmup_strategy") == "sequential_and_pairs":
            server = ServerProcess(config, mode, directory, config_path, git,
                host_module="benchmarks.runtime.online_http.preflight_server")
        else:
            server = ServerProcess(config, mode, directory, config_path, git)
        startup = server.wait_ready()
        trial["server_metadata"] = server.metadata
        metadata["server_pid"] = server.metadata["pid"]
        before = drain(server.base_url, config, mode=mode)
        if not before["idle_after_test"]:
            raise RuntimeError("server not idle before warmup")
        sampler = ResourceSampler(server.base_url, config, directory / "resources.jsonl", metadata)
        sampler.start()
        warm_trace = []
        warm_candidates = [item for item in trace if item.get("prompt_preflight_error") is None]
        for index in range(config.data["warmup_requests"]):
            warm_trace.append({**warm_candidates[index % len(warm_candidates)],
                               "request_id": f"warmup-{index}", "arrival_s": 0})
        if config.data.get("warmup_strategy") == "sequential_and_pairs":
            from .warmup import run_warmup
            warm_start = time.perf_counter()
            trial["process_warmup"] = run_warmup(config, trace, token_count, server, directory, journal, drain)
            from .preflight import read_events, warmup_coverage
            trial["warmup_coverage"] = warmup_coverage(config,
                [{**event, "mode": mode} for event in read_events(directory)])
            trial["warmup_coverage"]["process_identity"] = trial["process_warmup"]
            if config.data["evidence_kind"] == "real_gpu" and trial["warmup_coverage"]["status"] != "PASS":
                raise RuntimeError("TRIAL_WARMUP_COVERAGE: incomplete live-process tensor proof; no formal replay")
            warm_seconds = time.perf_counter() - warm_start
            warm_rows = []  # Actual successful rows were checked and persisted by the common recipe.
        else:
            warm_rows, warm_seconds = replay(server.base_url, config, warm_trace, token_count, journal, metadata, phase="warmup")
        warm_drain = drain(server.base_url, config, mode=mode)
        if any(not row["success"] for row in warm_rows) or not warm_drain["idle_after_test"]:
            raise RuntimeError("warmup failed or did not drain; formal phase not attempted")
        sampler.phase = "measured"
        measured_started = True
        rows, window = replay(server.base_url, config, trace, token_count, journal, metadata)
        sampler.phase = "drain"
        after = drain(server.base_url, config, mode=mode)
        summary = summarize(rows, window_s=window, slo=config.data["slo"])
        trial["summary"] = summary
        trial["phase_durations_s"] = {"startup": startup, "warmup": warm_seconds,
            "warmup_drain": warm_drain["seconds"], "measured": window, "drain": after["seconds"]}
        late = max([row.get("send_lateness_ms") or 0 for row in rows] or [0])
        trial["max_send_lateness_ms"] = late
        if late > config.data["arrival_tolerance_ms"]:
            trial["issues"].append("actual client sends exceeded frozen arrival tolerance")
        if summary["failed_requests"]:
            trial["issues"].append("retained failed/timeout/cancelled/length-mismatch requests")
        if summary["metric_validation_error_requests"]:
            trial["issues"].append("retained samples with invalid metric observations")
        if not after["idle_after_test"]:
            trial["issues"].append("resource drain unconfirmed")
        lifetime_slots, lifetime_gpu = [], []
        measured_slots, measured_gpu = [], []
        for sample in sampler.rows + [{"phase": "drain", "status": after.get("status") or {}}]:
            status = sample.get("status", {})
            slots = status.get("continuous", {}).get("peak_allocated_slots")
            gpu = status.get("gpu", {}).get("max_memory_allocated_mb")
            if slots is not None: lifetime_slots.append(slots)
            if gpu is not None: lifetime_gpu.append(gpu)
            if sample["phase"] == "measured":
                current = status.get("continuous", {}).get("allocated_slots")
                memory = status.get("gpu", {}).get("memory_allocated_mb")
                if current is not None: measured_slots.append(current)
                if memory is not None: measured_gpu.append(memory)
        trial["resources"] = {"before": before, "after": after,
            "peak_kv_slots_process_lifetime": max(lifetime_slots, default=None),
            "gpu_peak_allocated_mb_process_lifetime": max(lifetime_gpu, default=None),
            "measured_sample_peak_kv_slots": max(measured_slots, default=None),
            "measured_sample_peak_gpu_allocated_mb": max(measured_gpu, default=None)}
    except Exception as exc:
        trial["issues"].append(f"{type(exc).__name__}: {exc}")
        # Even a failed startup/warmup retains a planned row for every request
        # which has not reached the formal transport, rather than vanishing.
        if "summary" not in trial:
            journal.output.flush()
            rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()
                    if json.loads(line).get("phase") == "measured"]
            retained = {row["request_id"] for row in rows}
            for ordinal, item in enumerate(trace):
                if item["request_id"] in retained:
                    continue
                row = calculate_request_metrics({**request_record(config, item), **metadata,
                    "phase": "measured", "trace_ordinal": ordinal,
                    "timestamp": utc_now(), "outcome": "error" if measured_started else "not_sent", "success": False,
                    "error_code": "observation_lost_after_dispatch" if measured_started else "trial_setup_failed"})
                journal.append(row); rows.append(row)
            window = max([row.get("terminal_at_s") or 0 for row in rows] or [.000001])
            trial["summary"] = summarize(rows, window_s=max(window, .000001), slo=config.data["slo"])
    finally:
        if sampler is not None:
            sampler.close()
        journal.close()
        if server is not None:
            stopped = server.close()
            trial["shutdown"] = stopped
            trial.setdefault("phase_durations_s", {})["shutdown"] = stopped["seconds"]
            if (not stopped["stopped"] or not stopped.get("final") or stopped["final"]["pending_finalizers"]
                    or stopped["final"]["stream_consumers"] or stopped["final"]["owner_thread_alive"]):
                trial["issues"].append("server shutdown not confirmed")
        trial["hardware_after"] = hardware_snapshot()
    trial["comparison_valid"] = not trial["issues"]
    trial["validity_scope"] = "trial_local_only; final-verdict.json and root summary are authoritative"
    write_new(directory / "summary.json", trial)
    return trial


def _publish_experiment(config, output_dir, source, experiment, preflight=None, environment=None):
    """Append final verdicts; diagnostic summaries never silently become formal samples."""
    git = experiment["git"]
    trace = experiment["prepared_trace"]
    final_source = git_snapshot()
    experiment["source_unchanged"] = final_source["source_tree_sha256"] == source["source_tree_sha256"]
    for trial in experiment["trials"]:
        if not experiment["source_unchanged"]:
            trial["comparison_valid"] = False
            trial["issues"].append("working sources changed during experiment")
        raw = output_dir / trial["raw_jsonl"]
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("phase") == "measured"]
        gate = evaluate_trial(config, trace, rows, trial, preflight=preflight, git=git,
            environment=environment, artifact_evidence={
                "exclusive_directory": True, "history_preserved": True,
                "raw_journal_complete": len(rows) == len(trace),
                "source_unchanged": experiment["source_unchanged"],
                "config_sha256": config.sha256, "trace_sha256": trace_sha256(trace),
                "source_tree_sha256": git["source_tree_sha256"],
                "evidence": ["source-manifest.json", trial["raw_jsonl"], "exclusive open(x) artifact writer"]})
        local_ok = trial["comparison_valid"]
        gate["checks"].append({"code": "LOCAL_AND_PAIR_VALIDATION", "status": "PASS" if local_ok else "FAIL",
            "evidence": [trial["trial_id"] + "/summary.json", "summary.json.pair_validation"],
            "detail": "Existing local, pair and source checks remain mandatory."})
        if not local_ok:
            gate["status"] = "FAIL"
            gate["reason_codes"].append("LOCAL_AND_PAIR_VALIDATION")
            gate["eligibility"].update(formal_statistics=False, actual_generated_throughput=False, http_visible_proxy=False)
        trial["validity_gate"] = gate
        trial["formal_statistics"] = trial["summary"] if gate["eligibility"]["formal_statistics"] else None
        trial["comparison_valid"] = bool(gate["eligibility"]["formal_statistics"])
        trial["validity_scope"] = "final_gate_pair_and_source_verdict"
        write_new(output_dir / trial["trial_id"] / "final-verdict.json", {
            "config_sha256": config.sha256, "git": git, "trial_id": trial["trial_id"],
            "trial_local_valid": trial["trial_local_valid"], "comparison_valid": trial["comparison_valid"],
            "source_unchanged": experiment["source_unchanged"], "issues": trial["issues"],
            "validity_gate": gate, "pair_validation": experiment["pair_validation"][trial["pair_index"]]})
    write_new(output_dir / "summary.json", experiment)
    with (output_dir / "report.md").open("x", encoding="utf-8") as report:
        report.write(render_report(experiment))
    write_new(output_dir / "artifacts.json", {"config_sha256": config.sha256, "git": git,
        "sha256": {path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output_dir.rglob("*") if path.is_file()}})
    return experiment


def run_experiment(config, output_dir, *, preflight_path=None, preflight_sha256=None):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config.json"
    write_new(config_path, config.as_dict())
    trace, token_count = prepare_trace(config)
    source = git_snapshot()
    git = {key: value for key, value in source.items() if key != "files"}
    write_new(output_dir / "trace.json", {"config_sha256": config.sha256, "git": git,
        "trace_sha256": trace_sha256(trace), "requests": trace})
    write_new(output_dir / "source-manifest.json", {**source, "config_sha256": config.sha256})
    experiment = {"protocol_version": 1, "timestamp": utc_now(), "evidence_kind": config.data["evidence_kind"],
        "config_sha256": config.sha256, "trace_sha256": trace_sha256(trace), "config": config.as_dict(),
        "git": git, "client_environment": environment_snapshot(),
        "prepared_trace": trace, "planned_order": execution_order(config.data["pairs"]), "trials": []}
    preflight, environment = None, None
    if config.data["evidence_kind"] == "real_gpu":
        integrity = {"code": "PREFLIGHT_ARTIFACT_INTEGRITY", "status": "UNKNOWN", "evidence": ["No preflight report supplied."]}
        if preflight_path is not None:
            try:
                loaded = load_preflight(preflight_path, expected_sha256=preflight_sha256)
                preflight = loaded["report"]
                experiment["preflight_binding"] = {"report_sha256": loaded["sha256"], "path": str(preflight_path)}
                integrity = verify_preflight_artifacts(preflight_path)
            except (OSError, ValueError) as exc:
                integrity = {"code": "PREFLIGHT_ARTIFACT_INTEGRITY", "status": "FAIL", "evidence": [str(exc)]}
        environment = gpu_environment_snapshot(config)
        admission = validate_admission(config, trace, preflight, git, environment)
        admission["checks"].append(integrity)
        if integrity["status"] != "PASS":
            admission["reason_codes"].append(integrity["code"])
            admission.update(status="FAIL", admitted=False)
        experiment["admission"] = admission
        write_new(output_dir / "admission.json", {"config_sha256": config.sha256, "git": git, **admission})
        if not admission["admitted"]:
            experiment.update(aborted="GPU formal validity gate blocked; no server or formal request launched",
                              pair_validation=[])
            return _publish_experiment(config, output_dir, source, experiment, preflight, environment)
    else:
        experiment["admission"] = {"status": "NOT_APPLICABLE", "admitted": False,
            "scope": "CPU/Fake protocol tests only, never formal performance samples"}
    horizon = trace[-1]["arrival_s"] - trace[0]["arrival_s"]
    experiment["workload"] = {"request_count": len(trace), "arrival_horizon_s": horizon,
        "nominal_average_arrival_rate_rps": (len(trace) - 1) / horizon if horizon > 0 else None,
        "burst": horizon == 0, "generation_limits": sorted({item["max_tokens"] for item in trace}),
        "prompt_lengths": [item.get("observed_prompt_tokens") for item in trace]}
    for index, (pair, mode) in enumerate(experiment["planned_order"]):
        trial = run_trial(config, trace, token_count, mode=mode, pair=pair,
            directory=output_dir / f"trial-{index:03d}-{mode}", config_path=config_path, git=git)
        experiment["trials"].append(trial)
        if not trial.get("shutdown", {}).get("stopped", False):
            experiment["aborted"] = "previous server may still own resources; no further process launched"
            break
    experiment["pair_validation"] = []
    for pair in range(config.data["pairs"]):
        members = [trial for trial in experiment["trials"] if trial["pair_index"] == pair]
        issues = []
        if len(members) != 2:
            issues.append("pair incomplete; no cross-mode comparison")
        else:
            checks = []
            for trial in members:
                meta = trial.get("server_metadata", {})
                checks.append({key: meta.get(key) for key in ("model", "model_revision", "tokenizer_revision",
                    "tokenizer_files_sha256", "tokenizer_class", "chat_template_sha256",
                    "dtype", "device", "actual_model_dtype", "actual_model_device", "cuda_runtime", "gpu", "versions")})
            if checks[0] != checks[1]:
                issues.append("server model/tokenizer/device/environment provenance differs")
            outputs = []
            for trial in members:
                raw = output_dir / trial["raw_jsonl"]
                outputs.append({row["request_id"]: (row.get("output_sha256"), row.get("visible_output_tokens"),
                    row.get("finish_reason"), row.get("scheduled_at_s")) for row in
                    (json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()) if row["phase"] == "measured"})
            if outputs[0] != outputs[1]:
                issues.append("paired visible outputs/finish reasons or nominal arrivals differ")
        valid = not issues and all(trial["comparison_valid"] for trial in members)
        experiment["pair_validation"].append({"pair_index": pair, "valid": valid, "issues": issues})
        for trial in members:
            trial["trial_local_valid"] = trial["comparison_valid"]
            trial["comparison_valid"] = valid
            trial["issues"].extend(issues)
    return _publish_experiment(config, output_dir, source, experiment, preflight, environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--preflight", type=Path, help="Complete matching correctness report required for formal GPU execution")
    parser.add_argument("--preflight-sha256", help="Optional external SHA-256 binding for the supplied report")
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.execute:
        print(json.dumps({"plan_only": True, "config_sha256": config.sha256,
            "order": execution_order(config.data["pairs"]), "trace": build_trace(config)}, indent=2))
        return
    if config.data["evidence_kind"] != "cpu_fake" and not args.allow_gpu:
        parser.error("real-model execution requires --allow-gpu; no experiment executed")
    output = args.output_dir or Path("benchmarks/runtime/results/online_http") / (
        time.strftime("run-%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:12])
    result = run_experiment(config, output, preflight_path=args.preflight, preflight_sha256=args.preflight_sha256)
    print(json.dumps({"output_dir": str(output), "trials": len(result["trials"]),
                      "evidence_kind": result["evidence_kind"], "admission": result["admission"], "performance_claim": False}))
    if config.data["evidence_kind"] == "real_gpu" and not result["admission"]["admitted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
