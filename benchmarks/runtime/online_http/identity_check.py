"""Reconstruct identity edges from raw files; absent edges never follow counts.

All references are physical file hashes and one-based JSONL lines (or JSON
pointers). This is observed conformance, not Runtime bisimulation or admission.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .artifacts import write_new, Journal
from .transition_relation import map_events
from .workload import build_trace, trace_sha256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def referenced(path):
    digest = sha(path)
    return [(json.loads(line), {"file": Path(path).name, "line": n, "sha256": digest})
            for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1)]


def reconstruct(config, events, clients, *, current_source, terminals=()):
    """Inputs are (raw row, authenticated physical reference) pairs."""
    issues, mappings, seen = [], [], {}
    def issue(status, code, detail, evidence):
        issues.append({"status": status, "code": code, "detail": detail, "evidence": evidence})
    unique = []
    for row, ref in events:
        key = row.get("event_id")
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if key in seen and key is not None:
            if seen[key] != digest:
                issue("FAIL", "EVENT_ID_CONFLICT", "same event ID has changed payload", ref)
            continue
        if key is None:
            issue("UNKNOWN", "EVENT_ID_MISSING", "raw event lacks identity", ref)
        seen[key] = digest
        unique.append((row, {**ref, "event_id": key}))
    events = unique
    identities = {(r.get("run_id"), r.get("pid"), r.get("config_sha256")) for r, _ in events}
    if len(identities) != 1 or any(None in identity for identity in identities):
        issue("UNKNOWN" if any(None in identity for identity in identities) else "FAIL",
              "RAW_IDENTITY_SCOPE", "run/PID/config identity missing or conflicts", [])
    elif next(iter(identities))[2] != config.sha256:
        issue("FAIL", "CONFIG_IDENTITY_CONFLICT", "raw configuration differs from frozen file", [])
    for field in ("model_revision", "dtype", "device"):
        values = {r.get(field) for r, _ in events}
        if None in values:
            issue("UNKNOWN", "MODEL_IDENTITY_MISSING", field, [])
        elif values != {config.data[field]}:
            issue("FAIL", "MODEL_IDENTITY_CONFLICT", field, [])
    def select(kind, **values):
        return [(r, f) for r, f in events if r.get("kind") == kind and all(r.get(k) == v for k, v in values.items())]
    def required(kind, **values):
        found = select(kind, **values)
        if len(found) != 1:
            issue("UNKNOWN" if not found else "FAIL", "BOUNDARY_MISSING" if not found else "BOUNDARY_DUPLICATE",
                  {"kind": kind, **values}, [f for _, f in found])
        return found[0] if len(found) == 1 else ({}, {})
    expected = build_trace(config)
    planned = {r["request_id"]: r for r in expected}
    ids = [r.get("planned_request_id", r.get("request_id")) for r, _ in clients]
    if len(ids) != len(set(ids)):
        issue("FAIL", "DUPLICATE_PLANNED_ID", "duplicate client rows, not silently deduplicated", [])
    if set(ids) - planned.keys():
        issue("FAIL", "UNKNOWN_PLANNED_ID", sorted(set(ids) - planned.keys()), [])
    cids = [r.get("client_record_id") for r, _ in clients]
    if len([x for x in cids if x is not None]) != len(set(x for x in cids if x is not None)):
        issue("FAIL", "CLIENT_ID_CONFLICT", "client record ID reused", [])
    for item in expected:
        pid = item["request_id"]
        candidates = [(r, f) for r, f in clients if r.get("planned_request_id", r.get("request_id")) == pid]
        if len(candidates) != 1:
            issue("UNKNOWN", "CLIENT_RECORD_MISSING", pid, [])
            continue
        client, cref = candidates[0]
        if client.get("server_pid") is None:
            issue("UNKNOWN", "CLIENT_PID_MISSING", pid, cref)
        elif None in {r.get("pid") for r, _ in events}:
            issue("UNKNOWN", "RAW_PID_MISSING", pid, cref)
        elif {r.get("pid") for r, _ in events} != {client["server_pid"]}:
            issue("FAIL", "CLIENT_PID_CONFLICT", pid, cref)
        cid = client.get("client_record_id")
        if cid is None:
            issue("UNKNOWN", "CLIENT_ID_MISSING", pid, cref)
        begin, bref = required("identity_http_begin", client_record_id=cid, planned_request_id=pid)
        exchange = begin.get("http_exchange_id")
        created, nref = required("identity_request_created", http_exchange_id=exchange) if exchange else ({}, {})
        rid, obj = created.get("request_id"), created.get("request_object_identity")
        if rid is None or obj is None:
            issue("UNKNOWN", "REQUEST_OBJECT_MISSING", pid, nref)
        accept, aref = required("identity_admission_accept", request_id=rid) if rid else ({}, {})
        lease = accept.get("lease_id")
        enqueue, eref = required("identity_command_enqueue", request_id=rid) if rid else ({}, {})
        command = enqueue.get("command_id")
        start, sref = required("identity_command_start", command_id=command) if command else ({}, {})
        handoff, href = required("identity_runtime_handoff", request_id=rid) if rid else ({}, {})
        finish, fref = required("identity_command_finish", command_id=command) if command else ({}, {})
        final, lref = required("identity_finalizer_complete", request_id=rid) if rid else ({}, {})
        submit, subref = required("submit_end", request_id=rid) if rid else ({}, {})
        worker = [(r, f) for r, f in events if r.get("kind") == "worker_begin" and rid in (r.get("members") or [])]
        if not worker:
            issue("UNKNOWN", "WORKER_ID_MISSING", pid, [])
        edges = {"planned_request_id": pid, "client_record_id": cid, "http_lease_id": lease,
            "engine_request_state_id": rid, "runtime_context_id": handoff.get("runtime_context_id"),
            "worker_request_id": rid if worker else None, "terminal_result_id": finish.get("terminal_result_id")}
        if any(v is None for v in edges.values()):
            issue("UNKNOWN", "SEVEN_LAYER_CHAIN_INCOMPLETE", edges, [cref, bref, nref, href, fref])
        for row, ref in [(accept, aref), (enqueue, eref), (start, sref), (handoff, href), (finish, fref), (final, lref)]:
            if row and (row.get("request_id") != rid or row.get("request_object_identity") != obj):
                issue("FAIL", "REQUEST_OBJECT_CONFLICT", pid, ref)
        if handoff and handoff.get("runtime_context_id") != rid:
            issue("FAIL", "CONTEXT_ID_CONFLICT", pid, href)
        usage = client.get("authoritative_usage") or {}
        from .gate import authoritative_count
        if not authoritative_count(client, config.data):
            issue("UNKNOWN", "AUTHORITATIVE_COUNT_UNVERIFIED", pid, cref)
        if lease is None or client.get("lease_id") is None:
            issue("UNKNOWN", "CLIENT_LEASE_MISSING", pid, cref)
        elif str(client["lease_id"]) != str(lease):
            issue("FAIL", "CLIENT_LEASE_CONFLICT", pid, cref)
        if usage.get("engine_request_id") is None:
            issue("UNKNOWN", "CLIENT_ENGINE_ID_MISSING", pid, cref)
        elif usage["engine_request_id"] != rid:
            issue("FAIL", "CLIENT_ENGINE_CONFLICT", pid, cref)
        responses = [(r, f) for r, f in select("identity_response_identity", http_exchange_id=exchange)] if exchange else []
        if not responses or client.get("response_id") is None:
            issue("UNKNOWN", "RESPONSE_ID_MISSING", pid, cref)
        elif {r.get("response_id") for r, _ in responses} != {client["response_id"]}:
            issue("FAIL", "RESPONSE_ID_CONFLICT", pid, [f for _, f in responses])
        if client.get("scheduled_at_s") != item["arrival_s"] or client.get("max_tokens") != item["max_tokens"]:
            issue("FAIL", "FROZEN_REPLAY_CHANGED", pid, cref)
        payload = json.dumps({"model": config.data["model"], "messages": item["messages"],
            "max_tokens": item["max_tokens"], "stream": config.data["stream"],
            **{k: config.data[k] for k in ("temperature", "top_p", "top_k", "seed", "stop")}},
            ensure_ascii=False).encode("utf-8")
        if client.get("request_body_sha256") is None:
            issue("UNKNOWN", "REQUEST_BODY_UNOBSERVED", pid, cref)
        elif client["request_body_sha256"] != hashlib.sha256(payload).hexdigest():
            issue("FAIL", "REQUEST_BODY_CONFLICT", pid, cref)
        lateness = client.get("send_lateness_ms")
        if lateness is None:
            issue("UNKNOWN", "SEND_LATENESS_MISSING", pid, cref)
        elif lateness > config.data["arrival_tolerance_ms"]:
            issue("FAIL", "SEND_LATE_25MS", lateness, cref)
        # IDs alone cannot prove normal execution or expected token count.
        if not client.get("success") or not client.get("saw_done"):
            issue("FAIL", "OUTPUT_INCOMPLETE", pid, cref)
        count, reason = client.get("generated_output_tokens"), client.get("finish_reason")
        if (not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= item["max_tokens"] or
                reason not in {"length", "stop"} or (reason == "length" and count != item["max_tokens"])):
            issue("FAIL", "OUTPUT_LENGTH", pid, cref)
        terminal_records = [(r, f) for r, f in terminals if r.get("kind") == "terminal" and r.get("request_id") == rid]
        if terminals:
            if len(terminal_records) != 1:
                issue("UNKNOWN" if not terminal_records else "FAIL", "TERMINAL_COUNTER_MISSING_OR_DUPLICATE", pid,
                    [f for _, f in terminal_records])
            else:
                terminal, tref = terminal_records[0]
                if terminal.get("pid") != client.get("server_pid") or len(terminal.get("generated_token_ids", [])) != count:
                    issue("FAIL", "TERMINAL_COUNTER_CONFLICT", pid, tref)
        elif reason == "stop":
            issue("UNKNOWN", "EARLY_STOP_TERMINAL_UNOBSERVED", pid, cref)
        if not finish or finish.get("has_cache_ref") is not False:
            issue("UNKNOWN" if not finish else "FAIL", "TERMINAL_REFERENCE_NOT_CONFIRMED", pid, fref)
        if final and (final.get("released") is not True or final.get("has_cache_ref") is not False):
            issue("FAIL", "FINALIZER_UNCONFIRMED", pid, lref)
        if fref and lref and fref.get("line", 0) >= lref.get("line", 0):
            issue("FAIL", "LEASE_BEFORE_OWNER_CONFIRMATION", pid, [fref, lref])
        if bref and nref and aref and eref and sref and href and fref:
            numbers = [r["line"] for r in (bref, nref, aref, eref, sref, href, fref)]
            if numbers != sorted(numbers):
                issue("FAIL", "BOUNDARY_ORDER", pid, numbers)
        mappings.append({**edges, "request_object_identity": obj, "response_id": client.get("response_id"),
            "evidence": {"client": cref, "http": bref, "creation": nref, "admission": aref, "enqueue": eref,
                "owner_start": sref, "handoff": href, "submit": subref, "worker": [f for _, f in worker],
                "terminal": fref, "terminal_tokens": [f for _, f in terminal_records],
                "finalizer": lref, "response": [f for _, f in responses]}})
    runtime = {r["request_id"] for r, _ in events if r.get("kind") == "submit_end" and r.get("error") is None}
    mapped = {r["engine_request_state_id"] for r in mappings if r["engine_request_state_id"] is not None}
    if runtime != mapped:
        issue("UNKNOWN", "CLIENT_RUNTIME_COVERAGE", {"unlinked_runtime": sorted(runtime - mapped),
              "unlinked_clients": sorted(mapped - runtime)}, [])
    # Direct queue observations: FIFO is checked within each actual admission
    # queue; this is not an assertion about simultaneous TCP arrival order.
    queues = {}
    for row, ref in events:
        kind = row.get("kind")
        q = queues.setdefault(row.get("queue_identity"), [])
        if kind == "identity_queue_enter":
            q.append(row.get("request_id"))
        elif kind == "identity_queue_leave":
            rid = row.get("request_id")
            if rid not in q:
                issue("UNKNOWN", "QUEUE_ENTRY_MISSING", rid, ref)
            else:
                # Cancellation/remove and shutdown drain are not FIFO grants.
                if row.get("operation") == "get_batch" and q[0] != rid:
                    issue("FAIL", "QUEUE_FIFO_VIOLATION", rid, ref)
                q.remove(rid)
    if any(queues.values()):
        issue("UNKNOWN", "QUEUE_NOT_DRAINED", queues, [])
    for row, ref in events:
        source = (row.get("git") or {}).get("source_tree_sha256")
        if source != current_source.get("source_tree_sha256"):
            issue("UNKNOWN", "CURRENT_SOURCE_BINDING", "historical or missing source identity", ref)
            break
    if not events:
        issue("UNKNOWN", "TRACE_EMPTY", "no raw observations", [])
    status = "FAIL" if any(i["status"] == "FAIL" for i in issues) else "UNKNOWN" if issues else "PASS"
    return {"version": 1, "status": status, "scope": "observed six-request identity/boundary conformance only",
        "config_sha256": config.sha256, "trace_sha256": trace_sha256(expected), "chains": mappings,
        "issues": issues, "runtime_requests": len(runtime), "client_records": len(clients),
        "unlinked_runtime": sorted(runtime - mapped), "full_runtime_equivalence": "UNKNOWN"}


def historical_difference(directory):
    """The cancellation transport probe is a separate JSON object, not a row.

Do not invent its missing planned/client/terminal-result IDs. The explicit
lease and terminal_counters explain counting scope only, not a seven-edge chain.
"""
    directory = Path(directory)
    events = referenced(directory / "runtime-trace.jsonl")
    clients = referenced(directory / "requests.jsonl")
    probe_path = directory / "probe.json"
    probe = json.loads(probe_path.read_text())
    cancel = probe.get("faults", {}).get("cancel", {})
    runtime_ids = {r["request_id"] for r, _ in events if r.get("kind") == "submit_end" and r.get("error") is None}
    client_ids = {str((r.get("authoritative_usage") or {}).get("engine_request_id")) for r, _ in clients}
    terminals = cancel.get("terminal_counters", [])
    evidence = {"file": "probe.json", "sha256": sha(probe_path), "pointer": "/faults/cancel",
                "line": directory.joinpath("probe.json").read_text().splitlines().index('    "cancel": {') + 1}
    candidates = [t for t in terminals if str(t.get("engine_request_id")) in runtime_ids - client_ids
                  and str(t.get("engine_request_id")) == str(cancel.get("lease_id"))]
    return {"status": "PASS" if len(candidates) == 1 and len(runtime_ids - client_ids) == 1 else "UNKNOWN",
        "runtime_submits": len(runtime_ids), "client_jsonl_rows": len(clients),
        "missing_from_client_jsonl": sorted(runtime_ids - client_ids),
        "separate_cancel_probe": {"lease_id": cancel.get("lease_id"), "response_id": cancel.get("response_id"),
            "terminal_counters": terminals, "evidence": evidence},
        "interpretation": "separate fault-probe record explains count scope; no fabricated JSONL row",
        "seven_layer_chain": "UNKNOWN", "source_binding": "HISTORICAL"}


BOUNDARY_RULES = {
    "http_begin": ("HTTP_INGRESS", ("http_exchange_id", "planned_request_id", "client_record_id")),
    "request_created": ("ADMISSION_ID_BIND", ("http_exchange_id", "request_id", "request_object_identity", "queue_identity")),
    "admission_begin": ("ADMISSION_ATTEMPT", ("http_exchange_id", "admission_id")),
    "admission_accept": ("ADMISSION_ACCEPT", ("request_id", "request_object_identity", "lease_id")),
    "admission_reject": ("ADMISSION_REJECT", ("admission_id", "error_type")),
    "queue_enter": ("ADMISSION_QUEUE_ENTER", ("request_id", "request_object_identity", "queue_identity")),
    "queue_leave": ("ADMISSION_QUEUE_LEAVE", ("request_id", "request_object_identity", "queue_identity", "operation")),
    "command_enqueue": ("OWNER_ENQUEUE_ATTEMPT", ("request_id", "request_object_identity", "command_id", "future_identity")),
    "command_reject": ("OWNER_ENQUEUE_REJECT", ("request_id", "command_id", "error_type")),
    "command_start": ("OWNER_COMMAND_START", ("request_id", "request_object_identity", "command_id", "delivery_identity")),
    "runtime_handoff": ("RUNTIME_CONTEXT_BIND", ("request_id", "request_object_identity", "runtime_context_id", "context_object_identity")),
    "command_finish": ("OWNER_COMPLETION_CONFIRM", ("request_id", "request_object_identity", "command_id", "terminal_result_id", "has_cache_ref")),
    "response_identity": ("HTTP_RESPONSE_BIND", ("http_exchange_id", "response_id")),
    "http_headers": ("HTTP_HEADERS", ("http_exchange_id", "status", "lease_id")),
    "http_end": ("HTTP_HANDLER_RETURN", ("http_exchange_id",)),
    "disconnect": ("HTTP_DISCONNECT_OBSERVED", ("http_exchange_id",)),
    "cancel_intent": ("CANCEL_SIGNAL_SET", ("request_id", "request_object_identity", "lease_id")),
    "finalizer_complete": ("FINALIZER_CONFIRM", ("request_id", "request_object_identity", "lease_id", "released", "has_cache_ref")),
    "owner_shutdown": ("OWNER_SHUTDOWN_REQUEST", ("owner_identity", "status")),
    "owner_stopped": ("OWNER_STOP_FUTURE", ("owner_identity", "confirmed", "status")),
}


def boundary_mapping(events, result):
    """Per-boundary evidence predicates plus complete-chain relational findings.

These rules do not mutate the old transition projection or certify a complete
pre/post state. Missing chain evidence cannot be bypassed by field presence.
"""
    mapped = []
    def mentioned(evidence, line):
        if isinstance(evidence, dict):
            return evidence.get("line") == line or any(mentioned(v, line) for v in evidence.values())
        return isinstance(evidence, list) and any(mentioned(v, line) for v in evidence)
    for row, ref in events:
        kind = row.get("kind", "")
        if not kind.startswith("identity_"):
            continue
        name = kind.removeprefix("identity_")
        rule, fields = BOUNDARY_RULES.get(name, ("UNMODELED_BOUNDARY", ()))
        issues = [i for i in result["issues"] if mentioned(i.get("evidence"), ref["line"])]
        missing = [f for f in fields if row.get(f) is None]
        if missing or rule == "UNMODELED_BOUNDARY":
            issues.append({"status": "UNKNOWN", "code": "BOUNDARY_FIELDS_MISSING", "fields": missing})
        if name in {"command_start", "command_finish"}:
            linked = [e for e, _ in events if e.get("kind") == "identity_command_enqueue"
                      and e.get("command_id") == row.get("command_id")]
            if len(linked) != 1:
                issues.append({"status": "UNKNOWN", "code": "COMMAND_ENQUEUE_UNLINKED"})
            elif any(linked[0].get(k) != row.get(k) for k in ("request_id", "request_object_identity", "future_identity")):
                issues.append({"status": "FAIL", "code": "COMMAND_IDENTITY_CONFLICT"})
        status = "FAIL" if any(i["status"] == "FAIL" for i in issues) else "UNKNOWN" if issues else "PASS"
        mapped.append({**ref, "kind": kind, "rule_id": rule, "rule_version": 1,
                       "status": status, "issues": issues, "scope": "boundary evidence predicate, not full transition equivalence"})
    return mapped


def audit(directory, output, config, source):
    directory, output = Path(directory), Path(output)
    # Verify the original trace seal BEFORE any output directory is created.
    index = json.loads((directory / "trace-index.json").read_text())
    if not {"runtime-trace.jsonl", "trace-end.json"} <= index.get("sha256", {}).keys():
        raise ValueError("required trace hashes missing")
    for name, digest in index["sha256"].items():
        if Path(name).name != name or sha(directory / name) != digest:
            raise ValueError("raw evidence SHA-256 mismatch: " + name)
    events, clients = referenced(directory / "runtime-trace.jsonl"), referenced(directory / "requests.jsonl")
    # Client file must also have a saved seal, not a hash invented at audit time.
    seal = json.loads((directory / "identity-inputs.json").read_text())
    if not {"runtime-trace.jsonl", "trace-end.json", "trace-index.json", "requests.jsonl", "replay.json", "observer.jsonl"} <= seal.get("sha256", {}).keys():
        raise ValueError("required identity input hashes missing")
    for name, digest in seal["sha256"].items():
        if Path(name).name != name or sha(directory / name) != digest:
            raise ValueError("identity input SHA-256 mismatch: " + name)
    terminals = referenced(directory / "observer.jsonl") if (directory / "observer.jsonl").exists() else []
    result = reconstruct(config, events, clients, current_source=source, terminals=terminals)
    from .runner import is_idle
    replay = json.loads((directory / "replay.json").read_text()) if (directory / "replay.json").exists() else {}
    final = (replay.get("shutdown") or {}).get("final") or {}
    if not (is_idle((replay.get("after") or {}).get("status") or {}, expected_mode="continuous") and
            (replay.get("shutdown") or {}).get("stopped") is True and
            final.get("owner_thread_alive") is False and final.get("pending_finalizers") == 0 and
            final.get("stream_consumers") == 0):
        result["issues"].append({"status": "UNKNOWN", "code": "FINAL_DRAIN_UNCONFIRMED",
            "evidence": {"file": "replay.json", "sha256": sha(directory / "replay.json") if replay else None}})
        if result["status"] != "FAIL":
            result["status"] = "UNKNOWN"
    ending = json.loads((directory / "trace-end.json").read_text())
    if ending.get("errors"):
        result["issues"].append({"status": "UNKNOWN", "code": "OBSERVER_FAILED", "evidence": ending["errors"]})
        if result["status"] != "FAIL":
            result["status"] = "UNKNOWN"
    local = map_events([r for r, _ in events], input_sha256=sha(directory / "runtime-trace.jsonl"))
    # New boundary kinds remain separately checked, never relabeled in the old
    # checker. Preserve its UNKNOWNs and export every event/rule correspondence.
    output.mkdir(exist_ok=False)
    write_new(output / "identity.json", result)
    with_journal = Journal(output / "event-mapping.jsonl")
    try:
        for row in local["mapping"]:
            with_journal.append(row)
    finally:
        with_journal.close()
    write_new(output / "mapping-summary.json", {k: v for k, v in local.items() if k != "mapping"})
    boundaries = boundary_mapping(events, result)
    write_new(output / "boundary-mapping.json", {"rules": BOUNDARY_RULES, "mapping": boundaries,
        "counts": dict(Counter(r["status"] for r in boundaries))})
    write_new(output / "combined-mapping-summary.json", {"counts": dict(Counter(
        [r["status"] for r in local["mapping"] if not (r.get("kind") or "").startswith("identity_")] +
        [r["status"] for r in boundaries])), "total_raw_events": len(events),
        "scope": "old local rules plus explicit identity-boundary predicates; old report preserved unchanged"})
    local_failure = local["status"] == "FAIL" or any(r["status"] == "FAIL" for r in boundaries)
    write_new(output / "admission.json", {"admitted": False,
        "status": "FAIL" if result["status"] == "FAIL" or local_failure else "UNKNOWN",
        "formal_gpu_runs": 0, "identity_replay_status": result["status"],
        "unclosed_reason_codes": sorted(set(i["code"] for i in result["issues"]) |
            ({"LOCAL_TRANSITION_VIOLATION"} if local_failure else set()) | {
            "FROZEN_SCHEDULE_EQUIVALENCE", "FINITE_STATE_EXHAUSTION", "TAXONOMY_V2_STEP_RELATION",
            "CACHE_OWNERSHIP", "OBSERVED_ABSTRACT_STATE_DOMAIN", "V2_CLEANUP_RETRY_REPRESENTATION",
            "PREFLIGHT_DECLARED_PASS", "PREFLIGHT_WARMUP_SHAPES_COVERED", "PREFLIGHT_WARMUP_COVERAGE",
            "PREFLIGHT_SHAPE_PROOF_RECORDED"}),
        "scope": "not a new GPU preflight certificate; all earlier blockers remain in force"})
    lines = ["# Frozen correctness replay and identity audit", "", "Conclusion: **PARTIALLY CLOSED**.",
        f"Observed identity/replay status: **{result['status']}**. No performance admission.", "",
        f"Config SHA256: `{config.sha256}`.", f"Source SHA256: `{source.get('source_tree_sha256')}`.", "",
        "## Seven-layer identity chains", "",
        "| Planned | Client record | Lease | Engine | Context | Worker | Terminal result |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in result["chains"]:
        lines.append("| " + " | ".join(str(c[k]) for k in ("planned_request_id", "client_record_id", "http_lease_id",
            "engine_request_state_id", "runtime_context_id", "worker_request_id", "terminal_result_id")) + " |")
    lines += ["", "## Unclosed replay findings", ""]
    lines += [f"- {i['status']}: `{i['code']}` — {i.get('detail', 'see identity.json evidence')}" for i in result["issues"]]
    lines += ["", "Full source-linked evidence references are in identity.json; old local-rule UNKNOWNs are retained.",
        "This finite observed replay is not an all-path proof. Historical PREFLIGHT and global model blockers remain."]
    with (output / "report.md").open("x", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n")
    write_new(output / "artifacts.json", {"sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "inputs": {p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl"}}})
    return result
