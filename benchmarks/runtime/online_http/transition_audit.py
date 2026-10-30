"""Exclusive, model-only transition audit; never launches an HTTP/GPU workload."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .artifacts import ROOT, Journal, git_snapshot, write_new
from .config import load_config
from .workload import build_trace, trace_sha256
from .transition_relation import SCHEMA, map_events
from .retirement_composition import explore, check_composition
from .trace_check import check_directory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_equivalence(config, rows, clients, current_source):
    """Assess explicit observations, not equality of runtime durations.

    Warmup/fault probes stay in the result but never become measured replay.
    A matching observed path is not a proof for all possible arrival interleavings.
    """
    expected = build_trace(config)
    frozen = {r["request_id"]: r for r in expected}
    runtime_ids = {r.get("request_id") for r in rows if r.get("kind") == "submit_end" and r.get("error") is None}
    runtime_pids = {r.get("pid") for r in rows}
    historical = not rows or (rows[0].get("git") or {}).get("source_tree_sha256") != current_source.get("source_tree_sha256")
    checks = [{"code": "CURRENT_SOURCE_BINDING", "status": "UNKNOWN" if historical else "PASS",
               "classification": "HISTORICAL_EVIDENCE" if historical else "CURRENT_SOURCE",
               "detail": "compare complete dirty-tree SHA, not HEAD alone"}]
    issues, links = [], []
    for ordinal, row in enumerate(clients, 1):
        usage = row.get("authoritative_usage") or {}
        engine_id = usage.get("engine_request_id")
        matched = engine_id is not None and str(engine_id) in runtime_ids
        links.append({"client_line": ordinal, "frozen_request_id": row.get("request_id"),
            "engine_request_id": engine_id, "response_id": row.get("response_id"), "lease_id": row.get("lease_id"),
            "phase": row.get("phase"), "probe": row.get("probe"), "group": row.get("group"),
            "status": "PASS" if matched else "UNKNOWN",
            "basis": "explicit authoritative_usage.engine_request_id, never ordinal/time association"})
        if row.get("pid", row.get("server_pid")) not in runtime_pids:
            issues.append({"line": ordinal, "classification": "OBSERVATION_INSUFFICIENT", "detail": "client/server PID association missing or differs"})
    grants = [r for r in rows if r.get("kind") == "lease_granted"]
    linked_engines = {str(link["engine_request_id"]) for link in links if link["status"] == "PASS"}
    unlinked_runtime = sorted(runtime_ids - linked_engines)
    if unlinked_runtime:
        checks.append({"code": "CLIENT_RUNTIME_COVERAGE", "status": "UNKNOWN", "classification": "OBSERVATION_INSUFFICIENT",
                       "detail": "Runtime requests lack corresponding authoritative client records; origin not inferred",
                       "engine_request_ids": unlinked_runtime})
    linked_names = {str(link["engine_request_id"]): link["frozen_request_id"] for link in links
                    if link["status"] == "PASS"}
    admission_order = [{"event_id": r.get("event_id"), "seq": r.get("seq"),
                        "engine_request_id": r.get("request_id"), "lease_id": r.get("lease_id"),
                        "frozen_request_id": linked_names.get(str(r.get("request_id")))} for r in grants]
    checks.append({"code": "FROZEN_ADMISSION_CORRESPONDENCE", "status": "UNKNOWN",
                   "classification": "OBSERVATION_INSUFFICIENT",
                   "detail": "explicit queue arrival/removal identities and a tied/network-arrival refinement are not completely observed"})
    measured = [r for r in clients if r.get("phase") == "measured"]
    if not measured:
        checks.append({"code": "FROZEN_MEASURED_REPLAY", "status": "UNKNOWN", "classification": "PROTOCOL_SCOPE_UNDEFINED",
                       "detail": "no measured fixed-arrival replay exists; probe/warmup groups are not the frozen arrival schedule"})
    else:
        identities = [r.get("request_id") for r in measured]
        complete = len(identities) == len(frozen) and len(set(identities)) == len(identities) and set(identities) == set(frozen)
        checks.append({"code": "FROZEN_REQUEST_SET", "status": "PASS" if complete else "FAIL",
                       "classification": "OBSERVED_CONFORMANCE" if complete else "PROTOCOL_MISMATCH",
                       "detail": "all measured records retained, including failure/timeout/cancellation"})
        for ordinal, row in enumerate(measured, 1):
            spec = frozen.get(row.get("request_id"))
            if spec and (row.get("scheduled_at_s") != spec["arrival_s"] or row.get("max_tokens") != spec["max_tokens"]):
                issues.append({"line": ordinal, "classification": "PROTOCOL_MISMATCH", "status": "FAIL",
                               "detail": "frozen scheduled arrival or generation limit changed"})
            lateness = row.get("send_lateness_ms")
            if lateness is None:
                issues.append({"line": ordinal, "classification": "OBSERVATION_INSUFFICIENT", "detail": "send lateness missing"})
            elif lateness > config.data["arrival_tolerance_ms"]:
                issues.append({"line": ordinal, "classification": "PROTOCOL_MISMATCH", "status": "FAIL",
                               "detail": "frozen send-lateness tolerance exceeded; not a scheduling-order inference"})
        # A single finite observed replay cannot prove all schedule choices.
        checks.append({"code": "FROZEN_ALL_REACHABLE_PATHS", "status": "UNKNOWN", "classification": "MODEL_RULE_MISSING",
                       "detail": "no complete refinement relation for HTTP arrival, queue membership and owner command interleavings"})
    if not rows or any(r.get("config_sha256") != config.sha256 for r in rows):
        checks.append({"code": "FROZEN_CONFIG_SCOPE", "status": "FAIL" if measured else "UNKNOWN", "classification": "PROTOCOL_MISMATCH" if measured else "PROTOCOL_SCOPE_UNDEFINED",
                       "detail": "input trace does not bind the frozen GPU configuration (e.g. synthetic CPU fixture)"})
    if not rows or any(any(r.get(k) != config.data[k] for k in ("model_revision", "dtype", "device")) for r in rows):
        checks.append({"code": "FROZEN_MODEL_SCOPE", "status": "UNKNOWN", "classification": "PROTOCOL_SCOPE_UNDEFINED",
                       "detail": "model revision/dtype/device do not identify the frozen model execution"})
    if not clients or any(link["status"] != "PASS" for link in links):
        checks.append({"code": "CLIENT_RUNTIME_IDENTITY", "status": "UNKNOWN", "classification": "OBSERVATION_INSUFFICIENT",
                       "detail": "one or more explicit client-to-runtime links unavailable"})
    else:
        checks.append({"code": "CLIENT_RUNTIME_IDENTITY", "status": "PASS", "classification": "OBSERVED_CONFORMANCE",
                       "detail": "all explicit engine IDs linked; no timestamp association"})
    return {"status": "FAIL" if any(c["status"] == "FAIL" for c in checks) or any(i.get("status") == "FAIL" for i in issues) else "UNKNOWN",
        "config_sha256": config.sha256, "trace_sha256": trace_sha256(expected), "arrival_tolerance_ms": config.data["arrival_tolerance_ms"],
        "expected_requests": len(expected), "runtime_requests": len(runtime_ids), "client_records": len(clients),
        "unlinked_runtime_requests": unlinked_runtime, "measured_records": len(measured),
        "checks": checks, "issues": issues, "request_links": links, "observed_admission_order": admission_order,
        "frozen_input_order": list(frozen), "historical": historical,
        "timing_semantics": "recorded planned arrivals checked directly; wall-clock duration differences do not establish order violations"}


def search_report():
    """Extend the already exhaustive bounded graph with traversal/SCC evidence."""
    graph = explore()
    states, edges = graph["states"], graph["transitions"]
    adjacency = [set() for _ in states]
    reverse = [set() for _ in states]
    for edge in edges:
        adjacency[edge["from"]].add(edge["to"])
        reverse[edge["to"]].add(edge["from"])
    visited, order = set(), []
    for root in range(len(states)):
        if root in visited:
            continue
        visited.add(root)
        stack = [(root, iter(sorted(adjacency[root])))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                order.append(node); stack.pop()
            elif child not in visited:
                visited.add(child); stack.append((child, iter(sorted(adjacency[child]))))
    assigned, components = set(), []
    for root in reversed(order):
        if root in assigned:
            continue
        pending, component = [root], []
        assigned.add(root)
        while pending:
            node = pending.pop(); component.append(node)
            for child in sorted(reverse[node]):
                if child not in assigned:
                    assigned.add(child); pending.append(child)
        components.append(sorted(component))
    cyclic = [c for c in components if len(c) > 1 or c[0] in adjacency[c[0]]]
    initial = [i for i, s in enumerate(states) if s["owner"] == "open" and s["busy"] == -1 and
               not s["worker_closed"] and all(r["life"] == "new" for r in s["requests"])]
    relations = []
    for state in states:
        relations.append({"slot_owners": {"0": f"request-{i}" for i, r in enumerate(state["requests"]) if r["slot"]},
            "reference_owners": {f"request-{i}": f"cache-{i}" for i, r in enumerate(state["requests"]) if r["reference"]},
            "boundary_phases": ["decode" if r["life"] == "running" else "waiting" if r["life"] in {"new", "waiting"} else "finished"
                                for r in state["requests"]]})
    return {**graph, "initial_state_ids": initial, "initial_total": len(initial),
        "duplicate_successor_visits": len(edges) - (len(states) - len(initial)),
        "scc_total": len(components), "cyclic_scc_total": len(cyclic), "cyclic_sccs": cyclic,
        "symbolic_relations": relations,
        "phase_boundary": "Prefill is an atomic edge; interior Prefill/partial installation not enumerated",
        "fixed_workload_exhaustion": "UNKNOWN", "whole_runtime_exhaustion": "UNKNOWN"}


def audit(directory, output, config_path, *, clients_path=None, artifact_root=None, admission_path=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.is_relative_to(directory):
        raise ValueError("output must not modify input evidence directory")
    index = json.loads((directory / "trace-index.json").read_text(encoding="utf-8"))
    for name in ("runtime-trace.jsonl", "trace-end.json"):
        if sha(directory / name) != index["sha256"][name]:
            raise ValueError("raw artifact SHA256 mismatch: " + name)
    raw = (directory / "runtime-trace.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    source, config = git_snapshot(), load_config(config_path)
    clients = []
    prepared_identity = {"status": "UNKNOWN", "reason": "no authenticated prepared trace supplied"}
    client_integrity = {"status": "UNKNOWN", "reason": "no raw client evidence supplied"}
    if clients_path is not None:
        path = Path(clients_path).resolve()
        clients = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        client_integrity = {"status": "UNKNOWN", "sha256": sha(path), "reason": "hash observed but no historical index provided"}
        if artifact_root is not None:
            root = Path(artifact_root).resolve()
            if not path.is_relative_to(root):
                raise ValueError("client evidence outside historical root")
            manifest = json.loads((root / "artifacts.json").read_text(encoding="utf-8"))
            name = path.relative_to(root).as_posix()
            if sha(path) != manifest["sha256"].get(name):
                raise ValueError("client raw evidence hash mismatch")
            client_integrity["status"] = "PASS"
            client_integrity["reason"] = "matches immutable historical artifacts.json"
            prepared_path = root / "expected-shapes.json"
            if sha(prepared_path) != manifest["sha256"].get("expected-shapes.json"):
                raise ValueError("prepared trace manifest hash mismatch")
            prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
            prepared_hash = trace_sha256(prepared["trace"])
            if prepared_hash != prepared.get("trace_sha256"):
                raise ValueError("prepared trace content hash mismatch")
            pristine = build_trace(config)
            projected = [{key: row.get(key) for key in expected} for row, expected in zip(prepared["trace"], pristine)]
            matches = len(prepared["trace"]) == len(pristine) and projected == pristine
            prepared_identity = {"status": "PASS" if matches else "FAIL", "file_sha256": sha(prepared_path),
                "prepared_trace_sha256": prepared_hash, "pristine_trace_sha256": trace_sha256(pristine),
                "reason": "prepared trace adds observed tokenizer fields; hashes are intentionally distinct",
                "observed_prompts": [{k: row.get(k) for k in ("request_id", "observed_prompt_tokens", "max_tokens", "arrival_s")}
                                     for row in prepared["trace"]]}
    mapping = map_events(rows, input_sha256=hashlib.sha256(raw).hexdigest())
    old_check = check_directory(directory)
    composition = check_composition(rows)
    frozen = frozen_equivalence(config, rows, clients, source)
    if prepared_identity["status"] == "FAIL":
        frozen["checks"].append({"code": "PREPARED_FROZEN_INPUTS", "status": "FAIL", "classification": "PROTOCOL_MISMATCH"})
        frozen["status"] = "FAIL"
    if clients and client_integrity["status"] != "PASS":
        frozen["checks"].append({"code": "CLIENT_RAW_INTEGRITY", "status": "UNKNOWN", "classification": "OBSERVATION_INSUFFICIENT"})
    archived = None
    if admission_path is not None:
        archived = {"sha256": sha(admission_path), "kind": "historical-only; not a current-source certificate",
                    "report": json.loads(Path(admission_path).read_text(encoding="utf-8"))}
    checks = [
        {"code": "ABSTRACT_OBSERVED_TRANSITIONS", "status": mapping["status"]},
        {"code": "FROZEN_SCHEDULE_EQUIVALENCE", "status": frozen["status"]},
        {"code": "FINITE_STATE_EXHAUSTION", "status": "UNKNOWN", "detail": "bounded graph exhausted; not fixed six-request or whole Runtime"},
        {"code": "TAXONOMY_V2_STEP_RELATION", "status": "UNKNOWN", "detail": "resource-aware overlay is not a proof that v2.transitions represents every concrete step"},
        {"code": "INDEPENDENT_COMPOSITION", "status": composition["status"]},
        {"code": "INDEPENDENT_RAW_CHECKS", "status": old_check["status"],
         "detail": "legacy UNKNOWN/FAIL remains independent; it is not overridden by overlay conformance"},
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "source-inventory.json", source)
    write_new(output / "rules.json", SCHEMA)
    with_mapping = mapping.pop("mapping")
    journal = Journal(output / "event-mapping.jsonl")
    try:
        for row in with_mapping:
            journal.append(row)
    finally:
        journal.close()
    write_new(output / "search.json", search_report())
    write_new(output / "frozen-equivalence.json", frozen)
    write_new(output / "inconsistencies.json", {
        "event_findings": [{"line": row["line"], "event_id": row["event_id"], "rule_id": row["rule_id"],
                            "input_sha256": mapping["input_sha256"], "status": row["status"], "issues": row["issues"]}
                           for row in with_mapping if row["status"] != "PASS"],
        "frozen_findings": [c for c in frozen["checks"] if c["status"] != "PASS"] + frozen["issues"],
        "scope": "historical flag in frozen-equivalence.json applies to all raw event findings"})
    result = {"version": 1, "closure": "PARTIALLY CLOSED", "admitted": False, "formal_gpu_runs": 0,
        "mapping_summary": mapping, "checks": checks, "legacy_raw_checker": old_check,
        "composition_checker": composition, "historical_admission": archived,
        "current_source": {k: source[k] for k in ("sha", "dirty", "source_tree_sha256")},
        "input_identity": {k: rows[0].get(k) for k in ("run_id", "pid", "git", "config_sha256", "model_revision", "dtype", "device")} if rows else {},
        "client_integrity": client_integrity,
        "prepared_identity": prepared_identity,
        "config_file_sha256": sha(config_path), "config_sha256": config.sha256,
        "frozen_trace_sha256": trace_sha256(build_trace(config)),
        "input_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "input_directory": directory.relative_to(ROOT).as_posix() if directory.is_relative_to(ROOT) else directory.name,
        "admission_effect": "NONE; all legacy gates preserved; observed conformance is not bisimulation"}
    if not rows:
        result["input_identity"] = {}
    write_new(output / "audit.json", result)
    pending_checks = [{**check, "scope": "model_audit_not_gpu_preflight"} for check in checks if check["status"] != "PASS"]
    pending_checks += [{**check, "scope": "historical_raw" if frozen["historical"] else "current_cpu_raw"}
                       for check in old_check.get("checks", []) if check["status"] != "PASS"]
    pending_checks += [{**check, "scope": "composition_observations"} for check in composition.get("checks", [])
                       if check["status"] != "PASS"]
    if archived:
        pending_checks += [{**check, "scope": "historical_admission_archive_not_new_gate_run"}
                           for check in archived["report"].get("checks", []) if check["status"] != "PASS"]
    blocked = any(c["status"] == "FAIL" for c in pending_checks)
    write_new(output / "admission.json", {"status": "FAIL" if blocked else "UNKNOWN", "admitted": False,
        "current_gpu_preflight_executed": False, "formal_gpu_runs": 0,
        "status_semantics": "fail-closed eligibility summary, not a newly executed GPU gate",
        "unclosed_checks": pending_checks, "historical_admission_sha256": archived["sha256"] if archived else None,
        "remaining_requirement": "current-source GPU evidence plus complete frozen refinement/exhaustion; not established here"})
    lines = ["# Abstract transition / frozen workload audit", "", "## Identity and scope", "",
        f"Closure: **{result['closure']}**. Formal GPU runs: 0; admitted: false.",
        f"Raw SHA256: `{result['input_raw_sha256']}`.",
        f"Current source: `{source['source_tree_sha256']}`; historical evidence: {frozen['historical']}.", "",
        "## Observed event mapping", "", f"{mapping['counts']}",
        "Every raw event has a line/event/input-hash reference in event-mapping.jsonl.",
        "Projection fields are last direct observations; gaps are not inferred from time proximity.", "",
        "## Frozen workload", "", f"Measured records: {frozen['measured_records']}; result: {frozen['status']}.",
        "Warmup/fault records are retained but cannot prove frozen measured arrivals.", "",
        "## Search and proof boundary", "", "search.json reuses the bounded two-request graph, adds symbolic ownership and iterative SCC analysis.",
        "Prefill interiors, six-request frozen arrivals and full Runtime equivalence are not exhausted.", "",
        "## Admission", "", "Legacy UNKNOWN/FAIL results are preserved in audit.json. No gate was promoted.",
        "Permanent cleanup failure admits infinite retries; liveness requires fairness and eventual success."]
    with (output / "report.md").open("x", encoding="utf-8") as destination:
        destination.write("\n".join(lines) + "\n")
    write_new(output / "artifacts.json", {"sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "input_sha256": {"runtime-trace.jsonl": sha(directory / "runtime-trace.jsonl"),
                         "trace-end.json": sha(directory / "trace-end.json"), "trace-index.json": sha(directory / "trace-index.json")}})
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--clients", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args()
    result = audit(args.input, args.output, args.config, clients_path=args.clients,
                   artifact_root=args.artifact_root, admission_path=args.admission)
    print(json.dumps({"closure": result["closure"], "mapping": result["mapping_summary"]["counts"],
                      "admitted": result["admitted"]}))
