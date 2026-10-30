"""Observed-path refinement, not a reachable-state or performance experiment.

Taxonomy v2 is evaluated at explicit submit/step boundaries. Intermediate
resource operations retain their joint-model obligations; they are not erased
by declaring every unmodeled event a stutter.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .artifacts import git_snapshot, write_new, Journal
from .config import load_config
from .identity_check import referenced, reconstruct, boundary_mapping
from .kv_evolution import check_forwards, verdict
from .retirement import check_retirement
from .shapes import shape_key
from .taxonomy import build_taxonomy, initial_state, transitions, SOURCE_FILES
from .transition_relation import map_events
from .workload import build_trace, trace_sha256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event_references(pairs):
    """Attach the event's OWN ID to its physical line reference; never an ordinal ID."""
    return [(row, {**ref, "event_id": row.get("event_id")}) for row, ref in pairs]


def source_binding(target, current):
    return {"full_source": "PASS" if current.get("source_tree_sha256") == target.get("source_tree_sha256") and
            target.get("source_tree_sha256") is not None else "UNKNOWN",
            "production_rule_source_hashes_equal": all(current.get("files", {}).get(p) == target.get("files", {}).get(p) and
                target.get("files", {}).get(p) is not None for p in SOURCE_FILES),
            "scope": "unchanged production file hashes cannot promote an old full-source certificate"}


def match_boundary(taxonomy, before, after, shapes, terminal_lives):
    """One exact model edge, or iteration + observed terminal suffixes only.

At most active capacity terminal suffixes are considered, without BFS budgets
or arbitrary extra arrivals. Early STOP after a generated token is not LENGTH.
"""
    try:
        candidates = transitions(taxonomy, before)
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": "FAIL", "code": "INVALID_ABSTRACT_STATE", "detail": str(exc)}
    target_shapes = [shape_key(s) for s in shapes]
    explored = []
    for action, state, emitted in candidates:
        if [shape_key(s) for s in emitted] != target_shapes:
            continue
        if state == after:
            return {"status": "PASS", "rules": [action], "model_states": [before, state]}
        if action != "iteration":
            continue
        current, witness, states = state, [action], [before, state]
        for i in range(len(after)):
            if current[i] == after[i] or after[i][0] != "T":
                continue
            reason = {"finished": "stop", "cancelled": "cancel", "failed": "error"}.get(terminal_lives.get(i))
            desired = list(current); desired[i] = after[i]; desired = tuple(desired)
            valid = [(name, q) for name, q, output in transitions(taxonomy, current)
                     if name == reason and not output and q == desired]
            if len(valid) != 1:
                break
            witness.append(reason); current = valid[0][1]; states.append(current)
        if current == after:
            return {"status": "PASS", "rules": witness, "model_states": states}
        explored.append({"action": action, "candidate": current})
    return {"status": "FAIL", "code": "NO_ALLOWED_BOUNDARY_PATH", "candidates": explored,
            "classification": "model_counterexample_not_automatically_implementation_defect"}


def project_snapshot(snapshot, ids, submitted, acknowledged):
    """Project direct Context/Slot observations; do not guess absent objects.

T with a retained Slot/ref is deliberately unrepresentable in base v2. The
retirement overlay represents it separately; it is never silently projected away.
"""
    requests = snapshot.get("requests")
    worker = snapshot.get("worker") or {}
    slots, pending = worker.get("slots"), worker.get("retirements")
    if not isinstance(requests, dict) or not isinstance(slots, dict):
        return None, ["CONTEXT_OR_SLOT_SNAPSHOT_MISSING"], {}
    if (set(requests) | set(slots)) - set(ids):
        return None, ["UNREGISTERED_CONTEXT_OR_SLOT_ID"], {}
    state, unknown, lives = [], [], {}
    for i, rid in enumerate(ids):
        if rid not in submitted:
            state.append(("N", 0, 0)); continue
        request = requests.get(rid)
        if request is None:
            if rid in acknowledged:
                state.append(acknowledged[rid]); continue
            unknown.append("ABSENT_CONTEXT_WITHOUT_ACK:" + rid)
            state.append(None); continue
        life, count = request.get("status"), request.get("generated")
        lives[i] = life
        if type(count) is not int or request.get("request_object_identity") is None:
            unknown.append("REQUEST_COUNT_OR_ID_UNOBSERVED:" + rid); state.append(None); continue
        if life in {"finished", "failed", "cancelled"}:
            if rid in slots or request.get("has_cache_ref") is not False or isinstance(pending, dict) and rid in pending:
                unknown.append("V2_TERMINAL_WITH_PENDING_RESOURCE:" + rid)
            elif not isinstance(pending, dict):
                unknown.append("TERMINAL_RETIREMENT_RECORDS_UNOBSERVED:" + rid)
            state.append(("T", count, 0))
        elif life == "waiting" and request.get("phase") == "waiting" and count == 0:
            state.append(("W", 0, 0))
        elif life == "running" and request.get("phase") == "decode":
            value = slots.get(rid)
            width = (value.get("cache") or {}).get("length") if value else None
            if type(width) is not int or value.get("pending_first") is None:
                unknown.append("RUNNING_CACHE_OR_FIRST_FLAG_MISSING:" + rid); state.append(None)
            else:
                if value["pending_first"] != (count == 0):
                    unknown.append("FIRST_FLAG_COUNT_DISAGREEMENT:" + rid)
                state.append(("F" if count == 0 else "D", count, width))
        else:
            unknown.append("EXECUTION_PHASE_UNREPRESENTED:" + rid); state.append(None)
    return tuple(state) if not unknown else None, unknown, lives


def boundary_refinement(events, taxonomy, engine_ids):
    calls, submitted, acknowledged, results, closed = {}, set(), {}, [], set()
    ids = engine_ids
    unknown_ids = set()
    for row, ref in events:
        kind, op, rid = row.get("kind"), row.get("operation_id"), row.get("request_id")
        if kind in {"submit_begin", "step_begin", "acknowledge_begin"}:
            if op is not None:
                if op in calls:
                    results.append({"status": "FAIL", "code": "DUPLICATE_OPERATION_BEGIN", "evidence": ref})
                else:
                    calls[op] = (deepcopy(row), ref, set(submitted), deepcopy(acknowledged))
            else:
                results.append({"status": "UNKNOWN", "code": "OPERATION_ID_UNOBSERVED", "evidence": ref})
            continue
        if kind not in {"submit_end", "step_end", "acknowledge_end"}:
            continue
        before = calls.get(op) if op is not None else None
        if before is None:
            results.append({"status": "UNKNOWN", "code": "EXPLICIT_OPERATION_BEGIN_MISSING", "evidence": ref})
            continue
        if op in closed:
            results.append({"status": "FAIL", "code": "DUPLICATE_OPERATION_END", "evidence": ref})
            continue
        closed.add(op)
        start, bref, old_submitted, old_ack = before
        if start["kind"].removesuffix("_begin") != kind.removesuffix("_end"):
            results.append({"status": "FAIL", "code": "OPERATION_KIND_CONFLICT", "evidence": ref})
            continue
        a, missing_a, _ = project_snapshot(start.get("snapshot") or {}, ids, old_submitted, old_ack)
        if kind == "submit_end" and row.get("error") is None:
            if rid not in ids:
                unknown_ids.add(rid)
            submitted.add(rid)
        if kind == "acknowledge_end":
            old = (start.get("snapshot") or {}).get("requests", {}).get(rid)
            old_slots = ((start.get("snapshot") or {}).get("worker") or {}).get("slots")
            old_records = ((start.get("snapshot") or {}).get("worker") or {}).get("retirements")
            after_requests = (row.get("snapshot") or {}).get("requests")
            if (row.get("error") is None and old and old.get("status") in {"finished", "failed", "cancelled"}
                    and old.get("has_cache_ref") is False and isinstance(old_slots, dict) and rid not in old_slots
                    and isinstance(old_records, dict) and rid not in old_records
                    and isinstance(after_requests, dict) and rid not in after_requests):
                acknowledged[rid] = ("T", old["generated"], 0)
            else:
                results.append({"status": "UNKNOWN", "code": "ACK_NOT_CONFIRMED_BY_DIRECT_SNAPSHOTS", "evidence": ref})
        b, missing_b, lives = project_snapshot(row.get("snapshot") or {}, ids, submitted, acknowledged)
        shape_rows = []
        if kind == "step_end":
            shape_rows = [r for r, _ in events if r.get("kind") == "forward_begin" and
                          r.get("iteration_id") == row.get("iteration_id")]
        shapes = [{"mode": "continuous", "phase": r.get("phase"), "batch_size": len(r.get("members") or []),
            "input_shape": r.get("input_shape"), "attention_mask_shape": r.get("mask_shape"),
            "padding_width": r.get("input_shape", [None, None])[-1],
            "kv_length_before": (r.get("cache_before") or {}).get("length"),
            "kv_tensor_shapes_before": (r.get("cache_before") or {}).get("layers")} for r in shape_rows]
        if a is None or b is None or unknown_ids:
            match = {"status": "UNKNOWN", "code": "PROJECTION_UNDEFINED", "reasons": missing_a + missing_b,
                     "unknown_request_ids": sorted(unknown_ids, key=str)}
        elif row.get("error") is not None:
            match = {"status": "UNKNOWN", "code": "FAILED_MACRO_PARTIAL_EXECUTION", "detail": "v2 normal atomic iteration insufficient; joint overlay retained"}
        elif kind == "acknowledge_end":
            match = {"status": "PASS" if a == b else "FAIL", "rules": ["acknowledge-retired-context-stutter"]}
        else:
            match = match_boundary(taxonomy, a, b, shapes, lives)
        results.append({**match, "operation_id": op, "iteration_id": row.get("iteration_id"), "kind": kind,
            "before": a, "after": b, "before_direct": start.get("snapshot"), "after_direct": row.get("snapshot"),
            "evidence": {"begin": bref, "end": ref}, "forward_ids": [r.get("forward_id") for r in shape_rows],
            "scope": "execution projection at explicit boundaries, not per-instruction bisimulation"})
    for op in calls.keys() - closed:
        results.append({"status": "UNKNOWN", "code": "OPERATION_END_UNOBSERVED", "operation_id": op, "evidence": calls[op][1]})
    return {"status": verdict(results) if results else "UNKNOWN", "mapping": results,
        "counts": dict(Counter(r["status"] for r in results)), "submitted": sorted(submitted, key=str),
        "acknowledged": sorted(acknowledged, key=str), "initial_state": initial_state(taxonomy)}


def audit(input_root, output, config_path):
    """Authenticate immutable saved run, then recalculate all local predicates."""
    input_root, output = Path(input_root), Path(output)
    if output.resolve().is_relative_to(input_root.resolve()):
        raise ValueError("output must be outside immutable input run")
    index = json.loads((input_root / "artifacts.json").read_text(encoding="utf-8"))
    for required in ("source.json", "workload.json", "config.json", "continuous/runtime-trace.jsonl",
                     "continuous/requests.jsonl", "continuous/observer.jsonl", "continuous/trace-end.json"):
        if required not in index["sha256"] or sha(input_root / required) != index["sha256"][required]:
            raise ValueError("historical raw hash mismatch or missing index: " + required)
    source = json.loads((input_root / "source.json").read_text(encoding="utf-8"))
    source_digest = hashlib.sha256(json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if source_digest != source["source_tree_sha256"]:
        raise ValueError("source inventory self hash mismatch")
    config = load_config(config_path)
    current = git_snapshot()
    directory = input_root / "continuous"
    events, clients, terminals = [referenced(directory / name) for name in ("runtime-trace.jsonl", "requests.jsonl", "observer.jsonl")]
    events = event_references(events)
    ending = json.loads((directory / "trace-end.json").read_text(encoding="utf-8"))
    identity = reconstruct(config, events, clients, current_source=source, terminals=terminals)
    bmap = boundary_mapping(events, identity)
    local = map_events([r for r, _ in events], input_sha256=sha(directory / "runtime-trace.jsonl"))
    retirement = check_retirement([r for r, _ in events])
    kv = check_forwards(events)
    workload = json.loads((input_root / "workload.json").read_text(encoding="utf-8"))
    if workload.get("pristine_sha256") != trace_sha256(build_trace(config)) or workload.get("prepared_sha256") != trace_sha256(workload["requests"]):
        raise ValueError("frozen workload hash mismatch")
    for expected, actual in zip(build_trace(config), workload["requests"]):
        if any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError("frozen request fields changed")
    if len(workload["requests"]) != len(build_trace(config)):
        raise ValueError("frozen request count changed")
    geometries = {tuple(f["observed_geometry"]) for f in kv["forwards"] if f.get("observed_geometry") is not None}
    if len(geometries) == 1:
        layers, heads, dim = next(iter(geometries))
        manifest = {"trace": workload["requests"], "geometry": {"layers": layers, "kv_heads": heads, "head_dim": dim},
                    "trace_sha256": workload["prepared_sha256"], **{k: events[0][0][k] for k in
                        ("config_sha256", "model_revision", "dtype", "device", "git")}}
        taxonomy = build_taxonomy(manifest, source["files"], capacity=config.data["admission_limit"])
        by_planned = {c["planned_request_id"]: c["engine_request_state_id"] for c in identity["chains"]}
        ids = [by_planned.get(r["request_id"]) for r in workload["requests"]]
        boundary = boundary_refinement(events, taxonomy, ids)
    else:
        taxonomy = {"status": "UNKNOWN", "reason": "OBSERVED_GEOMETRY_UNKNOWN_OR_CONFLICT", "geometries": sorted(geometries)}
        boundary = {"status": "UNKNOWN", "mapping": [], "counts": {}, "reason": taxonomy["reason"]}
    binding = source_binding(source, current)
    boundary_by_end = {r.get("evidence", {}).get("end", {}).get("event_id"): r for r in boundary["mapping"]
                       if r.get("evidence", {}).get("end", {}).get("event_id") is not None}
    boundary_local = {r["line"]: r for r in bmap}
    local_by_line = {r["line"]: r for r in local["mapping"]}
    event_refinement = []
    for raw, ref in events:
        guard = boundary_local.get(ref["line"], local_by_line.get(ref["line"], {}))
        macro = boundary_by_end.get(raw.get("event_id"))
        local_row = local_by_line.get(ref["line"], {})
        event_refinement.append({"evidence": {**ref, "event_id": raw.get("event_id")},
            "kind": raw.get("kind"), "rule_id": guard.get("rule_id", "UNMODELED"),
            "local_conformance": guard.get("status", "UNKNOWN"),
            "local_issues": guard.get("issues", []),
            "before_direct_observations": local_row.get("before"),
            "after_direct_observations": local_row.get("after"),
            "observation_semantics": "field-source snapshots, not inferred complete adjacent global states",
            "execution_boundary_refinement": macro.get("status") if macro else "UNKNOWN",
            "abstract_rule_path": macro.get("rules") if macro else None,
            "uncovered_reason": None if macro and macro.get("status") == "PASS" else
                macro.get("code", "BOUNDARY_PROJECTION_UNKNOWN") if macro else
                "NO_INDIVIDUAL_V2_EDGE_ASSERTED: metadata observation or internal phase/resource step; retain joint overlay",
            "macro_evidence": macro.get("evidence") if macro else None})
    result = {"closure": "PARTIALLY CLOSED", "admitted": False, "formal_gpu_runs": 0,
        "input_root": str(input_root), "input_raw_sha256": sha(directory / "runtime-trace.jsonl"),
        "target_source": source["source_tree_sha256"], "auditor_source": current["source_tree_sha256"],
        "current_source_binding": binding["full_source"],
        "production_rule_source_hashes_equal": binding["production_rule_source_hashes_equal"],
        "identity_status_historical_source": identity["status"], "forward_physical": kv["physical_counts"],
        "forward_complete": kv["counts"], "boundary_execution_projection": boundary["status"],
        "boundary_counts": boundary["counts"], "collector_errors": ending.get("errors"),
        "event_total": len(events),
        "event_local_counts": dict(Counter(r["local_conformance"] for r in event_refinement)),
        "event_execution_boundary_counts": dict(Counter(r["execution_boundary_refinement"] for r in event_refinement)),
        "full_trace_refinement": "FAIL" if boundary["status"] == "FAIL" or kv["status"] == "FAIL" or retirement["status"] == "FAIL" else "UNKNOWN",
        "FROZEN_SCHEDULE_EQUIVALENCE": "UNKNOWN", "FINITE_STATE_EXHAUSTION": "UNKNOWN",
        "TAXONOMY_V2_STEP_RELATION": "UNKNOWN", "FORWARD_KV_EVOLUTION": kv["status"],
        "scope": "historical observed path only; effective-mask and tensor-value lineage obligations remain"}
    output.mkdir(exist_ok=False)
    for name, value in (("audit.json", result), ("identity.json", identity), ("kv.json", kv),
                        ("boundary-refinement.json", boundary), ("retirement.json", retirement),
                        ("taxonomy.json", taxonomy), ("boundary-mapping.json", bmap)):
        write_new(output / name, value)
    journal = Journal(output / "local-event-mapping.jsonl")
    try:
        for row in local["mapping"]:
            journal.append(row)
    finally:
        journal.close()
    journal = Journal(output / "event-refinement.jsonl")
    try:
        for row in event_refinement:
            journal.append(row)
    finally:
        journal.close()
    inherited = ["FROZEN_SCHEDULE_EQUIVALENCE", "FINITE_STATE_EXHAUSTION", "TAXONOMY_V2_STEP_RELATION",
        "CACHE_OWNERSHIP", "OBSERVED_ABSTRACT_STATE_DOMAIN", "V2_CLEANUP_RETRY_REPRESENTATION",
        "PREFLIGHT_DECLARED_PASS", "PREFLIGHT_WARMUP_SHAPES_COVERED", "PREFLIGHT_WARMUP_COVERAGE", "PREFLIGHT_SHAPE_PROOF_RECORDED"]
    write_new(output / "admission.json", {"admitted": False, "status": "FAIL" if result["full_trace_refinement"] == "FAIL" else "UNKNOWN",
        "formal_gpu_runs": 0, "checks": [
            {"code": "OBSERVED_PHYSICAL_KV_EQUATIONS", "status": verdict([{"status": f["physical_equations"]} for f in kv["forwards"]])},
            {"code": "FORWARD_KV_EVOLUTION", "status": kv["status"]},
            {"code": "OBSERVED_EXECUTION_BOUNDARY_PROJECTION", "status": boundary["status"]},
            {"code": "CURRENT_SOURCE_BINDING", "status": binding["full_source"]},
            *[{"code": code, "status": "UNKNOWN", "scope": "inherited obligation not re-certified"} for code in inherited]],
        "historical_failures": "immutable prior admission report remains FAIL; no new GPU preflight executed"})
    with (output / "report.md").open("x", encoding="utf-8") as report:
        report.write("# Transition refinement and KV evolution\n\nConclusion: **PARTIALLY CLOSED**.\n\n" +
            "## Observed path\n\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n\n" +
            "## Proof boundary\n\nPhysical tensor widths are not effective mask lengths. Split tensor-value lineage and model-forward call-ID links are absent.\n" +
            "Explicit execution boundaries can refine v2 while internal retirement and admission require the joint overlay.\n" +
            "No per-event global stutter is inferred from timestamp proximity or absent snapshots.\n" +
            "No all-path equivalence, exhaustive six-request search or performance admission is asserted.\n")
    write_new(output / "artifacts.json", {"sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.output, args.config), ensure_ascii=False))
