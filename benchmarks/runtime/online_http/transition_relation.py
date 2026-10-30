"""Resource-aware observation relation, never a Runtime state owner.

Only raw observations update the projection. Unobserved facts retain their last
observation line, not an assertion that they stayed unchanged. Span rules require
explicit call/phase/forward IDs; no timestamp or batch-position association.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from .shapes import digest
from .retirement import TERMINAL

VERSION = 1
RULES = {
    "BIND": "bind the actual request object to its explicit request ID",
    "PLAN_SELECT": "active decode order and created-time stable waiting order from observed batch inputs",
    "PLAN_TRIM": "retain decode and trim prefill by observed physical free slots",
    "EXECUTOR_DISPATCH": "dispatch exactly the selected phase members",
    "WORKER_DISPATCH": "Worker members equal the explicitly associated executor call",
    "WORKER_RESULT": "one phase-correct result per actual member",
    "PHASE_APPLY": "apply explicit Worker token/error results to observed Contexts",
    "FORWARD": "forward membership, input/mask/cache dimensions and growth",
    "CANCEL": "cancel terminalizes generation without assuming resource retirement",
    "RETIRE_BEGIN": "release attempt starts without losing original identity",
    "RETIRE_END": "check Slot absence, original reference and retry record separately",
    "SLOT_UPDATE": "apply observed per-request pool before/after ownership without clearing request references",
    "ACK": "forget only a terminal, confirmed-retired request",
    "LEASE": "finalize admission only after confirmed retirement",
    "OWNER": "observe running/draining/stopped separately from pending retirement",
    "CLOSE": "borrowed/owned backend close and unresolved cleanup",
    "OBSERVE": "supporting raw observation, not a claimed execution transition",
    "IDEMPOTENT": "reapplying the identical event does not execute its effect twice",
    "UNMODELED": "no rule; preserve UNKNOWN",
}
SCHEMA = {
    "version": VERSION, "kind": "resource-aware-transition-overlay-not-taxonomy-v2-replacement",
    "request_key": "raw request_id + original object identity, scoped to PID/run",
    "domains": {
        "life": ["waiting", "running", "finished", "failed", "cancelled", None],
        "phase": ["waiting", "prefill", "decode", "finished", None],
        "lease": ["held", "released", None],
        "resource": "JointState/ResourceStatus from existing retirement model; slot/ref evidence separate",
        "slot_owner": "explicit request_id -> slot_id -> cache_object_identity; missing is unknown",
        "owner": ["running", "draining", "stopped", None],
        "worker_ownership": [True, False, None],
        "retry": "attempt/error/confirmation observations; counters are not inferred",
        "queue_members": "UNKNOWN without explicit per-request queue evidence; counts are not membership",
    },
    "rules": RULES,
    "proof_boundary": "observed partial transitions, not bisimulation or full reachability",
}


def map_events(rows, *, input_sha256):
    facts, seen, calls, phase_submit, worker_calls, worker_results, plans, forwards = {}, {}, {}, {}, {}, {}, {}, {}
    confirmed, projections = {}, []
    for line, row in enumerate(rows, 1):
        kind, rid = row.get("kind"), row.get("request_id")
        issues, refs = [], []
        def unknown(detail, pointer="", category="OBSERVATION_INSUFFICIENT"):
            issues.append({"status": "UNKNOWN", "classification": category, "detail": detail, "field": pointer})
        def guard(condition, detail, pointer=""):
            if not condition:
                issues.append({"status": "FAIL", "classification": "IMPLEMENTATION_COUNTEREXAMPLE",
                               "detail": detail, "field": pointer})
        def fact(key, name, value):
            if key is None:
                unknown("request ID absent")
                return
            facts.setdefault(key, {})[name] = {"value": deepcopy(value), "line": line}
        def value(key, name):
            return facts.get(key, {}).get(name, {}).get("value")
        def identity(key, obj, pointer):
            if key is None:
                unknown("request ID absent", pointer)
                return
            if obj is None:
                unknown("original RequestState identity absent", pointer)
                return
            old = value(key, "object_identity")
            if old is not None and old != obj:
                guard(False, "original RequestState identity changed", pointer)
            fact(key, "object_identity", obj)
        def begin_key(name):
            key = row.get(name)
            if key is None:
                unknown("explicit call identity absent", name)
            return key
        def linked(table, key):
            old = table.get(key) if key is not None else None
            if old is None:
                unknown("explicitly linked begin/input observation absent")
            else:
                refs.append({"line": old["line"], "event_id": old["row"].get("event_id")})
            return old["row"] if old else None

        event_id = row.get("event_id")
        fingerprint = digest(row)
        if event_id in seen:
            previous_hash, previous_line = seen[event_id]
            projections.append({"line": line, "event_id": event_id, "rule_id": "IDEMPOTENT",
                "status": "PASS" if previous_hash == fingerprint else "FAIL", "input_sha256": input_sha256,
                "before": {}, "after": {}, "prior_event_line": previous_line,
                "issues": [] if previous_hash == fingerprint else [{"status": "FAIL",
                    "classification": "OBSERVATION_CONTRADICTION", "detail": "same event ID with different payload"}]})
            continue
        if event_id is None:
            unknown("event identity absent", "event_id")
        else:
            seen[event_id] = (fingerprint, line)
        before = deepcopy(facts)
        rule = "OBSERVE"
        snapshot = row.get("snapshot") or {}
        requests = snapshot.get("requests")
        if requests is None and kind == "retirement_evidence":
            requests = row.get("requests")
        if requests is None:
            requests = row.get("request_evidence")
        for key, request in (requests or {}).items():
            identity(key, request.get("request_object_identity"), "snapshot/requests/" + key)
            for source, name in (("status", "life"), ("phase", "phase"), ("generated", "tokens"),
                                 ("has_cache_ref", "cache_ref"), ("max_tokens", "max_tokens")):
                if source in request:
                    fact(key, name, request[source])
        worker = snapshot.get("worker", snapshot)
        slots, records = worker.get("slots"), worker.get("retirements")
        if isinstance(slots, dict):
            owners = [s.get("slot") for s in slots.values()]
            if None in owners:
                unknown("physical Slot owner ID unobservable", "snapshot/slots")
            guard(len([x for x in owners if x is not None]) == len(set(x for x in owners if x is not None)),
                  "physical Slot simultaneously owned by distinct requests", "snapshot/slots")
            candidates = set(slots) | set(requests or {}) | {
                key for key in facts if value(key, "slot_present") is not False}
            for key in candidates - {"@owner"}:
                fact(key, "slot_present", key in slots)
                fact(key, "slot_id", slots.get(key, {}).get("slot"))
                if key in slots:
                    fact(key, "cache_object_identity", slots[key].get("cache_object_identity"))
                    fact(key, "kv_length", slots[key].get("cache", {}).get("length"))
        if isinstance(records, dict):
            candidates = set(records) | set(requests or {}) | {
                key for key in facts if value(key, "cleanup_pending") is not False}
            for key in candidates - {"@owner"}:
                fact(key, "cleanup_pending", key in records)
                if key in records:
                    identity(key, records[key].get("request_object_identity"), "snapshot/retirements/" + key)
                    for name in ("attempts", "slot_released", "reference_cleared", "last_error"):
                        fact(key, name, records[key].get(name))
        if kind == "lease_granted":
            rule = "BIND"
            identity(rid, row.get("request_object_identity"), "request_object_identity")
            fact(rid, "lease_id", row.get("lease_id")); fact(rid, "lease", "held")
        elif kind == "submit_end":
            rule = "BIND"
            current = (requests or {}).get(rid)
            if row.get("error") is None:
                if current is None:
                    unknown("successful handoff Context not observed")
                else:
                    guard(current.get("status") == "waiting" and current.get("phase") == "waiting" and
                          current.get("generated") == 0, "new Context already executed before handoff")
        elif isinstance(kind, str) and kind.endswith("_begin") and kind.split("_begin")[0] in {"submit", "step", "cancel", "acknowledge", "close"}:
            key = begin_key("operation_id")
            if key:
                calls[key] = {"row": row, "line": line}
        elif kind == "scheduler_plan":
            rule = "PLAN_SELECT"
            key = begin_key("iteration_id")
            selection = row.get("selection_input")
            if not isinstance(selection, dict):
                unknown("batch selection inputs not observed", "selection_input")
            else:
                active, waiting = selection.get("active"), selection.get("waiting")
                limit = selection.get("max_active")
                if active is None or waiting is None or type(limit) is not int:
                    unknown("active/waiting membership or capacity absent")
                elif any(r.get("created_time") is None for r in waiting):
                    unknown("waiting FIFO timestamps absent")
                else:
                    for request in active + waiting:
                        identity(request.get("request_id"), request.get("request_object_identity"), "selection_input")
                    eligible = [r for r in waiting if r.get("status") not in TERMINAL]
                    ordered = sorted(eligible, key=lambda r: r["created_time"])
                    guard(row.get("decode") == [r["request_id"] for r in active], "decode selection differs from actual active order")
                    guard(row.get("prefill") == [r["request_id"] for r in ordered[:max(0, limit - len(active))]],
                          "prefill selection violates observed FIFO/capacity")
            plans[key] = {"row": row, "line": line}
        elif kind == "executor_plan":
            rule = "PLAN_TRIM"
            old = linked(plans, row.get("iteration_id"))
            free = row.get("free_slots")
            if old and type(free) is int:
                guard(row.get("decode") == old.get("decode") and row.get("prefill") == old.get("prefill", [])[:max(0, free)],
                      "executed plan not explained by physical free capacity")
            else:
                unknown("free capacity or plan absent")
            plans[(row.get("iteration_id"), "executed")] = {"row": row, "line": line}
        elif kind == "executor_submit":
            rule = "EXECUTOR_DISPATCH"
            old = linked(plans, (row.get("iteration_id"), "executed"))
            if old:
                guard(row.get("members") == old.get(row.get("phase")), "phase members differ from executed plan")
            key = begin_key("phase_id")
            phase_submit[key] = {"row": row, "line": line}
        elif kind == "worker_begin":
            rule = "WORKER_DISPATCH"
            old = linked(phase_submit, row.get("phase_id"))
            if old:
                guard(row.get("members") == old.get("members") and row.get("phase") == old.get("phase"), "Worker dispatch differs from Executor")
            worker_calls[row.get("phase_id")] = {"row": row, "line": line}
        elif kind == "worker_end":
            rule = "WORKER_RESULT"
            old = linked(worker_calls, row.get("phase_id"))
            results = row.get("results")
            if row.get("error") is None:
                if old and isinstance(results, list) and all(r.get("request_id") is not None for r in results):
                    ids = [r["request_id"] for r in results]
                    guard(len(ids) == len(set(ids)) and set(ids) == set(old.get("members", [])), "unknown, duplicate or missing Worker result")
                    guard(all(r.get("phase") == old.get("phase") for r in results), "Worker result phase mismatch")
                else:
                    unknown("complete Worker results unavailable")
            worker_results[row.get("phase_id")] = {"row": row, "line": line}
        elif kind == "executor_phase_end":
            rule = "PHASE_APPLY"
            old = linked(phase_submit, row.get("phase_id"))
            result = linked(worker_results, row.get("phase_id"))
            if old and result:
                start = (old.get("snapshot") or {}).get("requests", {})
                end = snapshot.get("requests", {})
                outputs = {r.get("request_id"): r for r in result.get("results") or []}
                for key in old.get("members", []):
                    a, b, output = start.get(key), end.get(key), outputs.get(key)
                    if not a or not b:
                        unknown("phase before/after Context missing")
                        continue
                    if result.get("error") is not None:
                        guard(b.get("status") in TERMINAL, "whole phase error left affected request active")
                    elif output is None:
                        unknown("request result missing")
                    elif output.get("error") is not None:
                        guard(b.get("status") == "failed", "request error was not terminalized as FAILED")
                    else:
                        if any(x is None for x in (a.get("generated"), b.get("generated"), output.get("has_token"))):
                            unknown("token application dimensions missing")
                        else:
                            guard(b["generated"] == a["generated"] + int(output["has_token"]), "token application not exactly once")
                        if row.get("phase") == "prefill":
                            guard(not output.get("has_token") and b.get("phase") == "decode", "prefill token/phase contract violation")
                        else:
                            guard(a.get("phase") == "decode" and a.get("status") not in TERMINAL, "Decode without active Prefill state")
        elif kind in {"forward_begin", "forward_end"}:
            rule = "FORWARD"
            if kind == "forward_begin":
                old = linked(worker_calls, row.get("phase_id"))
                members = row.get("members")
                shape, mask = row.get("input_shape"), row.get("mask_shape")
                cache = (row.get("cache_before") or {}).get("length")
                if (old and isinstance(members, list) and isinstance(shape, list) and len(shape) == 2 and
                    isinstance(mask, list) and len(mask) == 2 and cache is not None):
                    guard(len(members) == len(set(members)) == shape[0] and set(members) <= set(old.get("members", [])), "Forward member mapping mismatch")
                    guard(mask[0] == shape[0] and mask[1] == cache + shape[1], "mask/cache/input width equation violated")
                else:
                    unknown("Forward member/dimension evidence missing")
                forwards[begin_key("forward_id")] = {"row": row, "line": line}
            else:
                old = linked(forwards, row.get("forward_id"))
                if old and row.get("error") is None:
                    a = (old.get("cache_before") or {}).get("length")
                    b = (row.get("cache_after") or {}).get("length")
                    if a is None or b is None or not isinstance(old.get("input_shape"), list) or len(old["input_shape"]) != 2:
                        unknown("before/after KV length unavailable")
                    else:
                        guard(b == a + old["input_shape"][1], "successful forward KV did not grow by input width")
        elif kind in {"pool_allocate", "pool_update_request_cache", "pool_release"}:
            rule = "SLOT_UPDATE"
            inventory = row.get("slots")
            if rid is None or "before_slot" not in row or "after_slot" not in row or not isinstance(inventory, dict):
                unknown("pool before/after/owner evidence absent")
            else:
                values = list(inventory.values())
                guard(len(values) == len(set(values)), "pool owns one physical Slot for distinct requests")
                guard(inventory.get(rid) == row["after_slot"], "pool result contradicts owner inventory")
                if row.get("error") is None:
                    if kind == "pool_release":
                        guard(row["after_slot"] is None, "successful pool release left Slot occupied")
                    elif kind == "pool_allocate":
                        guard(row["before_slot"] is None and row["after_slot"] is not None, "allocation overwrote an existing Slot")
                    else:
                        guard(row["before_slot"] is not None and row["before_slot"] == row["after_slot"], "cache update changed physical Slot")
                fact(rid, "slot_present", rid in inventory)
                fact(rid, "slot_id", row["after_slot"])
        elif kind == "release_begin":
            rule = "RETIRE_BEGIN"
            identity(rid, row.get("request_object_identity"), "request_object_identity")
            confirmed[rid] = False
            calls[begin_key("release_id")] = {"row": row, "line": line}
        elif kind == "release_end":
            rule = "RETIRE_END"
            old = linked(calls, row.get("release_id"))
            if old:
                guard(old.get("request_id") == rid, "release call request identity mismatch")
            identity(rid, row.get("request_object_identity"), "request_object_identity")
            ref = row.get("request_cache_ref")
            fact(rid, "cache_ref", ref)
            if not isinstance(slots, dict) or not isinstance(records, dict) or ref is None:
                unknown("Slot/reference/retirement confirmation unobservable")
                confirmed[rid] = None
            elif row.get("error") is None:
                guard(rid not in slots and ref is False and rid not in records, "successful release did not retire all request resources")
                confirmed[rid] = rid not in slots and ref is False and rid not in records
            else:
                confirmed[rid] = False
                if ref is True:
                    if rid not in records and "session_request_ids" not in worker:
                        unknown("failed cleanup Session ownership unobservable")
                    else:
                        guard(rid in records or rid in worker.get("session_request_ids", []), "failed cleanup lost original identity")
            fact(rid, "release_confirmed", confirmed.get(rid))
        elif kind in {"cancel_end", "acknowledge_end", "close_end"}:
            rule = {"cancel_end": "CANCEL", "acknowledge_end": "ACK", "close_end": "CLOSE"}[kind]
            old = linked(calls, row.get("operation_id"))
            if old:
                previous = (old.get("snapshot") or {}).get("requests", {})
                if kind == "cancel_end" and rid in previous:
                    current = (requests or {}).get(rid)
                    if current:
                        guard(current.get("status") in TERMINAL, "cancel did not terminalize request")
                        if previous[rid].get("generated") is not None:
                            guard(current.get("generated") == previous[rid]["generated"], "cancel/retry generated tokens")
                if kind == "acknowledge_end" and row.get("error") is None and rid in previous:
                    guard(previous[rid].get("status") in TERMINAL, "acknowledged active request")
                    if confirmed.get(rid) is None:
                        unknown("ack resource confirmation missing")
                    else:
                        guard(confirmed[rid] is True, "ack before resource retirement")
        elif kind in {"lease_release_begin", "lease_release_end"}:
            rule = "LEASE"
            identity(rid, row.get("request_object_identity"), "request_object_identity")
            if row.get("released") is None:
                unknown("Lease release flag not observed")
            if kind == "lease_release_begin" and row.get("released") is not True:
                if confirmed.get(rid) is None:
                    unknown("lease release has no original resource confirmation")
                else:
                    guard(confirmed[rid] is True, "lease released before resource confirmation")
            if kind == "lease_release_end" and row.get("error") is None:
                if row.get("released") is None:
                    unknown("post-call Lease state absent")
                else:
                    guard(row["released"] is True, "Lease failed to release")
                    fact(rid, "lease", "released")
        elif kind in {"owner_publish", "http_observation"}:
            rule = "OWNER"
            if kind == "owner_publish":
                if row.get("closing") is None:
                    unknown("Owner running/draining evidence absent")
                else:
                    fact("@owner", "state", "draining" if row["closing"] else "running")
            elif row.get("label") == "owner-stopped":
                if row.get("owner_thread_alive") is None:
                    unknown("Owner stopped evidence absent")
                else:
                    guard(row["owner_thread_alive"] is False, "Owner reported stopped while thread alive")
                    fact("@owner", "state", "stopped")
        elif kind in {"worker_close_begin", "worker_close_end"}:
            rule = "CLOSE"
            if row.get("runtime_identity") is not None:
                if row.get("owns_worker") is None:
                    unknown("Worker ownership mode absent")
                else:
                    guard(row["owns_worker"] is True, "borrower closed shared backend")
            if kind == "worker_close_end" and row.get("error") is None:
                if slots is None or records is None:
                    unknown("backend final resource evidence absent")
                else:
                    guard(not slots and not records, "backend close claimed success with pending request resources")
        elif kind not in {"submit_end", "step_end", "retirement_evidence", "prefill_inputs", "decode_inputs",
                          "pool_allocate", "pool_update_request_cache", "pool_release", "collector_end"}:
            rule = "UNMODELED"
            unknown("event kind has no abstract rule", category="MODEL_RULE_MISSING")
        touched = {key for key in set(before) | set(facts) if before.get(key) != facts.get(key)}
        projections.append({"line": line, "seq": row.get("seq"), "event_id": event_id, "kind": kind,
            "rule_id": rule, "rule_sha256": digest(SCHEMA), "input_sha256": input_sha256,
            "event_sha256": fingerprint, "event_hash_basis": "canonical-json",
            "status": "FAIL" if any(i["status"] == "FAIL" for i in issues) else "UNKNOWN" if issues else "PASS",
            "issues": issues, "linked_evidence": refs,
            "before": {key: before.get(key) for key in sorted(touched)},
            "after": {key: deepcopy(facts.get(key)) for key in sorted(touched)},
            "projection_semantics": "last direct observations with field source lines; not inferred complete pre/post states"})
    counts = Counter(p["status"] for p in projections)
    return {"version": VERSION, "schema_sha256": digest(SCHEMA), "input_sha256": input_sha256,
        "mapping": projections, "counts": dict(counts), "rule_counts": dict(Counter(p["rule_id"] for p in projections)),
        "status": "FAIL" if counts["FAIL"] else "UNKNOWN" if counts["UNKNOWN"] or not rows else "PASS",
        "frozen_schedule_equivalence": "UNKNOWN", "whole_runtime_exhaustion": "UNKNOWN"}
