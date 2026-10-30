"""Offline retirement/lease/owner evidence checks and a bounded safety model.

Not a scheduler or production lifecycle store. Never issues runtime commands.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace

from .retirement import TERMINAL, check_retirement, replay

VERSION = 1


def check_composition(rows):
    base = check_retirement(rows)
    findings = []
    confirmed, identities, terminal = {}, {}, {}
    held = set()
    owner_threads = set()
    saw_lease = saw_owner = saw_final = saw_owned_close = False
    def report(code, status, row, reason):
        findings.append({"code": code, "status": status, "seq": row.get("seq"),
                         "event_id": row.get("event_id"), "reason": reason})
    for row in rows:
        kind, rid = row.get("kind"), row.get("request_id")
        snap = row.get("snapshot") or {}
        worker = snap.get("worker", snap)
        slots, records = worker.get("slots"), worker.get("retirements")
        for key, request in (snap.get("requests") or row.get("request_evidence") or {}).items():
            terminal[key] = request.get("status") in TERMINAL
            identity = request.get("request_object_identity")
            if identity is not None:
                if key in identities and identities[key] != identity:
                    report("COMPOSED_IDENTITY", "FAIL", row, "request identity changed")
                identities[key] = identity
        if kind == "release_begin":
            confirmed[rid] = False
        elif kind == "release_end":
            confirmed[rid] = (row.get("error") is None and isinstance(slots, dict) and rid not in slots and
                              row.get("request_cache_ref") is False and isinstance(records, dict) and rid not in records)
            if row.get("error") is not None and row.get("request_cache_ref") is True:
                sessions = worker.get("session_request_ids")
                if not isinstance(records, dict) or sessions is None:
                    report("RECOVERABLE_RETIREMENT", "UNKNOWN", row, "cleanup ownership unobservable")
                elif rid not in records and rid not in sessions:
                    report("RECOVERABLE_RETIREMENT", "FAIL", row, "lost request cleanup identity")
            for key, record in (records or {}).items():
                if record.get("request_object_identity") != identities.get(key):
                    report("COMPOSED_IDENTITY", "FAIL", row, "retirement identity differs from original request")
                if record.get("slot_released") is True and key in (slots or {}):
                    report("RECOVERABLE_RETIREMENT", "FAIL", row, "confirmed slot removal contradicts pool")
        elif kind == "lease_granted":
            saw_lease = True
            held.add(rid)
            identities[rid] = row.get("request_object_identity")
        elif kind == "lease_release_begin":
            saw_lease = True
            if not row.get("released"):
                if rid not in confirmed:
                    report("LEASE_AFTER_RETIREMENT", "UNKNOWN", row, "no retirement evidence")
                elif not confirmed[rid]:
                    report("LEASE_AFTER_RETIREMENT", "FAIL", row, "lease release before confirmed retirement")
                if terminal.get(rid) is not True:
                    report("LEASE_AFTER_RETIREMENT", "UNKNOWN", row, "terminal object not observed")
            if identities.get(rid) != row.get("request_object_identity"):
                report("COMPOSED_IDENTITY", "FAIL", row, "lease uses a different original object")
        elif kind == "lease_release_end" and row.get("error") is None:
            if row.get("released") is None or row.get("running_lease_ids") is None:
                report("LEASE_AFTER_RETIREMENT", "UNKNOWN", row, "post-release lease evidence missing")
            elif row.get("released") is not True or row.get("lease_id") in row.get("running_lease_ids", []):
                report("LEASE_AFTER_RETIREMENT", "FAIL", row, "lease still held after claimed release")
            held.discard(rid)
        elif kind == "owner_publish":
            saw_owner = True
            owner_threads.add(row.get("owner_thread"))
            deliveries, status = row.get("deliveries"), row.get("status") or {}
            if not isinstance(slots, dict) or not isinstance(records, dict):
                report("COMPOSED_CACHE_OWNERSHIP", "UNKNOWN", row, "authoritative cache snapshot missing")
            if deliveries is None or not isinstance(records, dict):
                report("NO_FALSE_IDLE", "UNKNOWN", row, "pending/retirement evidence missing")
            elif status.get("pending") != len(deliveries) or (records and status.get("pending") == 0):
                report("NO_FALSE_IDLE", "FAIL", row, "pending work hidden from owner status")
            for key, slot in (slots or {}).items():
                if slot.get("context_matches_cache") is not True and slot.get("retirement_matches_cache") is not True:
                    report("COMPOSED_CACHE_OWNERSHIP", "UNKNOWN" if "retirement_matches_cache" not in slot else "FAIL",
                           row, "slot lacks directly observed Session/retirement reference")
        elif kind == "worker_close_begin" and row.get("runtime_identity") is not None:
            if row.get("owns_worker") is False:
                report("WORKER_CLOSE_OWNERSHIP", "FAIL", row, "borrower closed shared worker")
            elif row.get("owns_worker") is True:
                saw_owned_close = True
                if slots or records:
                    report("WORKER_CLOSE_OWNERSHIP", "FAIL", row, "owned close before request resources retired")
            else:
                report("WORKER_CLOSE_OWNERSHIP", "UNKNOWN", row, "ownership unknown")
        elif kind == "close_end" and row.get("error") is None and snap.get("owns_worker") is False:
            saw_owned_close = True  # a borrower closing without backend close is valid evidence
        elif kind == "http_observation":
            status, scheduler = row.get("status") or {}, row.get("scheduler") or {}
            if records:
                if scheduler.get("running_requests") == 0 or row.get("pending_finalizers") == 0:
                    report("NO_FALSE_IDLE", "FAIL", row, "unconfirmed retirement but no lease/finalizer")
            if row.get("label") == "owner-stopped":
                saw_final = True
                fields = ("pending", "allocated_slots", "contexts", "claims", "released_ids", "finished_ids",
                          "runtime_active_requests", "pending_commands")
                refs = row.get("request_evidence") or {}
                if (any(status.get(field) is None for field in fields) or slots is None or records is None or
                    scheduler.get("running_requests") is None or scheduler.get("queued_requests") is None or
                    row.get("pending_finalizers") is None or row.get("owner_thread_alive") is None or
                    set(identities) - set(refs) or any(r.get("has_cache_ref") is None for r in refs.values())):
                    report("COMPOSED_FINAL_DRAIN", "UNKNOWN", row, "final resource dimensions absent")
                elif (any(status[field] != 0 for field in fields) or slots or records or held or
                      scheduler.get("running_requests") != 0 or scheduler.get("queued_requests") != 0 or
                      row.get("pending_finalizers") != 0 or row.get("owner_thread_alive") is not False or
                      any(r.get("has_cache_ref") is not False for r in refs.values())):
                    report("COMPOSED_FINAL_DRAIN", "FAIL", row, "observed final drain incomplete")
    if saw_owner:
        if None in owner_threads or len(owner_threads) != 1:
            report("OWNER_SERIALIZATION", "UNKNOWN", {}, "single owner thread not identified")
        else:
            for row in rows:
                if row.get("kind") in {"worker_begin", "release_begin"} and row.get("thread_id") not in owner_threads:
                    report("OWNER_SERIALIZATION", "FAIL", row, "worker operation outside serial owner")
    for condition, code in ((saw_owner, "OWNER_SERIALIZATION"), (saw_lease, "LEASE_AFTER_RETIREMENT"),
                            (saw_final, "COMPOSED_FINAL_DRAIN"), (saw_owned_close, "WORKER_CLOSE_OWNERSHIP")):
        if not condition:
            report(code, "UNKNOWN", {}, "required composition observation absent")
    codes = ("COMPOSED_IDENTITY", "RECOVERABLE_RETIREMENT", "LEASE_AFTER_RETIREMENT", "NO_FALSE_IDLE",
             "COMPOSED_CACHE_OWNERSHIP", "OWNER_SERIALIZATION", "WORKER_CLOSE_OWNERSHIP", "COMPOSED_FINAL_DRAIN")
    checks = []
    for code in codes:
        evidence = [f for f in findings if f["code"] == code]
        status = "FAIL" if any(f["status"] == "FAIL" for f in evidence) else "UNKNOWN" if any(f["status"] == "UNKNOWN" for f in evidence) else "PASS"
        checks.append({"code": code, "status": status, "evidence": evidence})
    statuses = [base["status"], *[c["status"] for c in checks]]
    return {"version": VERSION, "status": "FAIL" if "FAIL" in statuses else "UNKNOWN" if "UNKNOWN" in statuses else "PASS",
            "checks": checks, "retirement": base, "admitted": False,
            "frozen_schedule_equivalence": "UNKNOWN", "whole_runtime_exhaustion": "UNKNOWN"}


@dataclass(frozen=True)
class Request:
    life: str = "new"
    resource: str = "unallocated"
    lease: str = "none"
    slot: bool = False
    reference: bool = False
    attempts: int = 0  # 2 denotes two or more, not a retry limit
    context: bool = False
    completion: bool = False
    tokens_remaining: int = 2  # exact bounded decode length, not an unbounded self-loop


@dataclass(frozen=True)
class State:
    requests: tuple[Request, Request] = (Request(), Request())
    owner: str = "open"
    ownership: str = "borrowed"
    busy: int = -1
    worker_closed: bool = False


def successors(state):
    """Two ordered arrivals, one lease/slot, one serial cleanup operation."""
    if state.owner == "stopped":
        return []
    edges = []
    def change(index, action, **fields):
        requests = list(state.requests)
        requests[index] = replace(requests[index], **fields)
        edges.append((action, replace(state, requests=tuple(requests))))
    if state.owner == "open":
        for i, r in enumerate(state.requests):
            if r.life == "new" and (i == 0 or state.requests[i - 1].life != "new"):
                held = any(q.lease == "held" for q in state.requests)
                change(i, f"arrive:{i}", life="waiting", lease="queued" if held else "held")
        requests = tuple(replace(r, life="cancelled", lease="released") if r.lease == "queued" else r for r in state.requests)
        edges.append(("shutdown_intent", replace(state, owner="closing", requests=requests)))
    if state.busy != -1:
        i, r = state.busy, state.requests[state.busy]
        for action, fields in (
            ("success", dict(resource="retired", slot=False, reference=False)),
            ("failure_retained", dict(resource="retry")),
            ("failure_slot_removed", dict(resource="retry", slot=False)),
            ("failure_ref_cleared", dict(resource="retry", slot=False, reference=False)),
            ("unobservable", dict(resource="unknown")),
        ):
            requests = list(state.requests)
            requests[i] = replace(r, **fields)
            edges.append((action, replace(state, busy=-1, requests=tuple(requests))))
        return edges
    for i, r in enumerate(state.requests):
        if r.lease != "held":
            continue
        if not r.context and not r.completion:
            change(i, f"handoff:{i}", context=True)
        if r.context and r.life in {"waiting", "running"}:
            for terminal in ("cancelled", "failed"):
                change(i, f"{terminal}:{i}", life=terminal)
            if state.owner == "open" and r.life == "waiting" and not any(q.slot for q in state.requests):
                change(i, f"prefill:{i}", life="running", resource="live", slot=True, reference=True)
            if r.life == "running":
                if r.tokens_remaining > 1:
                    change(i, f"decode:{i}", tokens_remaining=r.tokens_remaining - 1)
                change(i, f"finish:{i}", life="finished", tokens_remaining=0)
        if r.context and r.life in TERMINAL and r.resource != "retired":
            requests = list(state.requests)
            requests[i] = replace(r, resource="pending", attempts=min(2, r.attempts + 1))
            edges.append((f"cleanup_begin:{i}", replace(state, busy=i, requests=tuple(requests))))
        if r.context and r.life in TERMINAL and r.resource == "retired":
            change(i, f"ack_complete:{i}", context=False, completion=True)
        if r.completion:
            requests = list(state.requests)
            requests[i] = replace(r, lease="released")
            for j, q in enumerate(requests):
                if q.lease == "queued":
                    requests[j] = replace(q, lease="held")
                    break
            edges.append((f"finalize_lease:{i}", replace(state, requests=tuple(requests))))
    if state.owner == "closing" and all(r.lease in {"none", "released"} and not r.context and not r.slot and not r.reference for r in state.requests):
        edges.append(("stop", replace(state, owner="stopped", worker_closed=state.ownership == "owned")))
    return edges


def explore():
    initial = (State(), State(ownership="owned"))
    seen, pending, edges = set(initial), deque(initial), []
    while pending:
        before = pending.popleft()
        for action, after in successors(before):
            edges.append((before, action, after))
            if after not in seen:
                seen.add(after); pending.append(after)
    ordered = sorted(seen, key=repr)
    index = {s: i for i, s in enumerate(ordered)}
    violations = []
    for i, state in enumerate(ordered):
        if (sum(r.slot for r in state.requests) > 1 or
            sum(r.lease == "held" for r in state.requests) > 1 or
            (state.worker_closed and (state.ownership != "owned" or state.owner != "stopped")) or
            any(r.lease == "released" and (r.slot or r.reference or r.context) for r in state.requests)):
            violations.append(i)
    retry = [(index[a], index[b]) for a, action, b in edges if action == "failure_retained" and
             any(q == a for _, q in successors(b))]
    return {"version": VERSION, "scope": "two ordered requests, at most two decode tokens each; one lease/slot/queue entry; serial cleanup; two ownership modes",
        "state_total": len(ordered), "transition_total": len(edges),
        "terminal_total": sum(s.owner == "stopped" for s in ordered), "unexplored_total": 0, "exhausted": True,
        "safety": "FAIL" if violations else "PASS", "violations": violations,
        "retry_cycle_witnesses": retry,
        "liveness": "requires eventual cleanup success, fair retry and client finalization",
        "termination": "NOT_GUARANTEED", "whole_runtime_equivalence": "UNKNOWN",
        "states": [asdict(s) for s in ordered],
        "transitions": [{"from": index[a], "action": action, "to": index[b]} for a, action, b in edges]}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from .artifacts import git_snapshot, write_new
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    integrity = replay(args.directory)
    rows = [] if "TRACE_ARTIFACT_INTEGRITY" in integrity.get("reason_codes", []) else [
        json.loads(line) for line in (args.directory / "runtime-trace.jsonl").read_text(encoding="utf-8").splitlines()]
    result = check_composition(rows)
    if integrity.get("status") == "FAIL":
        result["status"] = "FAIL"
    write_new(args.output, {"current_source": git_snapshot(), "raw_evidence": integrity,
                            "composition": result, "submodel": explore()})
