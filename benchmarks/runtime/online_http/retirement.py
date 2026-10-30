"""Offline joint lifecycle/retirement model, not an online state owner.

Slot absence, request-reference absence and a successful cleanup result are
separate evidence. Terminal generation does not imply any of these conditions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from enum import Enum

VERSION = 2
TERMINAL = {"finished", "cancelled", "failed"}


class ResourceStatus(str, Enum):
    UNALLOCATED = "unallocated"
    LIVE = "live"
    RELEASE_REQUESTED = "release_requested"
    RETRY_REQUIRED = "retry_required"
    RETIRED = "retired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class JointState:
    lifecycle: str = "waiting"
    resource: ResourceStatus = ResourceStatus.UNALLOCATED
    slot: int | None = None
    slot_known: bool = True
    cache_ref: bool | None = False
    confirmed: bool = False

    @property
    def drained(self):
        return (self.lifecycle in TERMINAL and self.slot_known and self.slot is None and
                self.cache_ref is False and self.resource in {ResourceStatus.UNALLOCATED, ResourceStatus.RETIRED})


def successors(state):
    """One request/one slot safety submodel. Invalid success is not a legal edge.

    Retry count is quotiented away: repeated failure has the same next state.
    This preserves a nonterminating retry cycle rather than assuming recovery.
    """
    r = ResourceStatus
    edges = []
    if state.lifecycle == "waiting":
        edges.append(("start", replace(state, lifecycle="running")))
    if state.lifecycle in {"waiting", "running"}:
        for target in ("cancelled", "failed") + (("finished",) if state.lifecycle == "running" else ()):
            edges.append((target, replace(state, lifecycle=target)))
    if state.lifecycle == "running" and state.resource is r.UNALLOCATED:
        edges.append(("allocate", replace(state, resource=r.LIVE, slot=0, cache_ref=True)))
    if state.lifecycle == "running" and state.resource is r.LIVE:
        edges.append(("decode_no_ownership_change", state))
    if state.resource is not r.RELEASE_REQUESTED:
        edges.append(("request_release", replace(state, resource=r.RELEASE_REQUESTED, confirmed=False)))
    else:
        edges += [
            ("success_absent_clear", replace(state, resource=r.RETIRED, slot=None, slot_known=True, cache_ref=False, confirmed=True)),
            ("error_slot_absent_ref_retained", replace(state, resource=r.RETRY_REQUIRED, slot=None, slot_known=True, cache_ref=True, confirmed=False)),
            ("error_absent_clear_but_unconfirmed", replace(state, resource=r.RETRY_REQUIRED, slot=None, slot_known=True, cache_ref=False, confirmed=False)),
            ("success_absent_ref_retained", replace(state, slot=None, slot_known=True, cache_ref=True, confirmed=True)),
            ("unobservable", replace(state, resource=r.UNKNOWN, slot_known=False, cache_ref=None, confirmed=False)),
        ]
        if state.slot == 0:
            edges.append(("error_slot_retained", replace(state, resource=r.RETRY_REQUIRED, confirmed=False)))
        if state.confirmed and state.slot_known and state.slot is None:
            edges.append(("observed_ref_clear", replace(state, resource=r.RETIRED, cache_ref=False)))
    return edges


def explore():
    """Exhaust this explicitly restricted finite graph, never the whole Runtime."""
    seen, pending = {JointState()}, deque([JointState()])
    edges = []
    while pending:
        before = pending.popleft()
        for action, after in successors(before):
            edges.append((before, action, after))
            if after not in seen:
                seen.add(after)
                pending.append(after)
    def key(s):
        return (s.lifecycle, s.resource.value, -1 if s.slot is None else s.slot, s.slot_known,
                -1 if s.cache_ref is None else int(s.cache_ref), s.confirmed)
    ordered = sorted(seen, key=key)
    ids = {s: i for i, s in enumerate(ordered)}
    retry_cycles = [i for i, s in enumerate(ordered) if s.resource is ResourceStatus.RETRY_REQUIRED and
        any(action == "request_release" and any(t == s for _, t in successors(q)) for action, q in successors(s))]
    invalid_retired = [i for i, s in enumerate(ordered) if s.resource is ResourceStatus.RETIRED and
                       (s.slot is not None or not s.slot_known or s.cache_ref is not False or not s.confirmed)]
    return {"model_version": VERSION, "scope": "one request, one slot, no tensor/deadline/admission/worker-ownership model",
        "state_total": len(ordered), "transition_total": len(edges),
        "drained_states": sum(s.drained for s in ordered), "unexplored_states": 0, "exhausted": True,
        "safety": "PASS" if not invalid_retired else "FAIL", "invalid_retired_states": invalid_retired,
        "retry_cycle_witnesses": retry_cycles, "unconditional_liveness": "NOT_PROVABLE",
        "unconditional_termination": "FALSE_FOR_SUBMODEL_WITH_PERMANENT_FAILURE",
        "states": [asdict(s) for s in ordered],
        "transitions": [{"from": ids[a], "action": action, "to": ids[b]} for a, action, b in edges],
        "whole_runtime_exhaustion": "UNKNOWN"}


def check_retirement(rows):
    """Recompute resource evidence from raw v1 trace events; missing != clear."""
    states, outcomes, pending, allocations = {}, [], set(), {}
    findings = []
    identities = {}
    def report(code, status, row, reason):
        findings.append({"code": code, "status": status, "seq": row.get("seq"),
                         "event_id": row.get("event_id"), "reason": reason})
    def ensure(rid):
        return states.setdefault(rid, JointState())
    def observe_identity(rid, identity, row):
        if identity is None:
            report("RETIREMENT_REQUEST_IDENTITY", "UNKNOWN", row, "original request object not observed: " + str(rid))
        elif rid in identities and identities[rid] != identity:
            report("RETIREMENT_REQUEST_IDENTITY", "FAIL", row, "original RequestState identity changed: " + rid)
        else:
            identities[rid] = identity
    if not rows:
        report("TRACE_PRESENT", "UNKNOWN", {}, "no raw events")
    def snapshot(row):
        snap = row.get("snapshot")
        if not isinstance(snap, dict):
            return
        worker = snap.get("worker", snap)
        slots = worker.get("slots")
        requests = snap.get("requests")
        if isinstance(slots, dict):
            physical = [v.get("slot") for v in slots.values()]
            if any(value is None for value in physical):
                report("RELEASE_EVIDENCE", "UNKNOWN", row, "Slot identity unobservable")
            physical = [value for value in physical if value is not None]
            if len(physical) != len(set(physical)):
                report("RESOURCE_SLOT_UNIQUENESS", "FAIL", row, "duplicate live slot owner")
        if not isinstance(requests, dict):
            return
        for rid, request in requests.items():
            old = ensure(rid)
            status = request.get("status")
            if status is None:
                report("REQUEST_LIFECYCLE", "UNKNOWN", row, "lifecycle not observed: " + rid)
                status = old.lifecycle
            if (status not in {"waiting", "running", *TERMINAL} or
                    (old.lifecycle in TERMINAL and status != old.lifecycle) or
                    (old.lifecycle == "running" and status == "waiting")):
                report("REQUEST_LIFECYCLE", "FAIL", row, "invalid lifecycle transition: " + rid)
            ref = request.get("has_cache_ref")
            slot_known = isinstance(slots, dict)
            slot = slots.get(rid, {}).get("slot") if slot_known else old.slot
            state = replace(old, lifecycle=status, slot=slot, slot_known=slot_known, cache_ref=ref)
            if slot_known and slot is not None and old.resource is ResourceStatus.UNALLOCATED:
                state = replace(state, resource=ResourceStatus.LIVE)
            if old.confirmed and slot_known and slot is None:
                if ref is False:
                    state = replace(state, resource=ResourceStatus.RETIRED)
                elif ref is True:
                    state = replace(state, resource=ResourceStatus.RELEASE_REQUESTED)
                    report("RETIRED_REFERENCE_CLEARED", "FAIL", row, "release returned successfully but RequestState still holds cache: " + rid)
                else:
                    state = replace(state, resource=ResourceStatus.UNKNOWN)
                    report("RETIRED_REFERENCE_CLEARED", "UNKNOWN", row, "request cache reference unobservable")
            states[rid] = state
    for row in rows:
        kind, rid = row.get("kind"), row.get("request_id")
        snapshot(row)
        if kind == "release_begin":
            observe_identity(rid, row.get("request_object_identity"), row)
            old = ensure(rid)
            pending.add(rid)
            states[rid] = replace(old, resource=ResourceStatus.RELEASE_REQUESTED, confirmed=False)
        elif kind == "release_end":
            observe_identity(rid, row.get("request_object_identity"), row)
            old = ensure(rid)
            slots = (row.get("snapshot") or {}).get("slots")
            known = isinstance(slots, dict)
            slot = slots.get(rid, {}).get("slot") if known else old.slot
            # New optional fields are actual object reads, not inferred from
            # Slot disappearance. Old traces remain independently replayable.
            ref = row.get("request_cache_ref", None)
            if rid not in pending:
                report("RELEASE_PAIRING", "UNKNOWN", row, "release begin absent")
            if row.get("error") is not None:
                state = replace(old, resource=ResourceStatus.RETRY_REQUIRED, slot=slot,
                                slot_known=known, cache_ref=ref, confirmed=False)
                report("CLEANUP_FAILURE_REPRESENTABLE", "PASS", row, "failed cleanup remains unconfirmed; Slot may be present or absent")
            elif not known:
                state = replace(old, resource=ResourceStatus.UNKNOWN, slot_known=False, cache_ref=ref, confirmed=False)
                report("RELEASE_EVIDENCE", "UNKNOWN", row, "Slot absence unobservable")
            elif slot is not None:
                state = replace(old, resource=ResourceStatus.RETRY_REQUIRED, slot=slot, slot_known=True, cache_ref=ref, confirmed=False)
                report("RELEASE_SUCCESS_WITH_SLOT", "FAIL", row, "successful return contradicts occupied Slot")
            else:
                state = replace(old, resource=ResourceStatus.RETIRED if ref is False else ResourceStatus.RELEASE_REQUESTED,
                                slot=None, slot_known=True, cache_ref=ref, confirmed=True)
                if ref is True:
                    report("RETIRED_REFERENCE_CLEARED", "FAIL", row, "successful return left actual RequestState cache reference")
                elif ref is None:
                    report("RETIRED_REFERENCE_CLEARED", "UNKNOWN", row, "original post-release reference missing")
                elif row.get("retirement_version") == 2 and rid in (row.get("snapshot") or {}).get("retirements", {}):
                    report("RELEASE_EVIDENCE", "FAIL", row, "successful return with pending retirement record")
            states[rid] = state
            pending.discard(rid)
            outcomes.append({"seq": row.get("seq"), "request_id": rid, **asdict(state)})
        elif kind == "retirement_evidence":
            for request_id, evidence in row.get("requests", {}).items():
                observe_identity(request_id, evidence.get("request_object_identity"), row)
                old = ensure(request_id)
                ref = evidence.get("has_cache_ref")
                state = replace(old, lifecycle=evidence["status"], cache_ref=ref)
                if old.confirmed and old.slot_known and old.slot is None and ref is False:
                    state = replace(state, resource=ResourceStatus.RETIRED)
                elif old.confirmed and ref is True:
                    state = replace(state, resource=ResourceStatus.RELEASE_REQUESTED)
                    report("RETIRED_REFERENCE_CLEARED", "FAIL", row, "post-action actual object retains cache: " + request_id)
                elif ref is None:
                    report("RETIRED_REFERENCE_CLEARED", "UNKNOWN", row, "post-action reference not observable")
                states[request_id] = state
        elif kind == "pool_allocate":
            slot = row.get("after_slot")
            if row.get("error") is None and slot is not None:
                if any(owner != rid and value == slot for owner, value in allocations.items()):
                    report("RESOURCE_SLOT_UNIQUENESS", "FAIL", row, "occupied Slot reused")
                allocations[rid] = slot
        elif kind == "pool_release" and row.get("error") is None and row.get("after_slot") is None:
            allocations.pop(rid, None)
        elif kind == "executor_submit" and row.get("phase") == "decode":
            for request_id in row.get("members", []):
                if ensure(request_id).lifecycle in TERMINAL:
                    report("NO_REGENERATION_ON_RETRY", "FAIL", row, "terminal request entered Decode")
        elif kind in {"close_end", "acknowledge_end"} and row.get("error") is None:
            # Empty Context registry does not prove external RequestState refs
            # were cleared. Require explicit before-object retirement evidence.
            pass
    if pending:
        report("RELEASE_PAIRING", "UNKNOWN", rows[-1] if rows else {}, "unpaired release requests")
    unresolved = [rid for rid, state in states.items() if not state.drained]
    if unresolved:
        report("RESOURCE_FINAL_DRAIN", "UNKNOWN", rows[-1] if rows else {}, "unconfirmed or missing retirement evidence: " + ",".join(unresolved))
    else:
        report("RESOURCE_FINAL_DRAIN", "PASS", rows[-1] if rows else {}, "all observed requests terminal with confirmed absence of owned resources")
    codes = ("TRACE_PRESENT", "RETIREMENT_REQUEST_IDENTITY", "REQUEST_LIFECYCLE", "RESOURCE_SLOT_UNIQUENESS", "RELEASE_PAIRING", "CLEANUP_FAILURE_REPRESENTABLE",
             "RELEASE_SUCCESS_WITH_SLOT", "RETIRED_REFERENCE_CLEARED", "RELEASE_EVIDENCE", "NO_REGENERATION_ON_RETRY", "RESOURCE_FINAL_DRAIN")
    checks = []
    for code in codes:
        entries = [f for f in findings if f["code"] == code]
        status = "FAIL" if any(f["status"] == "FAIL" for f in entries) else "UNKNOWN" if any(f["status"] == "UNKNOWN" for f in entries) else "PASS"
        checks.append({"code": code, "status": status, "evidence": entries})
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else "UNKNOWN" if unresolved or any(c["status"] == "UNKNOWN" for c in checks) else "PASS"
    return {"model_version": VERSION, "status": status, "checks": checks, "release_outcomes": outcomes,
        "final_states": {rid: asdict(s) for rid, s in states.items()}, "admitted": False,
        "whole_runtime_equivalence": "UNKNOWN", "liveness": "UNKNOWN", "termination": "NOT_PROVABLE_WITH_PERMANENT_FAILURE"}


def replay(directory):
    """Verify sealed raw artifacts before independent retirement recomputation."""
    import hashlib
    import json
    from pathlib import Path
    from .trace_check import check_directory
    directory = Path(directory)
    integrity = check_directory(directory)
    if "TRACE_ARTIFACT_INTEGRITY" in integrity.get("reason_codes", []):
        return integrity
    raw = (directory / "runtime-trace.jsonl").read_bytes()
    result = check_retirement([json.loads(line) for line in raw.splitlines()])
    if integrity.get("status") == "FAIL":
        result["status"] = "FAIL"
    return {**result, "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "source_trace_checks": integrity["checks"],
            "source_directory": directory.name}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from .artifacts import git_snapshot, write_new
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_new(args.output, {"current_source": git_snapshot(), "replay": replay(args.directory), "submodel": explore()})
