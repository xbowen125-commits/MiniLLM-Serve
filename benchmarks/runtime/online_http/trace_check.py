"""Independent raw runtime trace checks; observed conformance is not equivalence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .artifacts import write_new

TERMINAL = {"finished", "failed", "cancelled"}
IDENTITY = ("schema_version", "run_id", "pid", "config_sha256", "model_revision", "dtype", "device", "git", "geometry")


def check_rows(rows, ending):
    problems, unknown = {}, {}
    codes = ("TRACE_IDENTITY", "PLAN_EXECUTOR_MEMBERS", "EXECUTOR_WORKER_MEMBERS",
        "FORWARD_MEMBERS", "SLOT_UNIQUENESS", "CACHE_OWNERSHIP", "PHASE_LIFECYCLE",
        "PREFILL_ONCE", "FORWARD_KV_EVOLUTION", "REQUEST_TOKEN_EVOLUTION",
        "TERMINAL_NOT_DECODED", "RELEASE_TRUTHFUL", "FINAL_DRAIN", "COLLECTOR_COMPLETE",
        "OBSERVED_ABSTRACT_STATE_DOMAIN", "SLOT_EVENT_CONSISTENCY")
    def fail(code, row, reason):
        problems.setdefault(code, []).append({"seq": row.get("seq"), "event_id": row.get("event_id"), "reason": reason})
    def missing(code, row, reason):
        unknown.setdefault(code, []).append({"seq": row.get("seq"), "reason": reason})
    if not rows:
        return {"status": "UNKNOWN", "checks": [], "reason_codes": ["TRACE_EMPTY"]}
    identity = {k: rows[0].get(k) for k in IDENTITY}
    stages, phases, iterations, forwards, known, prefilled = {}, {}, {}, {}, {}, set()
    unconfirmed = set()
    totals = {"forwards": 0, "associated_forwards": 0, "slot_kv_rows": 0, "associated_slot_kv_rows": 0,
              "iterations": 0, "associated_iterations": 0}
    last_time = None
    last_close = None
    registered, requested_ids, linked_ids = set(), set(), set()
    pool_slots = {}
    for ordinal, row in enumerate(rows, 1):
        kind, it = row.get("kind"), row.get("iteration_id")
        if (any(row.get(k) != v or v is None for k, v in identity.items()) or
                row.get("seq") != ordinal or row.get("collector") != "runtime-boundaries-v1" or
                not isinstance(row.get("monotonic_s"), (int, float))):
            fail("TRACE_IDENTITY", row, "missing identity/sequence/time field")
        elif last_time is not None and row["monotonic_s"] < last_time:
            fail("TRACE_IDENTITY", row, "monotonic clock regressed")
        last_time = row.get("monotonic_s")
        snapshot = row.get("snapshot")
        if isinstance(snapshot, dict):
            worker = snapshot.get("worker", snapshot)
            slots = worker.get("slots")
            if slots is None:
                missing("CACHE_OWNERSHIP", row, "slot/cache snapshot absent")
            else:
                ids = [x.get("slot") for x in slots.values()]
                objects = [x.get("cache_object_identity") for x in slots.values()]
                if len(set(ids)) != len(ids):
                    fail("SLOT_UNIQUENESS", row, "same slot belongs to distinct requests")
                if len(set(objects)) != len(objects):
                    fail("CACHE_OWNERSHIP", row, "aliased live cache objects within one snapshot")
                for rid, value in slots.items():
                    totals["slot_kv_rows"] += 1
                    if (value.get("context_matches_cache") is True and value.get("context_request_id") == rid
                            and isinstance(value.get("cache", {}).get("length"), int)):
                        totals["associated_slot_kv_rows"] += 1
                    else:
                        if rid in unconfirmed or (kind == "release_end" and row.get("error") is not None):
                            missing("CACHE_OWNERSHIP", row, "failed cleanup retains slot without live session: " + rid)
                        else:
                            fail("CACHE_OWNERSHIP", row, "actual slot/cache/context reference mismatch: " + rid)
            requests = snapshot.get("requests")
            if isinstance(requests, dict):
                for rid, state in requests.items():
                    previous = known.get(rid)
                    count = state.get("generated")
                    if not isinstance(count, int) or not 0 <= count <= state.get("max_tokens", -1):
                        fail("REQUEST_TOKEN_EVOLUTION", row, "invalid generated count")
                    if previous:
                        if count < previous["generated"]:
                            fail("REQUEST_TOKEN_EVOLUTION", row, "generated count regressed")
                        if previous["status"] in TERMINAL and state["status"] != previous["status"]:
                            fail("PHASE_LIFECYCLE", row, "terminal lifecycle changed")
                        allowed = {"waiting": {"waiting", "prefill", "decode", "finished"},
                                   "prefill": {"prefill", "decode", "finished"},
                                   "decode": {"decode", "finished"}, "finished": {"finished"}}
                        if state.get("phase") not in allowed.get(previous.get("phase"), set()):
                            fail("PHASE_LIFECYCLE", row, "invalid execution phase transition")
                    known[rid] = state
                    if kind in {"step_begin", "step_end", "submit_end", "cancel_end"}:
                        slot = (slots or {}).get(rid)
                        if state["status"] in TERMINAL:
                            if slot is not None:
                                missing("OBSERVED_ABSTRACT_STATE_DOMAIN", row, "T state still owns cleanup resources: " + rid)
                        elif state["phase"] == "waiting":
                            if slot is not None or count != 0:
                                fail("OBSERVED_ABSTRACT_STATE_DOMAIN", row, "WAITING state already owns KV/tokens")
                        elif state["phase"] == "decode" and slot is not None:
                            if (slot["pending_first"] is True and count != 0) or (slot["pending_first"] is False and not 1 <= count < state["max_tokens"]):
                                fail("OBSERVED_ABSTRACT_STATE_DOMAIN", row, "pending-first/ready count violates v2 F/D domain")
                        else:
                            missing("OBSERVED_ABSTRACT_STATE_DOMAIN", row, "stable state lacks representable phase/cache: " + rid)
        if kind == "step_begin":
            totals["iterations"] += 1
            iterations[it] = {"begin": row, "phases": []}
            if it is None:
                missing("PLAN_EXECUTOR_MEMBERS", row, "iteration ID absent")
        elif kind == "submit_end" and row.get("error") is None:
            registered.add(row.get("request_id"))
        elif kind == "scheduler_plan":
            if it not in iterations:
                fail("PLAN_EXECUTOR_MEMBERS", row, "no iteration start")
                continue
            iterations[it]["scheduler"] = row
        elif kind == "executor_plan":
            plan = iterations.get(it, {}).get("scheduler")
            if plan is None or row.get("decode") != plan.get("decode"):
                fail("PLAN_EXECUTOR_MEMBERS", row, "decode membership differs from scheduler")
            elif row.get("prefill") != plan.get("prefill", [])[:max(0, row.get("free_slots", -1))]:
                fail("PLAN_EXECUTOR_MEMBERS", row, "prefill trim not explained by observed free slots")
            else:
                totals["associated_iterations"] += 1
            iterations.setdefault(it, {})["executor"] = row
        elif kind == "executor_submit":
            phases[row.get("phase_id")] = {"submit": row, "forwards": []}
            plan = iterations.get(it, {}).get("executor")
            if plan is None or row.get("members") != plan.get(row.get("phase")):
                fail("PLAN_EXECUTOR_MEMBERS", row, "executor phase members differ from actual plan")
            if row.get("phase") == "decode":
                for rid in row.get("members", []):
                    if rid not in known:
                        missing("PHASE_LIFECYCLE", row, "no registered request: " + rid)
                    elif known[rid]["status"] in TERMINAL:
                        fail("TERMINAL_NOT_DECODED", row, rid)
        elif kind == "worker_begin":
            requested_ids.update(row.get("members", []))
            linked_ids.update(rid for rid in row.get("members", []) if rid in registered)
            if not set(row.get("members", [])) <= registered:
                fail("EXECUTOR_WORKER_MEMBERS", row, "unregistered request identity")
            phase = phases.get(row.get("phase_id"))
            if phase is None or row.get("members") != phase["submit"].get("members"):
                fail("EXECUTOR_WORKER_MEMBERS", row, "worker membership differs from executor")
            if phase is not None:
                phase["worker"] = row
        elif kind == "forward_begin":
            totals["forwards"] += 1
            forwards[row.get("forward_id")] = row
            phase = phases.get(row.get("phase_id"))
            members = row.get("members")
            if phase is None or "worker" not in phase or not isinstance(members, list) or None in members:
                missing("FORWARD_MEMBERS", row, "actual worker input association missing")
                continue
            phase["forwards"].append(row)
            batch, width = row["input_shape"]
            if len(members) != batch or len(set(members)) != batch or not set(members) <= set(phase["worker"]["members"]):
                fail("FORWARD_MEMBERS", row, "forward group inconsistent with actual worker objects")
            else:
                totals["associated_forwards"] += 1
            cache = row.get("cache_before") or {}
            before = cache.get("length")
            if before is None or row.get("mask_shape") is None:
                missing("FORWARD_KV_EVOLUTION", row, "input mask/cache dimensions absent")
                continue
            geometry = row.get("geometry") or {}
            if any(type(geometry.get(k)) is not int for k in ("layers", "kv_heads", "head_dim")):
                missing("FORWARD_KV_EVOLUTION", row, "pinned layer geometry absent")
            elif row["phase"] == "decode":
                expected_layers = [{"keys": [batch, geometry["kv_heads"], before, geometry["head_dim"]],
                    "values": [batch, geometry["kv_heads"], before, geometry["head_dim"]]}
                    for _ in range(geometry["layers"])]
                if cache.get("layers") != expected_layers:
                    fail("FORWARD_KV_EVOLUTION", row, "merged KV layer dimensions mismatch")
            if row["phase"] == "prefill":
                if before != 0 or row["mask_shape"] != [batch, width]:
                    fail("FORWARD_KV_EVOLUTION", row, "invalid prefill mask/cache")
                for rid in members:
                    if rid in prefilled:
                        fail("PREFILL_ONCE", row, rid)
                    prefilled.add(rid)
            else:
                slots = phase["worker"]["snapshot"].get("slots") or {}
                if (any(rid not in slots or slots[rid].get("pending_first") is not False for rid in members) or
                        before != max((slots[rid]["cache"]["length"] for rid in members if rid in slots), default=-1) or
                        width != 1 or row["mask_shape"] != [batch, before + 1]):
                    fail("FORWARD_KV_EVOLUTION", row, "decode violates ready/cache merge rule")
        elif kind == "forward_end":
            begin = forwards.get(row.get("forward_id"))
            if begin is None:
                fail("FORWARD_MEMBERS", row, "unknown forward completion")
            elif row.get("error") is None:
                if (row.get("cache_after") or {}).get("length") != begin["cache_before"]["length"] + begin["input_shape"][1]:
                    fail("FORWARD_KV_EVOLUTION", row, "successful forward cache length mismatch")
        elif kind == "worker_end":
            phase = phases.get(row.get("phase_id"))
            if phase is None or "worker" not in phase:
                fail("EXECUTOR_WORKER_MEMBERS", row, "unpaired worker result")
                continue
            phase["end"] = row
            requested = phase["worker"]["members"]
            results = row.get("results")
            if results is not None:
                if any("request_id" not in r for r in results):
                    phase["invalid_results"] = True
                    continue
                actual = [r["request_id"] for r in results]
                phase["invalid_results"] = len(set(actual)) != len(actual) or set(actual) != set(requested)
                bad = {r["request_id"] for r in results if r.get("error")}
                forwarded = {rid for f in phase["forwards"] for rid in f["members"]}
                slots = phase["worker"]["snapshot"].get("slots") or {}
                expected = set(requested) if row["phase"] == "prefill" else {rid for rid in requested if
                    rid in slots and slots[rid].get("pending_first") is False}
                if not expected - forwarded <= bad and row.get("error") is None:
                    fail("FORWARD_MEMBERS", row, "omitted ready/prefill row has no request error")
        elif kind == "executor_phase_end":
            phase = phases.get(row.get("phase_id")) or {}
            if phase.get("invalid_results") and row.get("error") is None:
                fail("EXECUTOR_WORKER_MEMBERS", row, "protocol error not rejected by executor")
            if row.get("error") is None and not phase.get("invalid_results") and phase.get("end", {}).get("results") is not None:
                before_requests = (phase.get("submit", {}).get("snapshot") or {}).get("requests") or {}
                after_requests = (row.get("snapshot") or {}).get("requests") or {}
                old_slots = phase["worker"]["snapshot"].get("slots") or {}
                new_slots = phase["end"]["snapshot"].get("slots") or {}
                for result in phase["end"]["results"]:
                    rid = result["request_id"]
                    if rid not in before_requests or rid not in after_requests:
                        missing("REQUEST_TOKEN_EVOLUTION", row, "request lifecycle snapshot absent")
                        continue
                    delta = after_requests[rid]["generated"] - before_requests[rid]["generated"]
                    if delta != int(result["has_token"] and not result.get("error")):
                        fail("REQUEST_TOKEN_EVOLUTION", row, "result token not reflected exactly once")
                    if (result["finished"] or result.get("error")) and after_requests[rid]["status"] not in TERMINAL:
                        fail("PHASE_LIFECYCLE", row, "terminal/error result not applied")
                    if row["phase"] == "decode" and rid in old_slots and rid in new_slots:
                        if old_slots[rid]["pending_first"] is True:
                            expected_width = old_slots[rid]["cache"]["length"]
                        else:
                            related = [f for f in phase["forwards"] if rid in f["members"]]
                            expected_width = related[0]["cache_before"]["length"] + 1 if related else None
                        if new_slots[rid]["cache"]["length"] != expected_width:
                            fail("FORWARD_KV_EVOLUTION", row, "split-row cache does not match first-token/forward evolution")
        elif kind.startswith("pool_"):
            slots = row.get("slots")
            if slots is None:
                missing("SLOT_UNIQUENESS", row, "pool mapping absent")
            elif len(set(slots.values())) != len(slots):
                fail("SLOT_UNIQUENESS", row, "pool assigns a slot twice")
            if isinstance(slots, dict):
                rid = row.get("request_id")
                if row.get("before_slot") != pool_slots.get(rid) or row.get("after_slot") != slots.get(rid):
                    fail("SLOT_EVENT_CONSISTENCY", row, "pool request/slot event differs from actual mapping")
                if {k: v for k, v in slots.items() if k != rid} != {k: v for k, v in pool_slots.items() if k != rid}:
                    fail("SLOT_EVENT_CONSISTENCY", row, "one request operation changed another slot owner")
                pool_slots = dict(slots)
        elif kind == "release_end":
            rid = row["request_id"]
            if row.get("error") is not None:
                unconfirmed.add(rid)
                missing("V2_CLEANUP_RETRY_REPRESENTATION", row, "v2 terminal states omit pending failed cleanup resources")
            elif rid in row.get("snapshot", {}).get("slots", {}):
                fail("RELEASE_TRUTHFUL", row, "release succeeded but slot remains")
            else:
                unconfirmed.discard(rid)
        elif kind == "close_end":
            last_close = row
    if last_close is None:
        missing("FINAL_DRAIN", {}, "no runtime close observation")
    elif (last_close.get("error") is not None or last_close.get("snapshot", {}).get("requests") != {} or
          last_close.get("snapshot", {}).get("worker", {}).get("slots") != {} or unconfirmed or pool_slots):
        fail("FINAL_DRAIN", last_close, "cleanup incomplete or failed")
    if ending.get("errors") or ending.get("events") != len(rows) or rows[-1].get("kind") != "collector_end":
        fail("COLLECTOR_COMPLETE", rows[-1], "collector incomplete/errors")
    for fid, begin in forwards.items():
        if not any(r.get("kind") == "forward_end" and r.get("forward_id") == fid for r in rows):
            missing("COLLECTOR_COMPLETE", begin, "forward completion missing")
    for pid, phase in phases.items():
        if "worker" not in phase or "end" not in phase:
            missing("EXECUTOR_WORKER_MEMBERS", phase["submit"], "worker boundary incomplete")
    if any(ending.get(k) != identity[k] for k in IDENTITY):
        fail("TRACE_IDENTITY", rows[-1], "end marker identity differs from raw trace")
    checks = [{"code": code, "status": "FAIL" if code in problems else "UNKNOWN" if code in unknown else "PASS",
               "evidence": problems.get(code, []) + unknown.get(code, [])} for code in (*codes, *sorted(set(unknown) - set(codes)))]
    checks += [{"code": "FROZEN_SCHEDULE_EQUIVALENCE", "status": "UNKNOWN",
                "evidence": "observed local conformance does not prove HTTP/owner/scheduler transition equivalence"},
               {"code": "FINITE_STATE_EXHAUSTION", "status": "UNKNOWN",
                "evidence": "no exhaustive graph proof for errors/cleanup retry/admission timing"},
               {"code": "TAXONOMY_V2_STEP_RELATION", "status": "UNKNOWN",
                "evidence": "local transition predicates checked; full v2 transitions() successor relation does not model HTTP owner/admission/cleanup retry"}]
    totals.update(registered_requests=len(registered), worker_member_requests=len(requested_ids),
                  linked_worker_requests=len(linked_ids))
    return {"schema_version": 1, "status": "FAIL" if problems else "UNKNOWN", "checks": checks,
        "reason_codes": [c["code"] for c in checks if c["status"] != "PASS"], "association": totals,
        "closure": "UNKNOWN" if problems else "PARTIALLY_CLOSED", "admitted": False,
        "abstract_model": {"kind": "taxonomy v2 local transition predicates; not global reachability",
                           "counterexamples": problems, "unobservables": unknown,
                           "search": {"exhausted": False, "state_total": None, "transition_total": None,
                               "terminal_total": None, "unexplored_total": None,
                               "reason": "no complete owner/admission/within-phase failure and cleanup-retry state relation"}}}


def check_directory(directory, *, expected_identity=None):
    directory = Path(directory)
    try:
        index = json.loads((directory / "trace-index.json").read_text(encoding="utf-8"))
        for name in ("runtime-trace.jsonl", "trace-end.json"):
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != index["sha256"][name]:
                raise ValueError("trace SHA256 mismatch: " + name)
        rows = [json.loads(line) for line in (directory / "runtime-trace.jsonl").read_text(encoding="utf-8").splitlines()]
        ending = json.loads((directory / "trace-end.json").read_text(encoding="utf-8"))
        if any(ending.get(k) != index.get(k) for k in IDENTITY):
            raise ValueError("index/end identity mismatch")
        if expected_identity is not None and any(ending.get(k) != value for k, value in expected_identity.items()):
            raise ValueError("external expected source/config/process identity mismatch")
        return check_rows(rows, ending)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        return {"status": "FAIL", "closure": "UNKNOWN", "admitted": False,
            "reason_codes": ["TRACE_ARTIFACT_INTEGRITY"], "error": str(error)}


def seal(directory):
    directory = Path(directory)
    ending = json.loads((directory / "trace-end.json").read_text(encoding="utf-8"))
    write_new(directory / "trace-index.json", {**{k: ending[k] for k in IDENTITY},
        "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                   for name in ("runtime-trace.jsonl", "trace-end.json")}})
    result = check_directory(directory)
    write_new(directory / "trace-check.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = check_directory(args.directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("admitted") else 1)
