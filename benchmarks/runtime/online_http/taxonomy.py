"""Continuous shape grammar v2; not a performance or admission allowlist.

Local forward consistency, conditional execution witnesses and whole-trajectory
reachability are different proofs. This module never substitutes one for another.
It does not call or replace the serving scheduler, worker or v1 coverage gate.
"""
from __future__ import annotations

from collections import deque
from itertools import combinations

from .shapes import digest, observable, shape_key

VERSION = 2
SOURCE_FILES = (
    "src/minillm/continuous/inputs.py",
    "src/minillm/workers/transformers_batch_worker.py",
    "src/minillm/workers/batched_transformers_worker.py",
    "src/minillm/continuous/iteration_scheduler.py",
    "src/minillm/continuous/executor.py",
    "src/minillm/memory/slot_kv_pool.py",
)
RULES = (
    {"id": "P2", "precondition": "distinct valid waiting requests; 1<=B<=2; capacity available",
     "operation": "W=max(prompt_lengths); left pad; empty full-attention cache",
     "output": "ids=mask=[B,W]; cache_before=0; cache tensors=[]; split rows retain W",
     "source": [SOURCE_FILES[0] + "::BatchInputBuilder.build_prefill",
                SOURCE_FILES[1] + "::TransformersBatchWorker.prefill_batch"]},
    {"id": "D2", "precondition": "valid DECODE sessions with pending_first=False",
     "operation": "K=max(stored row cache widths); left-pad every layer to K; forward one token",
     "output": "ids=[B,1]; mask=[B,K+1]; L layers of keys/values [B,H,K,D]; split rows keep K+1",
     "source": [SOURCE_FILES[1] + "::TransformersBatchWorker._merge",
                SOURCE_FILES[1] + "::TransformersBatchWorker.decode_batch"]},
    {"id": "F2", "precondition": "pending_first=True after prefill",
     "operation": "sample saved prefill logits; emit first token; pending_first=False",
     "output": "no model forward; cache/mask width unchanged; generated count +1",
     "source": [SOURCE_FILES[1] + "::TransformersBatchWorker.decode_batch"]},
    {"id": "R2", "precondition": "split row cache exists; next ready membership may change",
     "operation": "normal terminal/cancel/error removes request; new prefill uses only plan-start free slots",
     "output": "survivor width does not shrink; regroup uses maximum survivor/newcomer width",
     "source": [SOURCE_FILES[1] + "::TransformersBatchWorker._split",
                SOURCE_FILES[3] + "::IterationScheduler.build_plan",
                SOURCE_FILES[4] + "::BatchExecutor.execute_plan"]},
    {"id": "S2", "precondition": "serial worker, capacity validated, release confirmed",
     "operation": "slot owns one independent split-row cache; slot IDs are not tensor row dimensions",
     "output": "stored row batch dimension=1; merged tensor batch dimension=ready group size",
     "source": [SOURCE_FILES[5] + "::SlotKVPool", SOURCE_FILES[1] + "::TransformersBatchWorker._split"]},
)


def build_taxonomy(manifest, source_hashes, *, capacity):
    trace = manifest["trace"]
    if type(capacity) is not int or not 1 <= capacity <= 2:
        raise ValueError("validated capacity must be within actual worker batch limit")
    if not trace or any(type(r.get("observed_prompt_tokens")) is not int or
                        r["observed_prompt_tokens"] <= 1 or type(r.get("max_tokens")) is not int or
                        r["max_tokens"] <= 0 for r in trace):
        raise ValueError("finite positive lengths and unambiguous prompts required")
    if len({r["request_id"] for r in trace}) != len(trace):
        raise ValueError("one submission per distinct request per epoch required")
    prompts = [r["observed_prompt_tokens"] for r in trace]
    geometry = manifest["geometry"]
    if any(type(geometry.get(k)) is not int or geometry[k] <= 0 for k in ("layers", "kv_heads", "head_dim")):
        raise ValueError("full-attention cache geometry required")
    # One ready forward increments the maximum stored width by at most one,
    # and consumes >=1 of the finite remaining generated tokens. Joining only
    # takes max; neither removal nor first-token sampling increases it.
    forward_budget = sum(r["max_tokens"] - 1 for r in trace)
    width_bound = max(prompts) + forward_budget
    prefill_cardinality = sum(len({max(g) for g in combinations(prompts, b)}) for b in range(1, capacity + 1))
    decode_cardinality = sum(width_bound - min(prompts) + 1 for b in range(1, capacity + 1)
                             if sum(r["max_tokens"] >= 2 for r in trace) >= b)
    return {"schema_version": VERSION, "kind": "conditional_continuous_shape_taxonomy",
        "rules": list(RULES), "rule_sha256": digest(list(RULES)),
        "source_hashes": {name: source_hashes[name] for name in SOURCE_FILES},
        "model_identity": {k: manifest[k] for k in ("config_sha256", "model_revision", "dtype", "device", "git")},
        "trace_sha256": manifest["trace_sha256"], "requests": trace, "geometry": geometry,
        "constraints": {"max_batch_size": capacity, "max_active": capacity, "requests_per_drained_epoch": len(trace),
            "min_prompt": min(prompts), "max_prompt": max(prompts),
            "ready_forward_budget": forward_budget, "max_cache_before_bound": width_bound,
            "cache_type": "full-attention cloned DynamicCache rows; no sliding/truncation",
            "scope": "fixed requests, at most once each per drained epoch; no retries/re-prefill/new workload"},
        "finite_space": {"status": "PASS", "proof": "global maximum width <= Pmax+sum(G_i-1); finite B,L,H,D and request budgets",
            "prefill_shape_upper_bound": prefill_cardinality, "decode_shape_upper_bound": decode_cardinality,
            "continuous_shape_cardinality_upper_bound": prefill_cardinality + decode_cardinality,
            "cardinality_semantics": "finite over-approximation only; not all these shapes are proved reachable or required",
            "not_proved": ["exact reachable set for frozen wall-clock arrivals and admission",
                "historical request-to-forward/slot-row association",
                "all error branches within a worker phase"]},
        "closure": "PARTIALLY_CLOSED", "admission_effect": "NONE; v1 gate unchanged"}


def make_shape(taxonomy, phase, batch_size, width):
    if type(batch_size) is not int or not 1 <= batch_size <= taxonomy["constraints"]["max_batch_size"]:
        raise ValueError("batch outside worker limit")
    if phase not in {"prefill", "decode"} or type(width) is not int or width < 1:
        raise ValueError("invalid phase or width")
    geometry = taxonomy["geometry"]
    return {"mode": "continuous", "phase": phase, "batch_size": batch_size,
        "input_shape": [batch_size, width if phase == "prefill" else 1],
        "attention_mask_shape": [batch_size, width if phase == "prefill" else width + 1],
        "padding_width": width if phase == "prefill" else 1,
        "kv_length_before": 0 if phase == "prefill" else width,
        "kv_tensor_shapes_before": [] if phase == "prefill" else [
            {"keys": [batch_size, geometry["kv_heads"], width, geometry["head_dim"]],
             "values": [batch_size, geometry["kv_heads"], width, geometry["head_dim"]]}
            for _ in range(geometry["layers"])]}


def check_shape(taxonomy, row):
    """Exact local tensor rule, never whole-schedule legality."""
    if not observable(row):
        return {"status": "UNKNOWN", "reason": "missing/inconsistent observable dimensions"}
    try:
        phase, batch = row["phase"], row["batch_size"]
        width = row["padding_width"] if phase == "prefill" else row["kv_length_before"]
        expected = make_shape(taxonomy, phase, batch, width)
    except ValueError as error:
        return {"status": "FAIL", "reason": str(error)}
    if shape_key(expected) != shape_key(row):
        return {"status": "FAIL", "reason": "exact input/mask/cache layer rule mismatch"}
    if phase == "prefill":
        possible = {max(r["observed_prompt_tokens"] for r in group)
                    for group in combinations(taxonomy["requests"], batch)}
        valid = width in possible
    else:
        c = taxonomy["constraints"]
        valid = (sum(r["max_tokens"] >= 2 for r in taxonomy["requests"]) >= batch and
                 c["min_prompt"] <= width <= c["max_cache_before_bound"])
    return {"status": "PASS" if valid else "FAIL", "rule": "P2" if phase == "prefill" else "D2",
        "reason": "source-derived local constraints only" if valid else "outside frozen workload bounds",
        "trajectory_legality": "UNKNOWN"}


def cohort_witnesses(taxonomy):
    """Generate conditional no-rejoin cohorts without reading any observations.

    These are exact local derivations conditional on the cohort being admitted
    together, not claims that FIFO plus frozen arrival times can reach each one.
    """
    found = {}
    for batch in range(1, taxonomy["constraints"]["max_batch_size"] + 1):
        for group in combinations(taxonomy["requests"], batch):
            width = max(r["observed_prompt_tokens"] for r in group)
            ids = [r["request_id"] for r in group]
            row = make_shape(taxonomy, "prefill", batch, width)
            found.setdefault(shape_key(row), {"shape": row, "rule": "P2", "cohort_assumption": ids,
                "forward_step": 0, "reachability": "CONDITIONAL_NOT_FROZEN_SCHEDULE_PROOF"})
            for step in range(1, min(r["max_tokens"] for r in group)):
                row = make_shape(taxonomy, "decode", batch, width + step - 1)
                found.setdefault(shape_key(row), {"shape": row, "rule": "D2", "cohort_assumption": ids,
                    "forward_step": step, "reachability": "CONDITIONAL_NOT_FROZEN_SCHEDULE_PROOF"})
    return [found[key] for key in sorted(found)]


def initial_state(taxonomy):
    """(not-arrived/waiting/pending-first/ready/terminal, emitted, cache width)."""
    return tuple(("N", 0, 0) for _ in taxonomy["requests"])


def transitions(taxonomy, state):
    """Pure proof-model transitions, not a runtime scheduler implementation.

    Ordered arrivals have no wall-clock duration in this abstraction. Terminal
    removals model safe boundaries, not failure during partial installation.
    """
    requests = taxonomy["requests"]
    if len(state) != len(requests):
        raise ValueError("state/request length mismatch")
    for row, request in zip(state, requests):
        stage, count, width = row
        if (stage not in {"N", "W", "F", "D", "T"} or type(count) is not int or
                not 0 <= count <= request["max_tokens"] or type(width) is not int or
                (stage in {"N", "W", "T"} and width != 0) or
                (stage in {"N", "W", "F"} and count != 0) or
                (stage in {"F", "D"} and not taxonomy["constraints"]["min_prompt"] <= width <= taxonomy["constraints"]["max_cache_before_bound"]) or
                (stage == "D" and not 1 <= count < request["max_tokens"])):
            raise ValueError("invalid abstract request state")
    if sum(row[0] in {"F", "D"} for row in state) > taxonomy["constraints"]["max_active"]:
        raise ValueError("active state exceeds capacity")
    events = []
    unseen = [i for i, (stage, _, _) in enumerate(state) if stage == "N"]
    if unseen:
        i = unseen[0]
        updated = list(state)
        updated[i] = ("W", 0, 0)
        events.append(("ordered-arrival", tuple(updated), ()))
    for i, (stage, count, width) in enumerate(state):
        if stage in {"W", "F", "D"}:
            for reason in (("stop", "cancel", "error") if stage == "D" else ("cancel", "error")):
                updated = list(state)
                updated[i] = ("T", count, 0)
                events.append((reason, tuple(updated), ()))
    active = [i for i, r in enumerate(state) if r[0] in {"F", "D"}]
    waiting = [i for i, r in enumerate(state) if r[0] == "W"]
    prefill = waiting[:taxonomy["constraints"]["max_active"] - len(active)]
    ready = [i for i in active if state[i][0] == "D"]
    shapes = []
    updated = list(state)
    if ready:
        width = max(state[i][2] for i in ready)
        shapes.append(make_shape(taxonomy, "decode", len(ready), width))
        for i in ready:
            updated[i] = ("D", state[i][1], width + 1)
    for i in active:
        count = state[i][1] + 1
        updated[i] = (("T", count, 0) if count >= requests[i]["max_tokens"] else
                      ("D", count, updated[i][2]))
    if prefill:
        width = max(requests[i]["observed_prompt_tokens"] for i in prefill)
        shapes.append(make_shape(taxonomy, "prefill", len(prefill), width))
        for i in prefill:
            updated[i] = ("F", 0, width)
    if active or prefill:
        events.append(("iteration", tuple(updated), tuple(shapes)))
    return events


def enumerate_model(taxonomy, *, max_states=2000):
    """Bounded deterministic BFS; budget exhaustion remains UNKNOWN.

    Even exhaustion of this abstract model is not proof of actual wall-clock,
    owner/admission or within-phase error semantics.
    """
    if type(max_states) is not int or max_states < 1:
        raise ValueError("positive state budget required")
    start = initial_state(taxonomy)
    visited, pending, shapes = {start}, deque([start]), {}
    truncated = False
    while pending:
        state = pending.popleft()
        for _, successor, rows in transitions(taxonomy, state):
            for row in rows:
                shapes.setdefault(shape_key(row), row)
            if successor not in visited:
                if len(visited) >= max_states:
                    truncated = True
                    continue
                visited.add(successor)
                pending.append(successor)
    return {"schema_version": VERSION, "states": len(visited), "state_budget": max_states,
        "abstract_graph_exhausted": not truncated,
        "enumeration_status": "UNKNOWN" if truncated else "PASS",
        "schedule_equivalence": "UNKNOWN", "closure": "PARTIALLY_CLOSED",
        "shapes": [shapes[k] for k in sorted(shapes)],
        "reason": "relaxed ordered arrivals and boundary failures; no frozen-clock/full-error equivalence proof"}
