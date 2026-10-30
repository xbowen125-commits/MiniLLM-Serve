"""Pure physical-KV trace relation; padded width is NOT effective token length.

Uses explicit phase/forward/iteration/request IDs only. No model execution,
timestamp joins, tensor-content equivalence or inferred mask sums.
"""
from collections import Counter


def verdict(checks):
    return "FAIL" if any(c["status"] == "FAIL" for c in checks) else "UNKNOWN" if any(c["status"] == "UNKNOWN" for c in checks) else "PASS"


def check_forwards(events):
    """events are (raw event, physical file/line/hash reference) pairs."""
    reports = []
    def find(kind, field, value):
        return [(r, ref) for r, ref in events if r.get("kind") == kind and value is not None and r.get(field) == value]
    for row, ref in events:
        if row.get("kind") != "forward_begin":
            continue
        checks, links = [], {"forward_begin": ref}
        def check(code, condition, evidence, *, known=True):
            checks.append({"code": code, "status": "UNKNOWN" if not known else "PASS" if condition else "FAIL", "evidence": evidence})
        def one(kind, key, value):
            found = find(kind, key, value)
            check("LINK_" + kind.upper(), len(found) == 1, [f for _, f in found], known=bool(found))
            if len(found) == 1:
                links[kind] = found[0][1]
                return found[0][0]
            return {}
        fid, phase_id, iteration = row.get("forward_id"), row.get("phase_id"), row.get("iteration_id")
        check("MODEL_CALL_ID_NOT_LINKED", False, ref, known=False)
        check("FORWARD_ID_UNIQUE", len(find("forward_begin", "forward_id", fid)) == 1, ref, known=fid is not None)
        check("ITERATION_PRESENT", True, ref, known=iteration is not None)
        end = one("forward_end", "forward_id", fid)
        submit = one("executor_submit", "phase_id", phase_id)
        worker = one("worker_begin", "phase_id", phase_id)
        worker_end = one("worker_end", "phase_id", phase_id)
        plan = one("executor_plan", "iteration_id", iteration)
        phase, members = row.get("phase"), row.get("members")
        members_observed = isinstance(members, list)
        check("PHASE_KNOWN", phase in {"prefill", "decode"}, ref, known=phase is not None)
        check("MEMBER_IDS", isinstance(members, list) and all(isinstance(r, str) and r for r in members) and
              len(members) == len(set(members)), ref, known=members is not None)
        members = members if isinstance(members, list) else []
        if submit and worker and plan:
            check("PLAN_EXECUTOR_WORKER_MEMBERS", submit.get("members") == worker.get("members") == plan.get(phase) and
                  set(members) <= set(worker.get("members") or []) and
                  members == [r for r in worker.get("members", []) if r in members],
                  [links[k] for k in ("executor_submit", "worker_begin", "executor_plan")], known=members_observed)
            check("PHASE_CALL_IDENTITY", submit.get("phase") == worker.get("phase") == phase and
                  submit.get("iteration_id") == worker.get("iteration_id") == iteration, ref)
        ids, mask = row.get("input_shape"), row.get("mask_shape")
        before, after = row.get("cache_before") or {}, end.get("cache_after") or {}
        width0, width1 = before.get("length"), after.get("length")
        dims_known = isinstance(ids, list) and len(ids) == 2 and isinstance(mask, list) and len(mask) == 2 and type(width0) is int
        check("INPUT_MASK_KV_WIDTH", dims_known and ids[0] == len(members) == mask[0] and mask[1] == width0 + ids[1],
              ref, known=dims_known and members_observed)
        if end and end.get("error") is None:
            check("FORWARD_KV_GROWTH", dims_known and width1 == width0 + ids[1], links["forward_end"],
                  known=dims_known and type(width1) is int)
        elif end:
            check("FAILED_FORWARD_OUTPUT", False, links["forward_end"], known=False)

        def tensor_rule(cache, batch, width, evidence, empty=False):
            layers = cache.get("layers")
            known = isinstance(layers, list) and type(width) is int
            if not known:
                check("CACHE_DIMENSIONS", False, evidence, known=False)
                return None
            if empty:
                check("EMPTY_PREFILL_CACHE", width == 0 and layers == [], evidence)
                return None
            check("CACHE_LAYERS_NONEMPTY", bool(layers), evidence)
            if batch is None:
                check("CACHE_BATCH_MEMBERS_UNOBSERVED", False, evidence, known=False)
            geometry = []
            for layer in layers:
                k, v = layer.get("keys"), layer.get("values")
                ok = isinstance(k, list) and len(k) == 4 and isinstance(v, list) and len(v) == 4
                check("CACHE_ROW_DIMENSIONS", ok and k == v and (batch is None or k[0] == batch) and k[2] == width and
                      type(k[1]) is int and k[1] > 0 and type(k[3]) is int and k[3] > 0,
                      evidence, known=ok)
                if ok:
                    geometry.append((k[1], k[3]))
            check("LAYER_GEOMETRY_CONSISTENT", len(set(geometry)) == 1, evidence, known=bool(geometry))
            return (len(layers), *geometry[0]) if geometry else None

        observed_batch = len(members) if members_observed else None
        g0 = tensor_rule(before, observed_batch, width0, ref, empty=phase == "prefill")
        g1 = tensor_rule(after, observed_batch, width1, links.get("forward_end", {})) if end.get("error") is None else None
        if phase == "decode" and g0 is not None and g1 is not None:
            check("GEOMETRY_BEFORE_AFTER", g0 == g1, [ref, links.get("forward_end")])
        slot0 = (row.get("snapshot") or {}).get("slots")
        slot1 = (worker_end.get("snapshot") or {}).get("slots")
        if isinstance(slot0, dict):
            check("SLOT_UNIQUENESS", len({v.get("slot") for v in slot0.values()}) == len(slot0), ref)
            objects = [v.get("cache_object_identity") for v in slot0.values()]
            check("CACHE_OBJECT_UNIQUENESS", len(set(objects)) == len(objects), ref, known=None not in objects)
        else:
            check("SLOT_BEFORE_UNOBSERVED", False, ref, known=False)
        if phase == "prefill":
            inputs = [(r, f) for r, f in find("prefill_inputs", "phase_id", phase_id) if r.get("members") == members]
            lengths = [r.get("prompt_lengths") for r, _ in inputs]
            valid = bool(lengths) and all(isinstance(v, list) and len(v) == len(members) and
                all(type(n) is int and n > 0 for n in v) for v in lengths)
            check("PREFILL_PADDING", valid and all(v == lengths[0] for v in lengths) and dims_known and
                  max(lengths[0]) == ids[1] and width0 == 0, [f for _, f in inputs], known=valid and dims_known)
            effective_before = {r: 0 for r in members} if width0 == 0 else None
            prompt_lengths = lengths[0] if valid else None
        else:
            effective_before, prompt_lengths = None, None
            check("DECODE_EFFECTIVE_KV_LENGTH", False, ref, known=False)  # mask.sum is not recorded
            old = [slot0.get(r) for r in members] if isinstance(slot0, dict) else []
            known = bool(old) and all(isinstance(v, dict) and type((v.get("cache") or {}).get("length")) is int for v in old)
            check("DECODE_MERGE_PADDING", known and dims_known and ids[1] == 1 and
                  width0 == max(v["cache"]["length"] for v in old), ref, known=known and dims_known)
            if known:
                for value in old:
                    check("DECODE_READY_NOT_FIRST", value.get("pending_first") is False, ref,
                          known=value.get("pending_first") is not None)
                    tensor_rule(value["cache"], 1, value["cache"]["length"], ref)
        # Context snapshots are associated through explicit phase ID, not time.
        contexts = (submit.get("snapshot") or {}).get("requests") or {}
        row_reports = []
        for index, rid in enumerate(members):
            context = contexts.get(rid)
            if context is None:
                check("REQUEST_STATE_BEFORE", False, links.get("executor_submit", {}), known=False)
            elif phase == "decode":
                check("ACTIVE_DECODE_ONLY", context.get("status") not in {"finished", "failed", "cancelled"} and
                      context.get("phase") == "decode", links["executor_submit"])
            old = slot0.get(rid) if isinstance(slot0, dict) else None
            new = slot1.get(rid) if isinstance(slot1, dict) else None
            for value, evidence in [(old, ref), (new, links.get("worker_end", {}))]:
                if value is not None:
                    check("REQUEST_CACHE_OWNER", value.get("context_request_id") == rid and
                          value.get("context_matches_cache") is True, evidence,
                          known=value.get("context_request_id") is not None and value.get("context_matches_cache") is not None)
            if new is not None:
                check("SPLIT_ROW_WIDTH", (new.get("cache") or {}).get("length") == width1, links.get("worker_end", {}),
                      known=type(width1) is int)
                tensor_rule(new.get("cache") or {}, 1, width1, links.get("worker_end", {}))
                if phase == "decode" and old is not None:
                    check("SLOT_SURVIVES_UPDATE", old.get("slot") == new.get("slot"), links.get("worker_end", {}))
            else:
                check("POST_FORWARD_ROW_UNOBSERVED", False, links.get("worker_end", {}), known=False)
            # No raw split-row source indices/digests exist. Logical members are
            # directly known; copied tensor VALUE lineage is deliberately unknown.
            check("SPLIT_ROW_VALUE_LINEAGE", False, ref, known=False)
            row_reports.append({"request_id": rid, "input_row": index, "row_mapping_source": ref,
                "request_object_identity": context.get("request_object_identity") if context else None,
                "physical_kv_before": (old.get("cache") or {}).get("length") if old else width0 if phase == "prefill" else None,
                "effective_kv_before": effective_before.get(rid) if effective_before else None,
                "prompt_tokens_observed": prompt_lengths[index] if prompt_lengths else None,
                "slot_before": old.get("slot") if old else None, "slot_after": new.get("slot") if new else None,
                "physical_kv_after_observed": (new.get("cache") or {}).get("length") if new else None,
                "post_forward_row_value_source": "UNKNOWN; no raw split lineage/digest", "cache_after": new})
        physical = [c for c in checks if c["code"] not in {"MODEL_CALL_ID_NOT_LINKED", "DECODE_EFFECTIVE_KV_LENGTH", "SPLIT_ROW_VALUE_LINEAGE", "POST_FORWARD_ROW_UNOBSERVED"}]
        reports.append({"forward_id": fid, "call_id": fid,
            "call_id_namespace": "runtime forward wrapper UUID; NOT observer.jsonl model call_id",
            "model_call_id": None, "iteration_id": iteration, "phase_id": phase_id, "phase": phase,
            "members": members, "input_shape": ids, "mask_shape": mask, "kv_before": before, "kv_after": after,
            "observed_geometry": g1, "geometry_semantics": "observed tensor consistency; not a pinned-model attestation",
            "rows": row_reports, "evidence": links, "checks": checks,
            "physical_equations": verdict(physical), "status": verdict(checks)})
    return {"version": 1, "forwards": reports, "counts": dict(Counter(r["status"] for r in reports)),
        "physical_counts": dict(Counter(r["physical_equations"] for r in reports)),
        "status": verdict([c for r in reports for c in r["checks"]]) if reports else "UNKNOWN",
        "unknown_codes": sorted({c["code"] for r in reports for c in r["checks"] if c["status"] == "UNKNOWN"}),
        "scope": "physical widths vs effective mask lengths separated; no tensor-value lineage proof"}
