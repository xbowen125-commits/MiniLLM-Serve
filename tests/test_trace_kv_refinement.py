"""Deterministic physical-KV and abstract-path counterexamples; CPU only."""
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.runtime.online_http.kv_evolution import check_forwards
from benchmarks.runtime.online_http.trace_refinement import match_boundary, project_snapshot, audit, boundary_refinement, source_binding, event_references
from benchmarks.runtime.online_http.taxonomy import build_taxonomy, make_shape, SOURCE_FILES
from benchmarks.runtime.online_http.artifacts import write_new
from benchmarks.runtime.online_http.transition_relation import map_events
from benchmarks.runtime.online_http.retirement import check_retirement
from test_abstract_transition_relation import sample as retirement_rows


def cache(width, batch=1):
    return {"length": width, "layers": [{"keys": [batch, 2, width, 4], "values": [batch, 2, width, 4]}] if width else []}


def slot(rid, sid, width):
    return {"slot": sid, "context_request_id": rid, "context_matches_cache": True,
            "cache_object_identity": 100 + sid, "pending_first": False, "cache": cache(width)}


def raw_forward():
    before = {"A": slot("A", 0, 5), "B": slot("B", 1, 3)}
    after = {"A": slot("A", 0, 6), "B": slot("B", 1, 6)}
    rows = [
        {"kind": "executor_plan", "iteration_id": 1, "decode": ["A", "B"], "prefill": []},
        {"kind": "executor_submit", "iteration_id": 1, "phase_id": "p", "phase": "decode", "members": ["A", "B"],
         "snapshot": {"requests": {r: {"status": "running", "phase": "decode", "request_object_identity": i}
                                      for i, r in enumerate(["A", "B"])}}},
        {"kind": "worker_begin", "iteration_id": 1, "phase_id": "p", "phase": "decode", "members": ["A", "B"]},
        {"kind": "forward_begin", "iteration_id": 1, "phase_id": "p", "forward_id": "f", "phase": "decode",
         "members": ["A", "B"], "input_shape": [2, 1], "mask_shape": [2, 6], "cache_before": cache(5, 2),
         "snapshot": {"slots": before}},
        {"kind": "forward_end", "forward_id": "f", "error": None, "cache_after": cache(6, 2)},
        {"kind": "worker_end", "phase_id": "p", "phase": "decode", "snapshot": {"slots": after}},
    ]
    return [(r, {"file": "raw.jsonl", "line": i, "sha256": "raw", "event_id": str(i)}) for i, r in enumerate(rows, 1)]


def save_counterexample(directory, rows, result):
    write_new(directory / "synthetic-raw.json", {"evidence_kind": "synthetic_counterexample_not_gpu", "events": rows})
    write_new(directory / "verdict.json", result)


def test_physical_width_can_pass_without_proving_effective_length_or_values():
    result = check_forwards(raw_forward())
    assert result["physical_counts"] == {"PASS": 1}
    assert result["counts"] == {"UNKNOWN": 1}
    assert set(result["unknown_codes"]) == {"DECODE_EFFECTIVE_KV_LENGTH", "SPLIT_ROW_VALUE_LINEAGE", "MODEL_CALL_ID_NOT_LINKED"}
    a, b = result["forwards"][0]["rows"]
    assert a["physical_kv_before"] == 5 and b["physical_kv_before"] == 3
    assert a["effective_kv_before"] is None


@pytest.mark.parametrize("mutation,code", [
    ("growth", "FORWARD_KV_GROWTH"), ("padding", "INPUT_MASK_KV_WIDTH"),
    ("row_owner", "REQUEST_CACHE_OWNER"), ("reverse_members", "PLAN_EXECUTOR_WORKER_MEMBERS"),
    ("foreign_member", "PLAN_EXECUTOR_WORKER_MEMBERS"), ("terminal", "ACTIVE_DECODE_ONLY"),
    ("slot", "SLOT_UNIQUENESS"), ("split_width", "SPLIT_ROW_WIDTH"),
    ("cache_shape", "CACHE_ROW_DIMENSIONS"), ("merged_width", "DECODE_MERGE_PADDING"),
])
def test_physical_and_member_counterexamples(tmp_path, mutation, code):
    rows = raw_forward()
    forward, end, worker_end = rows[3][0], rows[4][0], rows[5][0]
    if mutation == "growth": end["cache_after"] = cache(7, 2)
    if mutation == "padding": forward["mask_shape"] = [2, 7]
    if mutation == "row_owner": worker_end["snapshot"]["slots"]["A"]["context_request_id"] = "B"
    if mutation == "reverse_members": forward["members"] = ["B", "A"]
    if mutation == "foreign_member": forward["members"] = ["A", "foreign"]
    if mutation == "terminal": rows[1][0]["snapshot"]["requests"]["A"]["status"] = "finished"
    if mutation == "slot": forward["snapshot"]["slots"]["B"]["slot"] = 0
    if mutation == "split_width": worker_end["snapshot"]["slots"]["A"]["cache"] = cache(5)
    if mutation == "cache_shape": end["cache_after"]["layers"][0]["keys"][0] = 1
    if mutation == "merged_width": forward["snapshot"]["slots"]["A"]["cache"] = cache(4)
    result = check_forwards(rows)
    save_counterexample(tmp_path, rows, result)
    assert result["status"] == "FAIL"
    assert any(c["code"] == code and c["status"] == "FAIL" for c in result["forwards"][0]["checks"])


@pytest.mark.parametrize("kind,field", [("forward_begin", "phase_id"), ("forward_begin", "members"),
    ("forward_begin", "mask_shape"), ("forward_end", "cache_after"), ("worker_end", "snapshot")])
def test_missing_kv_evidence_unknown(tmp_path, kind, field):
    rows = raw_forward()
    next(r for r, _ in rows if r["kind"] == kind).pop(field)
    result = check_forwards(rows)
    save_counterexample(tmp_path, rows, result)
    assert result["status"] == "UNKNOWN"


def test_missing_completion_and_duplicate_call_not_silently_joined():
    rows = raw_forward()
    assert check_forwards([(r, f) for r, f in rows if r["kind"] != "forward_end"])["status"] == "UNKNOWN"
    result = check_forwards(rows + [deepcopy(rows[4])])
    assert result["status"] == "FAIL"


def taxonomy():
    return build_taxonomy({"trace": [{"request_id": "A", "observed_prompt_tokens": 2, "max_tokens": 3},
                                    {"request_id": "B", "observed_prompt_tokens": 5, "max_tokens": 4}],
        "geometry": {"layers": 1, "kv_heads": 2, "head_dim": 4}, "config_sha256": "config",
        "model_revision": "fake", "dtype": "float32", "device": "cpu", "git": {}, "trace_sha256": "trace"},
        {p: "source" for p in SOURCE_FILES}, capacity=2)


def test_prefill_then_first_token_without_forward_then_decode():
    t = taxonomy()
    assert match_boundary(t, (("W", 0, 0), ("N", 0, 0)), (("F", 0, 2), ("N", 0, 0)),
                          [make_shape(t, "prefill", 1, 2)], {})["status"] == "PASS"
    assert match_boundary(t, (("F", 0, 2), ("N", 0, 0)), (("D", 1, 2), ("N", 0, 0)), [], {})["status"] == "PASS"
    assert match_boundary(t, (("D", 1, 2), ("N", 0, 0)), (("D", 2, 3), ("N", 0, 0)),
                          [make_shape(t, "decode", 1, 2)], {})["status"] == "PASS"


def test_early_stop_is_iteration_followed_by_explicit_stop_not_length():
    t = taxonomy()
    result = match_boundary(t, (("D", 1, 2), ("N", 0, 0)), (("T", 2, 0), ("N", 0, 0)),
                            [make_shape(t, "decode", 1, 2)], {0: "finished"})
    assert result["status"] == "PASS" and result["rules"] == ["iteration", "stop"]


def test_regroup_old_decode_new_prefill_and_first_token():
    t = taxonomy()
    a, b = (("D", 1, 2), ("W", 0, 0)), (("D", 2, 3), ("F", 0, 5))
    assert match_boundary(t, a, b, [make_shape(t, "decode", 1, 2), make_shape(t, "prefill", 1, 5)], {})["status"] == "PASS"
    after = (("T", 3, 0), ("D", 1, 5))
    assert match_boundary(t, b, after, [make_shape(t, "decode", 1, 3)], {0: "finished"})["status"] == "PASS"


def test_illegal_phase_and_forward_not_accepted_as_stutter():
    t = taxonomy()
    assert match_boundary(t, (("W", 0, 0), ("N", 0, 0)), (("D", 1, 2), ("N", 0, 0)), [], {})["status"] == "FAIL"
    assert match_boundary(t, (("F", 0, 2), ("N", 0, 0)), (("D", 1, 2), ("N", 0, 0)),
                          [make_shape(t, "decode", 1, 2)], {})["status"] == "FAIL"


@pytest.mark.parametrize("slot_present,cache_ref", [(True, True), (False, True), (False, None)])
def test_v2_projection_does_not_erase_pending_retirement(slot_present, cache_ref):
    state, missing, _ = project_snapshot({"requests": {"A": {"status": "cancelled", "phase": "finished",
        "generated": 1, "request_object_identity": 22, "has_cache_ref": cache_ref}},
        "worker": {"slots": {"A": slot("A", 0, 2)} if slot_present else {}}}, ["A"], {"A"}, {})
    assert state is None and missing == ["V2_TERMINAL_WITH_PENDING_RESOURCE:A"]


def test_absent_context_not_invented_terminal_without_ack():
    state, missing, _ = project_snapshot({"requests": {}, "worker": {"slots": {}}}, ["A"], {"A"}, {})
    assert state is None and missing == ["ABSENT_CONTEXT_WITHOUT_ACK:A"]


def test_clear_slot_and_reference_but_retry_record_not_projected_retired():
    snapshot = {"requests": {"A": {"status": "failed", "phase": "finished", "generated": 1,
        "request_object_identity": 22, "has_cache_ref": False}}, "worker": {"slots": {}, "retirements": {"A": {"attempts": 1}}}}
    state, missing, _ = project_snapshot(snapshot, ["A"], {"A"}, {})
    assert state is None and missing == ["V2_TERMINAL_WITH_PENDING_RESOURCE:A"]
    snapshot["worker"].pop("retirements")
    state, missing, _ = project_snapshot(snapshot, ["A"], {"A"}, {})
    assert state is None and missing == ["TERMINAL_RETIREMENT_RECORDS_UNOBSERVED:A"]


def test_existing_release_model_preserves_failure_and_ref_retirement():
    rows = retirement_rows()
    end = next(r for r in rows if r["kind"] == "release_end")
    end.update(error="release refused", request_cache_ref=True)
    end["snapshot"] = {"slots": {"A": {"slot": 0}}, "retirements": {"A": {"request_object_identity": 111}}}
    result = check_retirement(rows)
    assert any(r.get("resource") == "retry_required" for r in result["release_outcomes"])


def test_success_with_original_cache_ref_is_not_retired(tmp_path):
    rows = retirement_rows()
    next(r for r in rows if r["kind"] == "release_end")["request_cache_ref"] = True
    result = map_events(rows, input_sha256="raw")
    save_counterexample(tmp_path, rows, result)
    assert result["status"] == "FAIL"


def test_hash_tamper_rejected_before_output(tmp_path):
    source = tmp_path / "input"; source.mkdir()
    write_new(source / "artifacts.json", {"sha256": {"source.json": "bad"}})
    write_new(source / "source.json", {})
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="hash"):
        audit(source, output, Path("benchmarks/runtime/online_http/configs/qwen_admission_v2.json"))
    assert not output.exists()


def test_old_full_source_unknown_even_when_production_rule_hashes_match():
    target = {"source_tree_sha256": "old", "files": {p: "same" for p in SOURCE_FILES}}
    current = {"source_tree_sha256": "new", "files": deepcopy(target["files"])}
    result = source_binding(target, current)
    assert result["full_source"] == "UNKNOWN" and result["production_rule_source_hashes_equal"]


def test_without_explicit_operation_id_no_boundary_is_invented():
    result = boundary_refinement([({"kind": "step_end", "snapshot": {}}, {"line": 1})], taxonomy(), ["A", "B"])
    assert result["status"] == "UNKNOWN"


def test_snapshot_unknown_member_not_ignored():
    state, missing, _ = project_snapshot({"requests": {"ghost": {}}, "worker": {"slots": {}}}, ["A"], set(), {})
    assert state is None and missing == ["UNREGISTERED_CONTEXT_OR_SLOT_ID"]


def test_missing_end_keeps_boundary_unclosed():
    result = boundary_refinement([({"kind": "step_begin", "operation_id": "o", "snapshot": {}}, {"line": 1})], taxonomy(), ["A", "B"])
    assert result["status"] == "UNKNOWN" and result["mapping"][0]["code"] == "OPERATION_END_UNOBSERVED"


def test_independent_recalculation_deterministic_without_modifying_input():
    rows = raw_forward(); original = deepcopy(rows)
    assert check_forwards(rows) == check_forwards(deepcopy(rows))
    assert rows == original


def test_macro_end_reference_uses_real_event_id_not_line_or_position():
    events = event_references([({"kind": "step_end", "event_id": "actual-event-uuid"}, {"file": "raw", "line": 17, "sha256": "raw"})])
    assert events[0][1]["event_id"] == "actual-event-uuid"
    assert event_references([({}, {"line": 18})])[0][1]["event_id"] is None
