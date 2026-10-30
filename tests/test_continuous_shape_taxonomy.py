"""Pure synthetic evidence; no model loading, CUDA or performance execution."""
from copy import deepcopy
import json

import pytest

from benchmarks.runtime.online_http.artifacts import write_new
from benchmarks.runtime.online_http.config import load_config
from benchmarks.runtime.online_http.shapes import build_manifest, digest, shape_key
from benchmarks.runtime.online_http.taxonomy import (
    SOURCE_FILES, build_taxonomy, check_shape, cohort_witnesses, enumerate_model,
    initial_state, make_shape, transitions,
)
from benchmarks.runtime.online_http import taxonomy_audit


@pytest.fixture
def taxonomy():
    config = load_config("benchmarks/runtime/online_http/configs/qwen_admission_v2.json")
    trace = [{**config.data["requests"][i], "observed_prompt_tokens": p, "max_tokens": g}
             for i, (p, g) in enumerate(((3, 3), (5, 4), (7, 2)))]
    manifest = build_manifest(config, trace,
        {"sha": "a" * 40, "dirty": True, "source_tree_sha256": "b" * 64},
        {"layers": 2, "kv_heads": 2, "head_dim": 4})
    return build_taxonomy(manifest, {n: "c" * 64 for n in SOURCE_FILES}, capacity=2)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("batch", [1, 2])
def test_phase_and_batch_rules(taxonomy, phase, batch):
    row = make_shape(taxonomy, phase, batch, 5)
    assert check_shape(taxonomy, row)["status"] == "PASS"
    assert row["input_shape"] == [batch, 5 if phase == "prefill" else 1]
    assert row["attention_mask_shape"] == [batch, 5 if phase == "prefill" else 6]
    assert len(row["kv_tensor_shapes_before"]) == (0 if phase == "prefill" else 2)


def iteration(taxonomy, state):
    return next(s for kind, s, _ in transitions(taxonomy, state) if kind == "iteration")


def test_first_token_has_no_forward_and_decode_grows(taxonomy):
    state = (("W", 0, 0), ("W", 0, 0), ("N", 0, 0))
    first = iteration(taxonomy, state)
    assert first[:2] == (("F", 0, 5), ("F", 0, 5))
    events = next(e for e in transitions(taxonomy, first) if e[0] == "iteration")
    assert events[2] == ()
    second = events[1]
    assert second[:2] == (("D", 1, 5), ("D", 1, 5))
    third = iteration(taxonomy, second)
    assert third[:2] == (("D", 2, 6), ("D", 2, 6))


@pytest.mark.parametrize("reason", ["cancel", "error", "stop"])
def test_terminal_removal_preserves_survivor_width(taxonomy, reason):
    state = (("D", 2, 6), ("D", 2, 6), ("W", 0, 0))
    removed = next(s for kind, s, _ in transitions(taxonomy, state) if kind == reason and s[0][0] == "T")
    assert removed[1] == state[1]
    step = next(e for e in transitions(taxonomy, removed) if e[0] == "iteration")
    assert [r["phase"] for r in step[2]] == ["decode", "prefill"]
    assert step[1][2] == ("F", 0, 7)


def test_normal_finish_does_not_admit_same_iteration(taxonomy):
    state = (("D", 2, 6), ("D", 2, 6), ("W", 0, 0))
    step = next(e for e in transitions(taxonomy, state) if e[0] == "iteration")
    assert step[1][0][0] == "T"
    assert step[1][2][0] == "W"  # plan-start capacity was full
    assert [r["phase"] for r in step[2]] == ["decode"]


def test_join_and_regroup_keeps_padding(taxonomy):
    # Survivor already inherited six columns; newcomer prefilled at seven.
    state = (("T", 3, 0), ("D", 2, 6), ("F", 0, 7))
    first = iteration(taxonomy, state)
    assert first[1] == ("D", 3, 7) and first[2] == ("D", 1, 7)
    next_step = next(e for e in transitions(taxonomy, first) if e[0] == "iteration")
    row = next_step[2][0]
    assert row["input_shape"] == [2, 1]
    assert row["attention_mask_shape"] == [2, 8]
    assert row["kv_tensor_shapes_before"][0]["keys"] == [2, 2, 7, 4]


def test_reordered_fields_and_duplicate_records_are_same_shape(taxonomy):
    row = make_shape(taxonomy, "decode", 2, 5)
    copy = dict(reversed(list(row.items())))
    copy["observation_id"] = "another-record"
    assert shape_key(row) == shape_key(copy)
    assert check_shape(taxonomy, copy)["status"] == "PASS"


@pytest.mark.parametrize("change", [
    {"attention_mask_shape": [2, 99]}, {"padding_width": 2}, {"phase": "unknown"},
    {"kv_length_before": 999}, {"mode": "legacy"},
])
def test_invalid_extra_not_hidden(taxonomy, change):
    row = make_shape(taxonomy, "decode", 2, 5)
    row.update(change)
    assert check_shape(taxonomy, row)["status"] != "PASS"


def test_unknown_cache_dimension_stays_unknown(taxonomy):
    row = make_shape(taxonomy, "decode", 2, 5)
    row["kv_tensor_shapes_before"][0]["keys"] = None
    assert check_shape(taxonomy, row)["status"] == "UNKNOWN"


def test_wrong_layer_count_or_head_dimension_rejected(taxonomy):
    row = make_shape(taxonomy, "decode", 2, 5)
    row["kv_tensor_shapes_before"].pop()
    assert check_shape(taxonomy, row)["status"] == "FAIL"
    row = make_shape(taxonomy, "decode", 2, 5)
    row["kv_tensor_shapes_before"][0]["keys"][1] = 9
    assert check_shape(taxonomy, row)["status"] == "FAIL"


def test_batch_limit_and_unknown_prompt_width(taxonomy):
    with pytest.raises(ValueError):
        make_shape(taxonomy, "decode", 3, 5)
    assert check_shape(taxonomy, make_shape(taxonomy, "prefill", 2, 4))["status"] == "FAIL"


def test_all_conditional_witnesses_rule_derived_and_deterministic(taxonomy):
    a = cohort_witnesses(taxonomy)
    assert a == cohort_witnesses(deepcopy(taxonomy))
    assert len({shape_key(r["shape"]) for r in a}) == len(a)
    assert all(check_shape(taxonomy, r["shape"])["status"] == "PASS" for r in a)
    assert all(r["reachability"].startswith("CONDITIONAL") for r in a)


def test_model_enumeration_reproducible_and_not_clock_proof(taxonomy):
    a = enumerate_model(taxonomy, max_states=200)
    assert a == enumerate_model(taxonomy, max_states=200)
    assert a["states"] <= 200 and a["enumeration_status"] == "UNKNOWN"
    assert a["schedule_equivalence"] == "UNKNOWN"
    assert a["closure"] == "PARTIALLY_CLOSED"


def test_small_model_graph_can_exhaust_without_claiming_runtime_closure(taxonomy):
    small = deepcopy(taxonomy)
    small["requests"] = small["requests"][:1]
    small["constraints"]["max_active"] = small["constraints"]["max_batch_size"] = 1
    result = enumerate_model(small, max_states=1000)
    assert result["abstract_graph_exhausted"] and result["enumeration_status"] == "PASS"
    assert result["schedule_equivalence"] == "UNKNOWN"


def test_arrival_and_invalid_state(taxonomy):
    event = transitions(taxonomy, initial_state(taxonomy))[0]
    assert event[0] == "ordered-arrival" and event[1][0][0] == "W"
    with pytest.raises(ValueError):
        transitions(taxonomy, (("D", 9, 1), ("N", 0, 0), ("N", 0, 0)))


def test_raw_references_duplicates_and_missing_membership(taxonomy, tmp_path):
    shape = make_shape(taxonomy, "prefill", 2, 5)
    for mode in ("legacy", "continuous"):
        (tmp_path / mode).mkdir()
        rows = [] if mode == "legacy" else [{**shape, "kind": "forward_completed", "success": True,
            "execution_stage": "warmup", "observation_id": "o" + str(i), "call_id": "c" + str(i),
            "pid": 10, "run_id": "synthetic", "monotonic_s": 1. + i, "collector": "forward-boundary-v1"} for i in range(2)]
        (tmp_path / mode / "observer.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    replay = {"checks": [{"code": "SHAPE_UNDECLARED_OBSERVATIONS", "evidence": [shape]}]}
    records = taxonomy_audit.classify_extras(tmp_path, taxonomy, replay)
    assert records[0]["category"] == "KNOWN_RULE_OMISSION_CONDITIONAL"
    assert records[0]["observation_count"] == 2 and records[0]["duplicate_observations"] == 1
    assert records[0]["raw_references"][1]["line"] == 2
    assert records[0]["actual_request_membership"]["request_ids"] is None
    assert records[0]["full_execution_legality"] == "UNKNOWN"


def test_changed_rule_versions_change_hash(taxonomy):
    changed = deepcopy(taxonomy)
    changed["rules"][0]["output"] += "different"
    assert digest(taxonomy) != digest(changed)


def test_generated_artifacts_and_exclusive_output(taxonomy, tmp_path, monkeypatch):
    # Synthetic transport evidence: v1 authenticity is separately tested by
    # test_online_http_shapes; this test exercises only the v2 artifact writer.
    root, out = tmp_path / "synthetic-history", tmp_path / "new-audit"
    root.mkdir()
    config = load_config("benchmarks/runtime/online_http/configs/qwen_admission_v2.json")
    manifest = build_manifest(config, taxonomy["requests"], taxonomy["model_identity"]["git"], taxonomy["geometry"])
    shape = make_shape(taxonomy, "prefill", 2, 5)
    replay = {"checks": [{"code": "SHAPE_UNDECLARED_OBSERVATIONS", "evidence": [shape]}]}
    for name, value in (("config.json", config.data), ("expected-shapes.json", manifest),
                        ("shape-coverage.json", replay), ("preflight.json", {}),
                        ("shape-proof-index.json", {}), ("model-config.json", {}), ("artifacts.json", {})):
        write_new(root / name, value)
    for mode in ("legacy", "continuous"):
        (root / mode).mkdir()
        row = {**shape, "kind": "forward_completed", "success": True, "execution_stage": "warmup",
            "observation_id": "observed-1", "call_id": "call-1", "pid": 1, "run_id": root.name,
            "monotonic_s": 1.0, "collector": "forward-boundary-v1"}
        (root / mode / "observer.jsonl").write_text(json.dumps(row) if mode == "continuous" else "", encoding="utf-8")
    monkeypatch.setattr(taxonomy_audit, "verify_preflight_artifacts", lambda _p: {"status": "PASS"})
    monkeypatch.setattr(taxonomy_audit, "verify_bundle", lambda _p: replay)
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = taxonomy_audit.run_audit(root, out, max_states=20)
    assert not result["admitted"] and result["formal_requests_launched"] == 0
    assert result["counts"]["extra_shapes"] == 1
    diff = json.loads((out / "manifest-diff-v2.json").read_text(encoding="utf-8"))
    assert diff["removed_required"] == diff["new_admission_allowed_shapes"] == 0
    assert before == {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        taxonomy_audit.run_audit(root, out, max_states=20)
    with pytest.raises(ValueError, match="historical"):
        taxonomy_audit.run_audit(root, root / "forbidden", max_states=20)


def test_invalid_historical_integrity_stops_before_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(taxonomy_audit, "verify_preflight_artifacts", lambda _p: {"status": "FAIL"})
    with pytest.raises(ValueError, match="integrity"):
        taxonomy_audit.run_audit(tmp_path / "history", tmp_path / "output")
    assert not (tmp_path / "output").exists()
