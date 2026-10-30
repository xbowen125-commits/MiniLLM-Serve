"""Synthetic proof tests only: no GPU, model loading or HTTP performance run."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.runtime.online_http.artifacts import write_new
from benchmarks.runtime.online_http.config import ProtocolConfig, load_config
from benchmarks.runtime.online_http.shapes import (
    IDENTITY, build_manifest, digest, save_proof, verify_bundle,
)


@pytest.fixture
def bundle(tmp_path):
    data = load_config("benchmarks/runtime/online_http/configs/qwen_admission_v2.json").data
    data.update(admission_limit=1)
    data["requests"] = [{**data["requests"][0], "max_tokens": 2}]
    config = ProtocolConfig.from_dict(data)
    trace = [{**config.data["requests"][0], "observed_prompt_tokens": 3}]
    git = {"sha": "a" * 40, "dirty": True, "source_tree_sha256": "b" * 64}
    manifest = build_manifest(config, trace, git, {"layers": 1, "kv_heads": 2, "head_dim": 4})
    write_new(tmp_path / "config.json", config.data)
    write_new(tmp_path / "expected-shapes.json", manifest)
    modes = []
    for mode, pid in (("legacy", 10), ("continuous", 11)):
        directory = tmp_path / mode
        directory.mkdir()
        ready = {k: manifest[k] for k in IDENTITY}
        ready.update(mode=mode, pid=pid)
        write_new(directory / "server-ready.json", ready)
        write_new(directory / "warmup-active", {**ready, "monotonic_s": 0.0})
        write_new(directory / "warmup-finished", {**ready, "started_at_s": 0.0,
            "finished_at_s": 100.0, "status": "PASS"})
        rows = [{**shape, **ready, "schema_version": 1, "run_id": tmp_path.name,
            "observation_id": mode + str(i), "expected_manifest_sha256": digest(manifest),
            "collector": "forward-boundary-v1", "monotonic_s": float(i + 1),
            "kind": "forward_completed", "execution_stage": "warmup", "success": True}
            for i, shape in enumerate(manifest["shapes"]) if shape["mode"] == mode]
        (directory / "observer.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        modes.append({"mode": mode, "metadata": ready})
    result = save_proof(tmp_path, manifest, modes)
    assert result["status"] == "PASS"
    return tmp_path, manifest, result


def rewrite_rows(root, change):
    path = root / "continuous/observer.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    change(rows)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    index_path = root / "shape-proof-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["sha256"]["continuous/observer.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index), encoding="utf-8")


def test_complete_proof_and_independent_replay(bundle):
    root, manifest, saved = bundle
    assert verify_bundle(root, expected_identity={k: manifest[k] for k in IDENTITY}) == saved
    assert saved["counts"] == dict(expected=4, observed=4, covered=4, missing=0, extra=0, unknown=0)


def test_missing_required_shape_fails(bundle):
    root, _, _ = bundle
    rewrite_rows(root, lambda rows: rows.pop())
    result = verify_bundle(root)
    assert "SHAPE_REQUIRED_COVERAGE" in result["reason_codes"]
    assert result["counts"]["missing"] == 1


def test_extra_shape_is_not_silently_ignored(bundle):
    root, _, _ = bundle
    def extra(rows):
        row = deepcopy(rows[0])
        row.update(observation_id="extra", input_shape=[2, 3], attention_mask_shape=[2, 3], batch_size=2)
        rows.append(row)
    rewrite_rows(root, extra)
    assert "SHAPE_UNDECLARED_OBSERVATIONS" in verify_bundle(root)["reason_codes"]


@pytest.mark.parametrize("key,value", [
    ("pid", 999), ("run_id", "different-run"), ("config_sha256", "c" * 64),
    ("model_revision", "c" * 40), ("git", {"sha": "d" * 40}),
    ("expected_manifest_sha256", "d" * 64), ("schema_version", 0),
    ("observation_id", None), ("dtype", "float16"), ("device", "cpu"),
])
def test_observer_identity_mismatch_fails(bundle, key, value):
    root, _, _ = bundle
    rewrite_rows(root, lambda rows: rows[0].update({key: value}))
    assert "SHAPE_OBSERVATION_IDENTITY" in verify_bundle(root)["reason_codes"]


def test_current_source_identity_mismatch(bundle):
    root, manifest, _ = bundle
    identity = {k: deepcopy(manifest[k]) for k in IDENTITY}
    identity["git"]["source_tree_sha256"] = "e" * 64
    assert verify_bundle(root, expected_identity=identity)["status"] == "FAIL"


@pytest.mark.parametrize("action", ["missing", "corrupt"])
def test_raw_evidence_integrity(bundle, action):
    root, _, _ = bundle
    path = root / "continuous/observer.jsonl"
    if action == "missing":
        path.unlink()
    else:
        path.write_text("{}", encoding="utf-8")
    assert "SHAPE_ARTIFACT_INTEGRITY" in verify_bundle(root)["reason_codes"]


def test_unobservable_required_dimension_never_passes(bundle):
    root, _, _ = bundle
    rewrite_rows(root, lambda rows: rows[0].update(kv_tensor_shapes_before=None))
    result = verify_bundle(root)
    assert "SHAPE_REQUIRED_DIMENSIONS" in result["reason_codes"]
    assert result["counts"]["unknown"] == 1


def test_forward_counts_without_shapes_cannot_pass(bundle):
    root, _, _ = bundle
    def counts_only(rows):
        rows[:] = [{"kind": "forward_count", "count": 100}]
    rewrite_rows(root, counts_only)
    assert verify_bundle(root)["status"] == "FAIL"


def test_old_artifact_cannot_establish_coverage(tmp_path):
    write_new(tmp_path / "shape-proof-index.json", {"status": "PASS", "forward_count": 100})
    assert verify_bundle(tmp_path)["status"] == "FAIL"


def test_continuous_interleavings_remain_explicit_unknown():
    config = load_config("benchmarks/runtime/online_http/configs/qwen_admission_v2.json")
    trace = [{**config.data["requests"][0], "observed_prompt_tokens": 37}]
    manifest = build_manifest(config, trace, {}, {"layers": 24, "kv_heads": 2, "head_dim": 64})
    assert manifest["unknown_obligations"][0]["id"] == "continuous_interleavings"
    assert manifest["unknown_obligations"][0]["required"] is True


def test_manifest_truncation_rejected_even_if_rehashed(bundle):
    root, _, _ = bundle
    path = root / "expected-shapes.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["shapes"].pop()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    index_path = root / "shape-proof-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["manifest_sha256"] = digest(manifest)
    index["sha256"]["expected-shapes.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index), encoding="utf-8")
    assert verify_bundle(root)["status"] == "FAIL"


def test_observation_outside_actual_warmup_window_fails(bundle):
    root, _, _ = bundle
    rewrite_rows(root, lambda rows: rows[0].update(monotonic_s=101.0))
    assert "SHAPE_OBSERVATION_IDENTITY" in verify_bundle(root)["reason_codes"]


def test_missing_nested_cache_dimension_is_unknown(bundle):
    root, _, _ = bundle
    rewrite_rows(root, lambda rows: rows[1].update(kv_tensor_shapes_before=[{"keys": None, "values": [1, 2, 3, 4]}]))
    assert "SHAPE_REQUIRED_DIMENSIONS" in verify_bundle(root)["reason_codes"]


def test_shape_verdict_is_not_trusted_without_raw_replay(bundle):
    root, _, original = bundle
    saved = json.loads((root / "shape-coverage.json").read_text(encoding="utf-8"))
    saved["counts"]["covered"] = 9999
    (root / "shape-coverage.json").write_text(json.dumps(saved), encoding="utf-8")
    assert verify_bundle(root) == original


def test_missing_warmup_marker_is_not_success(bundle):
    root, _, _ = bundle
    (root / "continuous/warmup-finished").unlink()
    assert verify_bundle(root)["status"] == "FAIL"
