"""Protocol-contract tests use no model, network, CUDA or third-party library."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from benchmarks.runtime.online_http.config import ProtocolConfig, load_config
from benchmarks.runtime.online_http.workload import build_trace, execution_order, trace_sha256


CONFIGS = Path(__file__).resolve().parents[1] / "benchmarks/runtime/online_http/configs"


def smoke():
    return load_config(CONFIGS / "smoke.json")


def test_canonical_config_hash_is_order_independent_and_immutable():
    config = smoke()
    data = config.as_dict()
    reordered = dict(reversed(list(data.items())))
    assert ProtocolConfig.from_dict(reordered).config_sha256 == config.config_sha256
    assert len(config.sha256) == 64
    data["requests"][0]["messages"][0]["content"] = "changed"
    assert config.data["requests"][0]["messages"][0]["content"] == "Say hello."
    with pytest.raises(FrozenInstanceError):
        config._canonical_json = "{}"


@pytest.mark.parametrize("field,value", [
    ("pairs", 0), ("pairs", 21), ("max_requests", 0),
    ("max_client_concurrency", 0), ("max_client_concurrency", 17),
    ("admission_limit", 3), ("request_timeout_s", float("inf")),
    ("request_timeout_s", -1), ("queue_capacity", -1),
    ("max_client_concurrency", True), ("stream", "true"),
    ("max_run_seconds", 10), ("telemetry_interval_s", 0),
    ("telemetry_interval_s", .001), ("poll_interval_s", .001),
    ("arrival_tolerance_ms", -1),
])
def test_protocol_rejects_unbounded_or_invalid_limits(field, value):
    data = smoke().as_dict()
    data[field] = value
    with pytest.raises(ValueError):
        ProtocolConfig.from_dict(data)


def test_slo_is_predeclared_and_changes_config_hash():
    data = smoke().as_dict()
    original = smoke().config_sha256
    data["slo"]["ttft_ms"] += 1
    assert ProtocolConfig.from_dict(data).config_sha256 != original
    del data["slo"]
    with pytest.raises(ValueError, match="missing"):
        ProtocolConfig.from_dict(data)


def test_trace_is_identical_fresh_copy_and_preserves_simultaneous_order():
    config = smoke()
    first = build_trace(config)
    second = build_trace(config)
    assert first == second and first is not second
    assert first[0]["arrival_s"] == first[1]["arrival_s"] == 0
    assert [row["request_id"] for row in first[:2]] == ["short-01", "long-01"]
    assert trace_sha256(first) == trace_sha256(second)
    first[0]["arrival_s"] = 1
    assert trace_sha256(first) != trace_sha256(second)
    assert build_trace(config) == second


def test_arrival_trace_rejects_duplicates_negative_or_unsorted_arrivals():
    original = smoke().as_dict()
    for mutate in (
        lambda rows: rows[1].update(request_id=rows[0]["request_id"]),
        lambda rows: rows[0].update(arrival_s=-1),
        lambda rows: rows[0].update(arrival_s=2),
    ):
        data = deepcopy(original)
        mutate(data["requests"])
        with pytest.raises(ValueError):
            ProtocolConfig.from_dict(data)


def test_ab_ba_pair_order_is_explicit():
    assert execution_order(3) == [
        (0, "legacy"), (0, "continuous"),
        (1, "continuous"), (1, "legacy"),
        (2, "legacy"), (2, "continuous"),
    ]
    with pytest.raises(ValueError):
        execution_order(0)


def test_real_config_requires_full_pinned_revisions_and_explicit_device():
    config = load_config(CONFIGS / "qwen_protocol.json")
    assert config.data["device"] == "cuda"
    assert config.data["model_revision"] == config.data["tokenizer_revision"]
    assert config.data["pairs"] == 10
    for field, value in (("model_revision", "main"),
                         ("tokenizer_revision", "latest"), ("device", "auto")):
        data = config.as_dict()
        data[field] = value
        with pytest.raises(ValueError):
            ProtocolConfig.from_dict(data)


def test_protocol_hash_covers_arrival_generation_and_admission():
    config = smoke()
    for field, value in (("temperature", 0.5), ("seed", 999),
                         ("admission_limit", 1), ("queue_capacity", 7)):
        data = config.as_dict()
        data[field] = value
        assert ProtocolConfig.from_dict(data).config_sha256 != config.config_sha256
    data = config.as_dict()
    data["requests"][-1]["arrival_s"] += 0.01
    assert ProtocolConfig.from_dict(data).config_sha256 != config.config_sha256


def test_expected_token_lengths_are_optional_not_invented():
    config = load_config(CONFIGS / "qwen_protocol.json")
    assert all("expected_prompt_tokens" not in row for row in build_trace(config))
    assert all("expected_visible_output_tokens" not in row for row in build_trace(config))
    assert json.loads(json.dumps(config.as_dict())) == config.as_dict()


def test_unknown_config_fields_are_rejected_instead_of_silently_ignored():
    data = smoke().as_dict()
    data["typo_slo"] = {}
    with pytest.raises(ValueError, match="unknown"):
        ProtocolConfig.from_dict(data)
