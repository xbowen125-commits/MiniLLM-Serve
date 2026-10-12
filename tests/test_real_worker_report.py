"""Real-worker report parsing and arithmetic independent of model execution."""

import json

import pytest

from benchmarks.runtime.real_model.report_generator import parse_benchmark, aggregate_metrics, generate_report


def data():
    return json.loads(json.dumps({
        "schema_version": 1, "scope": "real-transformers-worker", "model_name": "test-model",
        "device": "cpu", "dtype": "float32", "timestamp": "2026-10-12T00:00:00+00:00",
        "configuration": {"repeats": 2, "max_tokens": 4, "warmups": 1},
        "runs": [
            {"run_index": i, "prompt_length": 3, "generated_tokens": 4,
             "metrics": {"ttft_ms": 10 + 10 * i, "decode_latency_ms": 30 + 10 * i,
                         "total_latency_ms": 40 + 20 * i, "generated_tokens": 4,
                         "tokens_per_second": 100 + 20 * i}, "finish_reason": "length"}
            for i in range(2)
        ],
    }))


def test_json_parsing():
    parsed = parse_benchmark(data())
    assert parsed["device"] == "cpu"
    assert len(parsed["runs"]) == 2


def test_aggregation_is_arithmetic_mean():
    means = aggregate_metrics(data())
    assert means["ttft_ms"] == {"average": 15, "samples": 2}
    assert means["decode_latency_ms"]["average"] == 35
    assert means["total_latency_ms"]["average"] == 50
    assert means["tokens_per_second"]["average"] == 110


def test_markdown_contains_setup_means_runs_and_limits():
    report = generate_report(data())
    for heading in ("Experiment Setup", "Metrics", "Run Summary", "Limitations"):
        assert heading in report
    assert "| TTFT | 15" in report
    assert "| 0 | 3 | 4 |" in report
    assert "| 1 | 3 | 4 |" in report
    assert "single-request" in report
    assert "production request throughput" in report


def test_missing_observation_is_not_zero():
    payload = data()
    payload["runs"][0]["metrics"]["ttft_ms"] = None
    assert aggregate_metrics(payload)["ttft_ms"] == {"average": 20, "samples": 1}
    assert "N/A" in generate_report(payload)


def test_invalid_artifact_is_rejected():
    payload = data()
    payload["configuration"]["repeats"] = 3
    with pytest.raises(ValueError, match="repeats"):
        parse_benchmark(payload)
    payload = data()
    payload["runs"][0]["metrics"]["ttft_ms"] = float("nan")
    with pytest.raises(ValueError, match="metric"):
        parse_benchmark(payload)
