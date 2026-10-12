"""Format tests use FakeWorker; real model benchmark is explicitly slow."""

import json
import os

import pytest
import torch

from benchmarks.runtime.real_model.benchmark_real_worker import BenchmarkConfig, run_benchmark
from benchmarks.runtime.real_model.metrics import calculate_metrics, cache_storage_bytes
from benchmarks.runtime.runner import FakeWorker


def test_mock_benchmark_json_format_and_repeat_count():
    result = run_benchmark(BenchmarkConfig(model_name="mock", max_tokens=2, warmups=1, repeats=2),
                           worker=FakeWorker())
    encoded = json.loads(json.dumps(result, allow_nan=False))
    assert encoded["scope"] == "mock-worker-format-validation"
    assert encoded["model_name"] == "mock"
    assert encoded["device"] == "cpu"
    assert encoded["timestamp"]
    assert len(encoded["runs"]) == 2
    for run in encoded["runs"]:
        assert run["generated_tokens"] == 2
        assert run["prompt_length"] > 0
        assert run["metrics"]["ttft_ms"] >= 0
        assert run["metrics"]["decode_latency_ms"] >= 0
        assert run["metrics"]["total_latency_ms"] >= run["metrics"]["decode_latency_ms"]
        assert run["metrics"]["tokens_per_second"] > 0
        assert all(value is None for value in run["cuda_memory"].values())
    assert "prompt" not in encoded["configuration"]


def test_timing_formulas():
    metrics = calculate_metrics(start=1, first_token=1.2, end=2,
                                decode_seconds=0.7, generated_tokens=4)
    assert metrics.ttft_ms == pytest.approx(200)
    assert metrics.decode_latency_ms == pytest.approx(700)
    assert metrics.total_latency_ms == 1000
    assert metrics.tokens_per_second == 4


def test_cache_accounting_deduplicates_tensor_storages():
    from types import SimpleNamespace
    tensor = torch.zeros(4, dtype=torch.float32)
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=tensor, values=tensor[:2])])
    assert cache_storage_bytes(cache) == 16
    assert cache_storage_bytes(object()) is None


@pytest.mark.slow
def test_local_real_worker_benchmark_smoke():
    config = BenchmarkConfig(
        model_name=os.environ.get("MINILLM_RUNTIME_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        device=os.environ.get("MINILLM_RUNTIME_TEST_DEVICE", "cpu"),
        max_tokens=4, warmups=0, repeats=1,
    )
    result = run_benchmark(config)
    assert result["scope"] == "real-transformers-worker"
    assert 1 <= result["runs"][0]["generated_tokens"] <= 4
