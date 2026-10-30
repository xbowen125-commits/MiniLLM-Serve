"""Metric contracts independent of CUDA timing or performance expectations."""

import pytest
from benchmarks.runtime.gpu_batch.analysis import distribution, request_timing, execution_order, summarize


def test_metric_boundaries_and_single_token():
    assert request_timing(10, 12, 18, 4) == {"ttft_s": 2, "latency_s": 8, "tpot_s": 2}
    assert request_timing(10, 12, 12, 1)["tpot_s"] is None
    with pytest.raises(ValueError):
        request_timing(10, 9, 12, 2)


def test_order_is_balanced_and_retains_every_repeat():
    assert execution_order(2) == [(0, 1), (0, 2), (1, 2), (1, 1)]
    assert len(execution_order(10)) == 20


def test_summary_keeps_slow_samples_and_sample_stddev():
    d = distribution([1, 2, 9])
    assert d["count"] == 3 and d["mean"] == 4 and d["median"] == 2
    assert d["min"] == 1 and d["max"] == 9
    assert d["stddev"] == pytest.approx(4.3588989435)
    assert d["p95"] == pytest.approx(8.3)
    assert distribution([1])["stddev"] is None


def test_pooled_throughput_is_not_mean_of_rates():
    runs = [{"batch_size": 1, "latency_s": seconds, "generated_tokens": 16,
        "tokens_per_second": 16 / seconds, "requests": [dict(ttft_s=.1, latency_s=seconds, tpot_s=.1)]}
        for seconds in (1, 2)]
    s = summarize(runs)["1"]
    assert s["tokens_per_second"]["mean"] == 12
    assert s["pooled_tokens_per_second"] == pytest.approx(32 / 3)


def test_direct_forward_diagnostic_counter_without_module_hooks():
    from benchmarks.runtime.gpu_batch.benchmark import DiagnosticWorker
    from minillm.workers import LoadedModel
    from minillm.continuous import BatchExecutor
    from test_transformers_batch_worker import FakeCache, FakeModel, FakeTokenizer, context
    worker = DiagnosticWorker(LoadedModel(FakeModel(), FakeTokenizer()), cache_factory=lambda _: FakeCache())
    executor = BatchExecutor(worker)
    try:
        executor.add(context('a', 'short'))
        executor.step()
        executor.step()
        executor.step()
        assert len(worker.forwards) == 2
        assert worker.forwards[1]['input_shape'] == [1, 1]
    finally:
        executor.close()
        worker.close()
