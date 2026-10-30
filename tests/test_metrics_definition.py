"""Exact independent metric formulas, including distinct batching averages."""

import pytest
from benchmarks.runtime.orca.metrics import summarize


def test_metric_formulas():
    rows = [{"request_id": "a", "generated_tokens": 2, "ttft_s": 3, "total_latency_s": 5},
            {"request_id": "b", "generated_tokens": 1, "ttft_s": 4, "total_latency_s": 4}]
    phases = [{"start_s": 0, "end_s": 1, "batch_size": 2},
              {"start_s": 1, "end_s": 4, "batch_size": 1}]
    iterations = [{"decode_request_ids": ("a", "b"), "prefill_request_ids": ()},
                  {"decode_request_ids": ("a",), "prefill_request_ids": ()}]
    m = summarize(rows, phases, iterations, 4, 5)
    assert m["average_ttft_s"] == 3.5
    assert m["average_total_latency_s"] == 4.5
    assert m["average_batch_size"] == 1.5
    assert m["batch_utilization"] == pytest.approx(5 / 16)
    assert m["simulated_tokens_per_second"] == .6
    assert m["iteration_count"] == 2
    assert m["decode_fairness"] == 1
