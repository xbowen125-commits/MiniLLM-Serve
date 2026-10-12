"""Deterministic tests for internal decode metrics."""

import pytest

from minillm.batching import DecodeMetrics, DecodeState, FinishReason


def test_decode_metrics_calculates_ttft_total_latency_and_tpot():
    state = DecodeState(request_id="request", max_tokens=4, created_time=9.0)
    state.start(now=10.0)
    state.record_step()
    state.append_token(1, now=10.1)
    state.record_step()
    state.append_token(2, now=10.3)
    state.finish(FinishReason.STOP, now=10.5)

    metrics = DecodeMetrics.from_state(state)

    assert metrics.generated_tokens == 2
    assert metrics.decode_steps == 2
    assert metrics.first_token_latency_ms == pytest.approx(100.0)
    assert metrics.total_latency_ms == pytest.approx(500.0)
    assert metrics.tpot_ms == pytest.approx(200.0)


def test_decode_metrics_keep_unavailable_latencies_explicit():
    state = DecodeState(request_id="request", max_tokens=2)

    metrics = DecodeMetrics.from_state(state)

    assert metrics.first_token_latency_ms is None
    assert metrics.total_latency_ms is None
    assert metrics.tpot_ms is None
    assert metrics.generated_tokens == 0
    assert metrics.decode_steps == 0
