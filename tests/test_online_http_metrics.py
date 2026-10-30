"""No model/GPU: frozen client metric formulas and failure accounting."""

import math

import pytest

from benchmarks.runtime.online_http.metrics import calculate_request_metrics, summarize


SLO = {"ttft_ms": 1000, "tpot_ms": 500, "e2e_ms": 5000}


def sample(**changes):
    row = {
        "success": True, "outcome": "completed", "sent_at_s": 1.0,
        "first_content_at_s": 1.2, "last_content_at_s": 1.8,
        "terminal_at_s": 2.0, "visible_output_tokens": 4,
        "generated_output_tokens": None, "expected_visible_output_tokens": 4,
        "length_match": True,
    }
    return {**row, **changes}


def test_metrics_use_content_boundaries_not_termination_or_chunk_count():
    row = sample(content_chunks=99)
    result = calculate_request_metrics(row)
    assert result["ttft_ms"] == pytest.approx(200)
    assert result["tpot_ms"] == pytest.approx(200)
    assert result["e2e_ms"] == pytest.approx(1000)
    assert result["visible_output_tokens"] == 4
    assert "ttft_ms" not in row


def test_nonstream_does_not_invent_first_token_observation():
    result = calculate_request_metrics(sample(first_content_at_s=None, last_content_at_s=None))
    assert result["ttft_ms"] is None and result["tpot_ms"] is None
    assert result["e2e_ms"] == 1000
    summary = summarize([result], window_s=1, slo=SLO)
    assert summary["completed_requests"] == 1
    assert summary["slo_eligible_requests"] == 0


def test_single_token_has_no_tpot_but_can_meet_other_slo_criteria():
    row = sample(visible_output_tokens=1, expected_visible_output_tokens=1)
    assert calculate_request_metrics(row)["tpot_ms"] is None
    assert summarize([row], window_s=1, slo=SLO)["slo_eligible_requests"] == 1


def test_invalid_timestamps_retained_but_not_used_as_metrics():
    row = sample(first_content_at_s=0.5, last_content_at_s=math.inf)
    result = calculate_request_metrics(row)
    assert result["first_content_at_s"] == 0.5
    assert result["ttft_ms"] is None and result["tpot_ms"] is None
    assert len(result["metric_validation_errors"]) == 2
    assert summarize([row], window_s=1, slo=SLO)["slo_eligible_requests"] == 0


def test_unknown_actual_sse_usage_is_na_visible_proxy_is_separate():
    result = summarize([sample(), sample()], window_s=2, slo=SLO)
    assert result["completed_generated_output_tokens_per_s"] is None
    assert result["completed_visible_output_tokens_per_s"] == 4
    assert result["completed_requests_per_s"] == 1
    assert result["token_accounting"]["actual_count_unknown_completed_requests"] == 2
    known = summarize([sample(generated_output_tokens=5)] * 2, window_s=2, slo=SLO)
    assert known["completed_generated_output_tokens_per_s"] == 5
    mixed = summarize([sample(), sample(generated_output_tokens=5)], window_s=2, slo=SLO)
    assert mixed["completed_generated_output_tokens_per_s"] is None


def test_failed_timeout_cancelled_mismatch_and_not_sent_all_retained():
    rows = [sample(), sample(outcome="error", success=False), sample(outcome="timeout", success=False),
            sample(outcome="cancelled", success=False), sample(outcome="output_length_mismatch", length_match=False),
            sample(outcome="not_sent", success=False, sent_at_s=None, terminal_at_s=None)]
    result = summarize(rows, window_s=2, slo=SLO)
    assert result["planned_requests"] == 6 and result["attempted_requests"] == 5
    assert result["completed_requests"] == 1 and result["failed_requests"] == 5
    assert sum(result["outcome_counts"].values()) == 6
    assert result["rates"]["timeout_rate"] == pytest.approx(1 / 6)
    assert result["rates"]["failure_rate"] == pytest.approx(5 / 6)
    assert result["distributions"]["all_attempted_e2e_ms"]["count"] == 5
    assert result["distributions"]["completed_e2e_ms"]["count"] == 1


def test_percentiles_and_sample_standard_deviation():
    rows = [sample(sent_at_s=0, first_content_at_s=0.1, last_content_at_s=0.2, terminal_at_s=value)
            for value in (1, 2, 3, 4)]
    result = summarize(rows, window_s=4, slo=SLO)["distributions"]["completed_e2e_ms"]
    assert result["mean"] == 2500 and result["median"] == 2500
    assert result["p50"] == 2500 and result["p95"] == pytest.approx(3850)
    assert result["p99"] == pytest.approx(3970)
    assert result["sample_stddev"] == pytest.approx(math.sqrt(5 / 3) * 1000)
    assert result["min"] == 1000 and result["max"] == 4000


def test_configured_length_mismatch_excludes_request_from_completed_numerator():
    result = summarize([sample(visible_output_tokens=3)], window_s=1, slo=SLO)
    assert result["completed_requests"] == 0
    assert result["completed_visible_output_tokens_per_s"] == 0
    assert result["outcome_counts"] == {"output_length_mismatch": 1}
    assert result["rates"]["output_length_mismatch_rate"] == 1


@pytest.mark.parametrize("window", [0, -1, math.inf, math.nan])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError, match="window_s"):
        summarize([], window_s=window, slo=SLO)


def test_slo_cannot_invent_unregistered_metric_or_invalid_threshold():
    with pytest.raises(ValueError, match="slo"):
        summarize([], window_s=1, slo={"chunk_latency": 1})
    with pytest.raises(ValueError, match="threshold"):
        summarize([], window_s=1, slo={"ttft_ms": -1})
