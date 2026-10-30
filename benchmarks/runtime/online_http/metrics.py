"""Frozen client-observed metrics for the online HTTP comparison protocol.

SSE content events are not tokens.  Visible output token counts must come from
retokenizing the complete returned text with the same pinned tokenizer in both
modes.  They are deliberately separate from actual generated token counts,
which require authoritative server usage and can be unavailable for SSE.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from benchmarks.benchmark_utils import distribution


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def calculate_request_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with client timing metrics; never infer tokens from chunks.

    All timestamps are seconds from the same phase-local monotonic origin.
    TTFT is first nonempty content arrival minus the recorded send start.  TPOT
    is ``(last_content - first_content) / (visible_tokens - 1)`` for N > 1;
    termination, finish-reason and [DONE] events do not participate.  E2E ends at
    the received terminal event (or the observed failure/timeout/cancellation).
    Nonstream responses have no observable first-token timestamp and therefore
    TTFT/TPOT remain None.  Invalid timing observations are preserved in the row
    but excluded from derived metrics and listed in metric_validation_errors.
    """
    result = dict(row)
    errors: list[str] = []
    times: dict[str, float | None] = {}
    for name in ("sent_at_s", "first_content_at_s", "last_content_at_s", "terminal_at_s"):
        raw = row.get(name)
        times[name] = _number(raw)
        if raw is not None and times[name] is None:
            errors.append(f"{name} must be a finite nonnegative number")
    sent = times["sent_at_s"]
    first = times["first_content_at_s"]
    last = times["last_content_at_s"]
    terminal = times["terminal_at_s"]
    if sent is not None:
        for name in ("first_content_at_s", "last_content_at_s", "terminal_at_s"):
            if times[name] is not None and times[name] < sent:
                errors.append(f"{name} precedes sent_at_s")
                times[name] = None
    first = times["first_content_at_s"]
    last = times["last_content_at_s"]
    terminal = times["terminal_at_s"]
    if first is not None and last is not None and last < first:
        errors.append("last_content_at_s precedes first_content_at_s")
        last = None
    if terminal is not None:
        if first is not None and first > terminal:
            errors.append("first_content_at_s follows terminal_at_s")
            first = None
        if last is not None and last > terminal:
            errors.append("last_content_at_s follows terminal_at_s")
            last = None
    counts = {}
    for name in ("visible_output_tokens", "generated_output_tokens", "expected_visible_output_tokens"):
        counts[name] = _token_count(row.get(name))
        if row.get(name) is not None and counts[name] is None:
            errors.append(f"{name} must be a nonnegative integer")
    visible = counts["visible_output_tokens"]
    expected = counts["expected_visible_output_tokens"]
    result["ttft_ms"] = (first - sent) * 1000 if first is not None and sent is not None else None
    result["e2e_ms"] = (terminal - sent) * 1000 if terminal is not None and sent is not None else None
    result["tpot_ms"] = (
        (last - first) * 1000 / (visible - 1)
        if first is not None and last is not None and visible is not None and visible > 1
        else None
    )
    result["length_match"] = (
        visible == expected if visible is not None and expected is not None else row.get("length_match")
    )
    if result.get("outcome") == "completed" and result["length_match"] is False:
        # Wire completion is still preserved by success=True; the experiment
        # outcome must not hide a configured output-length mismatch.
        result["outcome"] = "output_length_mismatch"
    result["metric_validation_errors"] = errors
    return result


def _distribution(values: Sequence[float | None]) -> dict[str, Any]:
    """Reuse shared linear percentiles; sample SD needs at least two samples."""
    result = dict(distribution(values))
    samples = [value for value in values if value is not None]
    result["median"] = result["p50"]
    result["sample_stddev"] = statistics.stdev(samples) if len(samples) >= 2 else None
    return result


def _slo_qualified(row: Mapping[str, Any], slo: Mapping[str, float]) -> bool:
    for metric, threshold in slo.items():
        observed = row.get(metric)
        if metric == "tpot_ms" and _token_count(row.get("visible_output_tokens")) in (0, 1):
            # There is no subsequent-token interval for these outputs.  Other
            # configured criteria (including TTFT) still have to be observable.
            continue
        if observed is None or observed > threshold:
            return False
    return not row.get("metric_validation_errors")


def summarize(
    rows: Sequence[Mapping[str, Any]], *, window_s: float, slo: Mapping[str, float]
) -> dict[str, Any]:
    """Summarize one mode/phase, retaining every planned outcome in denominators.

    ``window_s`` is the formal replay window from its monotonic origin until
    the last client terminal observation, including scheduled arrival gaps; it
    excludes server/model startup, warmup and post-request resource drain.
    SLO thresholds must be frozen in the experiment config before execution.
    Every configured criterion must be observed and met; TPOT is not applicable
    for at most one visible output token.  Missing SSE usage never becomes a
    guessed actual-generation throughput.  The separately labelled visible
    text metric is a common client-observable proxy, not GPU token throughput.
    """
    if _number(window_s) is None or window_s <= 0:
        raise ValueError("window_s must be finite and greater than zero")
    if not slo or set(slo) - {"ttft_ms", "tpot_ms", "e2e_ms"}:
        raise ValueError("slo must contain only frozen ttft_ms, tpot_ms, e2e_ms thresholds")
    if any(_number(value) is None for value in slo.values()):
        raise ValueError("SLO thresholds must be finite and nonnegative")
    calculated = [calculate_request_metrics(row) for row in rows]
    outcomes = Counter(str(row.get("outcome", "error")) for row in calculated)
    attempted = [row for row in calculated if row.get("outcome") != "not_sent"]
    completed = [
        row for row in calculated
        if row.get("outcome") == "completed" and row.get("success") is True
        and row.get("length_match") is not False
    ]
    qualifying = [row for row in completed if _slo_qualified(row, slo)]
    generated = [_token_count(row.get("generated_output_tokens")) for row in completed]
    visible = [_token_count(row.get("visible_output_tokens")) for row in completed]
    actual_total = sum(generated) if all(value is not None for value in generated) else None
    visible_total = sum(visible) if all(value is not None for value in visible) else None
    denominator = len(calculated)
    rate = lambda count: count / denominator if denominator else None
    return {
        "planned_requests": denominator,
        "attempted_requests": len(attempted),
        "completed_requests": len(completed),
        "failed_requests": denominator - len(completed),
        "outcome_counts": dict(sorted(outcomes.items())),
        "window_s": window_s,
        "completed_requests_per_s": len(completed) / window_s,
        "completed_generated_output_tokens": actual_total,
        "completed_generated_output_tokens_per_s": actual_total / window_s if actual_total is not None else None,
        "completed_visible_output_tokens": visible_total,
        "completed_visible_output_tokens_per_s": visible_total / window_s if visible_total is not None else None,
        "slo_thresholds_ms": dict(slo),
        "slo_eligible_requests": len(qualifying),
        "slo_goodput_requests_per_s": len(qualifying) / window_s,
        "rates": {
            "failure_rate": rate(denominator - len(completed)),
            "error_rate": rate(outcomes["error"]),
            "timeout_rate": rate(outcomes["timeout"]),
            "cancellation_rate": rate(outcomes["cancelled"]),
            "output_length_mismatch_rate": rate(outcomes["output_length_mismatch"]),
            "not_sent_rate": rate(outcomes["not_sent"]),
        },
        "distributions": {
            "all_attempted_e2e_ms": _distribution([row["e2e_ms"] for row in attempted]),
            "completed_ttft_ms": _distribution([row["ttft_ms"] for row in completed]),
            "completed_tpot_ms": _distribution([row["tpot_ms"] for row in completed]),
            "completed_e2e_ms": _distribution([row["e2e_ms"] for row in completed]),
        },
        "token_accounting": {
            "generated_output_tokens": "authoritative server usage; unavailable SSE usage stays N/A",
            "visible_output_tokens": "complete visible response text retokenized using the common pinned tokenizer; never SSE chunk count",
            "actual_count_unknown_completed_requests": sum(value is None for value in generated),
            "visible_count_unknown_completed_requests": sum(value is None for value in visible),
            "requests_with_configured_visible_length_expectation": sum(row.get("expected_visible_output_tokens") is not None for row in calculated),
            "requests_without_visible_length_expectation": sum(row.get("expected_visible_output_tokens") is None for row in calculated),
        },
        "metric_validation_error_requests": sum(bool(row["metric_validation_errors"]) for row in calculated),
    }
