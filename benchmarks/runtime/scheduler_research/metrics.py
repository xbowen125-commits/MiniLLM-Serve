"""Virtual admission latency and length-normalized fairness summaries."""

from statistics import mean
from ..metrics import queue_percentile


def fairness_score(rows) -> float:
    """Jain index on inverse slowdown: service_time / total_latency.

    In [1/n, 1]; equal slowdown scores 1 even if all requests wait excessively.
    This is a diagnostic, not proof of starvation freedom or QoS.
    """
    if not rows:
        raise ValueError("fairness needs completed request observations")
    values = [r["service_seconds"] / r["total_latency"] for r in rows]
    return sum(values) ** 2 / (len(values) * sum(v * v for v in values))


def summarize(rows, starvation_threshold):
    waits = [r["queue_delay"] for r in rows]
    if not waits:
        raise ValueError("metrics need completed requests")
    return {"completed_requests": len(rows), "average_queue_delay": mean(waits),
            "p50_wait_time": queue_percentile(waits, 50),
            "p95_wait_time": queue_percentile(waits, 95),
            "p99_wait_time": queue_percentile(waits, 99),
            "average_ttft": mean(r["ttft"] for r in rows),
            "average_total_latency": mean(r["total_latency"] for r in rows),
            "starvation_count": sum(w > starvation_threshold for w in waits),
            "fairness_score": fairness_score(rows)}
