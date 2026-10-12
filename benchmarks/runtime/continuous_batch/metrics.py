"""Virtual-clock metrics, deliberately separate from runtime wall-clock data."""

from statistics import mean


def request_metrics(arrival, started, first_token, finished, generated_tokens):
    """TTFT and total latency include queue delay and are arrival-relative."""
    if not arrival <= started <= first_token <= finished or generated_tokens < 1:
        raise ValueError("invalid request observations")
    return {"queue_delay_s": started - arrival, "ttft_s": first_token - arrival,
            "total_latency_s": finished - arrival, "generated_tokens": generated_tokens}


def summarize(rows, elapsed, decode_batch_sizes):
    if not rows or elapsed <= 0:
        raise ValueError("summary needs completed requests and positive elapsed time")
    return {"completed_requests": len(rows),
            "average_ttft_s": mean(r["ttft_s"] for r in rows),
            "average_queue_delay_s": mean(r["queue_delay_s"] for r in rows),
            "average_total_latency_s": mean(r["total_latency_s"] for r in rows),
            "generated_tokens": sum(r["generated_tokens"] for r in rows),
            "simulated_duration_s": elapsed,
            "simulated_tokens_per_second": sum(r["generated_tokens"] for r in rows) / elapsed,
            "simulated_requests_per_second": len(rows) / elapsed,
            "average_active_decode_batch_size": mean(decode_batch_sizes),
            "max_active_decode_batch_size": max(decode_batch_sizes)}
