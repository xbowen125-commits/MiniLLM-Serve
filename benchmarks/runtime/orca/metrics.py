"""Virtual timing, busy-phase utilization and eligible-decode fairness."""

from statistics import mean


def decode_fairness(service_counts: dict[str, int], eligible_counts: dict[str, int]) -> float:
    """Jain index on tokens / eligible decode iterations, excluding wait time.

    Equal treatment of eligible active requests scores 1, even if admission is
    unfair. Raw token totals cannot measure fairness across heterogeneous lengths.
    """
    if set(service_counts) != set(eligible_counts) or not service_counts:
        raise ValueError("fairness requires matching nonempty request sets")
    if any(type(v) is not int or v < 0 for v in service_counts.values()) or any(
        type(v) is not int or v < 1 for v in eligible_counts.values()):
        raise ValueError("invalid service/eligibility counts")
    if any(service_counts[r] > eligible_counts[r] for r in service_counts):
        raise ValueError("at most one token per eligible iteration")
    rates = [service_counts[r] / eligible_counts[r] for r in service_counts]
    denominator = len(rates) * sum(v * v for v in rates)
    return sum(rates) ** 2 / denominator if denominator else 0.0


def summarize(rows, phase_trace, iteration_trace, capacity, elapsed):
    """Capacity fraction weighted by virtual phase duration; idle gaps excluded."""
    busy = sum(p["end_s"] - p["start_s"] for p in phase_trace)
    used = sum(p["batch_size"] * (p["end_s"] - p["start_s"]) for p in phase_trace)
    eligible = {r["request_id"]: 0 for r in rows}
    for iteration in iteration_trace:
        for request_id in iteration["decode_request_ids"]:
            eligible[request_id] += 1
    served = {r["request_id"]: r["generated_tokens"] for r in rows}
    return {"completed_requests": len(rows), "generated_tokens": sum(served.values()),
        "average_ttft_s": mean(r["ttft_s"] for r in rows),
        "average_total_latency_s": mean(r["total_latency_s"] for r in rows),
        "simulated_duration_s": elapsed,
        "simulated_tokens_per_second": sum(served.values()) / elapsed,
        "simulated_requests_per_second": len(rows) / elapsed,
        "batch_utilization": used / (capacity * busy),
        "average_batch_size": mean(len(t["decode_request_ids"]) + len(t["prefill_request_ids"])
                                   for t in iteration_trace),
        "iteration_count": len(iteration_trace),
        "decode_fairness": decode_fairness(served, eligible),
        "decode_service_counts": served, "eligible_decode_iterations": eligible}
