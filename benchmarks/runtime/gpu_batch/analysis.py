"""Transparent statistics over every saved run; no outlier removal."""

import math
from statistics import mean, median, stdev


def distribution(values):
    values = list(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "stddev": None,
                "min": None, "max": None, "p95": None}
    if any(not math.isfinite(v) for v in values):
        raise ValueError("non-finite measurement")
    ordered = sorted(values)
    position = (len(values) - 1) * .95
    lo = int(position)
    p95 = ordered[lo] + (ordered[min(lo + 1, len(values) - 1)] - ordered[lo]) * (position - lo)
    return {"count": len(values), "mean": mean(values), "median": median(values),
            "stddev": stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values), "p95": p95}


def request_timing(start, first, finish, tokens):
    if tokens < 1 or not start <= first <= finish:
        raise ValueError("invalid token/timestamp observations")
    return {"ttft_s": first - start, "latency_s": finish - start,
            "tpot_s": (finish - first) / (tokens - 1) if tokens > 1 else None}


def execution_order(repeats):
    return [(repeat, size) for repeat in range(repeats)
            for size in ((1, 2) if repeat % 2 == 0 else (2, 1))]


def summarize(runs):
    summary = {}
    for size in (1, 2):
        selected = [r for r in runs if r["batch_size"] == size]
        requests = [q for r in selected for q in r["requests"]]
        summary[str(size)] = {
            "run_count": len(selected),
            "tokens_per_second": distribution(r["tokens_per_second"] for r in selected),
            "batch_latency_s": distribution(r["latency_s"] for r in selected),
            "request_ttft_s": distribution(q["ttft_s"] for q in requests),
            "request_tpot_s": distribution(q["tpot_s"] for q in requests if q["tpot_s"] is not None),
            "request_latency_s": distribution(q["latency_s"] for q in requests),
            "pooled_tokens_per_second": (sum(r["generated_tokens"] for r in selected) /
                                         sum(r["latency_s"] for r in selected)) if selected else None,
        }
    return summary


def markdown(result):
    lines = ["# GPU Batch Validity Run", "", "## Configuration and provenance", "",
             "```json", __import__('json').dumps({k: v for k, v in result.items()
                 if k not in {"runs", "warmup_runs", "summary"}}, indent=2), "```", "",
             "## Every measured run", "",
             "| Order | Batch | Tokens | TTFT mean (s) | Latency (s) | Tokens/s | Forward calls | Peak bytes |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for i, r in enumerate(result["runs"]):
        lines.append(f"| {i} | {r['batch_size']} | {r['generated_tokens']} | "
            f"{mean(q['ttft_s'] for q in r['requests']):.6f} | {r['latency_s']:.6f} | "
            f"{r['tokens_per_second']:.6f} | {r.get('forward_call_count', 'N/A')} | {r['gpu_memory_bytes']['peak_allocated']} |")
    lines.extend(["", "## Distributions (all samples retained)", "",
        "Request distributions pool correlated requests from the same batch, not independent repetitions.",
        "Sample stddev uses n-1; singletons report null. P95 uses linear interpolation.", "",
        "```json", __import__('json').dumps(result["summary"], indent=2), "```", "",
        "## Limitations", "",
        "Synchronized wall-clock runtime timings include Python, KV copying and synchronization; CUDA events (optional)",
        "cover only each forward call's device timeline, not pure kernel time. No outliers are removed.",
        "Identical prompts/precision do not fix GPU clocks, laptop power/thermal behavior or competing workloads.",
        "Telemetry snapshots are not continuous profiling and cannot prove causality. No stable speedup is assumed.", ""])
    return "\n".join(lines)
