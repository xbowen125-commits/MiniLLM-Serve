"""Bounded stability and scheduling probe for a running MiniLLM-Serve server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from benchmark_http import get_json, load_local_tokenizer, make_prompt, request_once
    from benchmark_utils import runtime_metadata, write_csv, write_json, write_markdown
except ModuleNotFoundError:  # imported as benchmarks.stress_http in unit tests
    from benchmarks.benchmark_http import get_json, load_local_tokenizer, make_prompt, request_once
    from benchmarks.benchmark_utils import runtime_metadata, write_csv, write_json, write_markdown


def disconnect_stream(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    timeout: float,
    request_index: int,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "stream": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status = 0
    request_id = None
    observed_content = False
    error = None
    error_code = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
        status = response.status
        request_id = response.headers.get("X-MiniLLM-Request-ID")
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                content = event.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    observed_content = True
                    break
        finally:
            response.close()
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            error_code = payload.get("error", {}).get("code") or payload.get("error", {}).get("type")
        except Exception:
            error_code = f"http_{status}"
        error = f"HTTPError: {status}"
    except Exception as exc:
        error_code = type(exc).__name__
        error = f"{type(exc).__name__}: {exc}"
    return {
        "kind": "client_disconnect",
        "request_index": request_index,
        "status": status,
        "request_id": request_id,
        "observed_content": observed_content,
        "client_closed_stream": True,
        "success": status == 200 and observed_content,
        "error_code": error_code,
        "e2e_ms": (time.perf_counter() - started) * 1000,
        "error": error,
    }


def scheduler_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int | float | None:
    first = before.get(key)
    last = after.get(key)
    if isinstance(first, (int, float)) and isinstance(last, (int, float)):
        return last - first
    return None


def count_error_codes(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count client-visible failures across every stress-test phase."""
    counts: dict[str, int] = {}
    for record in records:
        if not record["success"]:
            code = record.get("error_code") or f"http_{record.get('status', 0)}"
            counts[code] = counts.get(code, 0) + 1
    return counts


def wait_until_idle(base_url: str, timeout: float, cleanup_timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + cleanup_timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = get_json(f"{base_url.rstrip('/')}/scheduler/status", timeout)
        if latest.get("running_requests") == 0 and latest.get("queued_requests") == 0:
            return latest
        time.sleep(0.1)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded MiniLLM-Serve HTTP stability test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--disconnect-requests", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cleanup-timeout", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    if not 1 <= args.requests <= 200:
        parser.error("--requests must be between 1 and 200")
    if not 1 <= args.concurrency <= 32:
        parser.error("--concurrency must be between 1 and 32")
    if not 0 <= args.disconnect_requests <= 20:
        parser.error("--disconnect-requests must be between 0 and 20")
    if args.timeout <= 0 or args.cleanup_timeout <= 0:
        parser.error("timeouts must be positive")

    base_url = args.base_url.rstrip("/")
    health = get_json(f"{base_url}/health", args.timeout)
    before = get_json(f"{base_url}/scheduler/status", args.timeout)
    if health.get("model") != args.model:
        parser.error(f"server model is {health.get('model')!r}, not {args.model!r}")
    tokenizer = load_local_tokenizer(args.tokenizer or args.model, args.allow_model_download)
    prompt = make_prompt(tokenizer, args.prompt_tokens)
    token_count = (
        (lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        if tokenizer is not None
        else None
    )

    common = {
        "base_url": base_url,
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
        "seed": args.seed,
        "timeout": args.timeout,
        "token_count": token_count,
    }
    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(request_once, **common, request_index=index)
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            record = future.result()
            record["kind"] = "load"
            runs.append(record)
    load_wall_seconds = time.perf_counter() - started
    after_load = get_json(f"{base_url}/scheduler/status", args.timeout)

    disconnects = [
        disconnect_stream(
            base_url=base_url,
            model=args.model,
            prompt=prompt,
            max_tokens=max(args.max_tokens, 64),
            seed=args.seed,
            timeout=args.timeout,
            request_index=index,
        )
        for index in range(args.disconnect_requests)
    ]
    after_disconnects = get_json(f"{base_url}/scheduler/status", args.timeout)

    # Deterministic consecutive calls provide an HTTP-level cache-isolation probe.
    isolation_runs = [
        request_once(**{**common, "stream": False}, request_index=1000 + index)
        for index in range(2)
    ]
    cache_isolation_consistent = (
        all(run["success"] for run in isolation_runs)
        and isolation_runs[0]["output_sha256"] == isolation_runs[1]["output_sha256"]
    )
    after_isolation = get_json(f"{base_url}/scheduler/status", args.timeout)
    after = wait_until_idle(base_url, args.timeout, args.cleanup_timeout)
    idle_after = after.get("running_requests") == 0 and after.get("queued_requests") == 0

    successful = sum(bool(run["success"]) for run in runs)
    all_client_records = runs + disconnects + isolation_runs
    error_codes = count_error_codes(all_client_records)
    request_ids = [
        int(run["request_id"])
        for run in sorted(runs, key=lambda item: item["client_started_ns"])
        if run.get("request_id") is not None
    ]
    fifo_observed = request_ids == sorted(request_ids) if request_ids else None
    summary = {
        "requests": len(runs),
        "client_concurrency": args.concurrency,
        "server_gpu_concurrency": before.get("max_concurrent_requests"),
        "successful": successful,
        "failed": len(runs) - successful,
        "rejected": error_codes.get("queue_full", 0),
        "queue_timeouts": error_codes.get("queue_timeout", 0),
        "inference_timeouts": error_codes.get("inference_timeout", 0),
        "client_disconnect_attempts": len(disconnects),
        "client_disconnects": sum(bool(run["success"]) for run in disconnects),
        "disconnect_failures": sum(not bool(run["success"]) for run in disconnects),
        "error_rate": (len(runs) - successful) / len(runs),
        "all_phase_error_rate": (
            sum(not bool(run["success"]) for run in all_client_records) / len(all_client_records)
        ),
        "request_throughput_rps": successful / load_wall_seconds if load_wall_seconds else None,
        "error_codes": error_codes,
        "fifo_observed": fifo_observed,
        "cache_isolation_consistent": cache_isolation_consistent,
        "idle_after_test": idle_after,
        "scheduler_completed_delta": scheduler_delta(before, after, "completed_requests"),
        "scheduler_rejected_delta": scheduler_delta(before, after, "rejected_requests"),
        "scheduler_cancelled_delta": scheduler_delta(before, after, "cancelled_requests"),
        "scheduler_queue_timeout_delta": scheduler_delta(before, after, "queue_timeout_requests"),
        "scheduler_inference_timeout_delta": scheduler_delta(
            before, after, "inference_timeout_requests"
        ),
        "gpu_allocated_change_mb": (
            after.get("gpu", {}).get("memory_allocated_mb", 0)
            - before.get("gpu", {}).get("memory_allocated_mb", 0)
            if before.get("gpu", {}).get("available") and after.get("gpu", {}).get("available")
            else None
        ),
        "gpu_peak_mb": after.get("gpu", {}).get("max_memory_allocated_mb"),
    }
    metadata = runtime_metadata()
    metadata.update(
        {
            "timestamp": datetime.now().astimezone().isoformat(),
            "base_url": base_url,
            "model": args.model,
            "server_health": health,
            "max_requests_guard": 200,
            "max_concurrency_guard": 32,
            "timeout_seconds": args.timeout,
        }
    )
    payload = {
        "metadata": metadata,
        "summary": summary,
        "scheduler_before": before,
        "scheduler_after": after,
        "scheduler_after_load": after_load,
        "scheduler_after_disconnects": after_disconnects,
        "scheduler_after_isolation": after_isolation,
        "runs": sorted(runs, key=lambda item: item["request_index"]),
        "disconnect_runs": disconnects,
        "cache_isolation_runs": isolation_runs,
    }
    prefix = args.output_prefix or Path("benchmarks/results") / datetime.now().strftime(
        "stress-%Y%m%d-%H%M%S"
    )
    write_json(prefix.with_suffix(".json"), payload)
    write_csv(prefix.with_suffix(".csv"), runs + disconnects)
    write_markdown(
        prefix.with_suffix(".md"),
        title="MiniLLM-Serve Stability Test",
        metadata=metadata,
        rows=[summary],
        columns=list(summary),
        notes=[
            "This is a bounded test; hard guards cap requests at 200 and client concurrency at 32.",
            "FIFO observation compares server request IDs with client start order and is not a formal proof under OS scheduling jitter.",
            "A disconnect is counted when the client closes after the first content delta; scheduler counters confirm server cleanup.",
            "GPU peak is the server process lifetime peak exposed by PyTorch, not an isolated per-request peak.",
            "A true model-forward fault is covered by Mock ASGI tests because the production API has no fault-injection endpoint.",
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Reports: {prefix.with_suffix('.json')}, {prefix.with_suffix('.csv')}, {prefix.with_suffix('.md')}")
    if not idle_after:
        raise SystemExit("scheduler did not become idle before cleanup timeout")


if __name__ == "__main__":
    main()
