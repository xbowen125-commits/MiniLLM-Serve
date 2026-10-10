"""Bounded HTTP benchmark for MiniLLM-Serve.

Client-observed metrics are intentionally separate from engine metrics.  SSE
chunks are reconstructed into text and tokenized; chunk count is never treated
as token count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    from benchmark_utils import distribution, runtime_metadata, write_csv, write_json, write_markdown
except ModuleNotFoundError:  # imported as benchmarks.benchmark_http in unit tests
    from benchmarks.benchmark_utils import (
        distribution,
        runtime_metadata,
        write_csv,
        write_json,
        write_markdown,
    )


def parse_positive_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_local_tokenizer(model: str, allow_download: bool):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, local_files_only=not allow_download)
    except Exception as exc:
        print(f"Tokenizer unavailable; streaming token metrics will be N/A: {exc}")
        return None


def make_prompt(tokenizer, target_tokens: int) -> str:
    seed = "HTTP serving benchmark prompt. 本测试用于测量并发推理与排队延迟。 "
    text = seed * max(2, target_tokens)
    if tokenizer is None:
        return " ".join(text.split()[:target_tokens])
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _error_code(payload: Any) -> str | None:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return error.get("code") or error.get("type")
    return None


def request_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    seed: int,
    timeout: float,
    token_count: Callable[[str], int] | None,
    request_index: int,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": seed,
            "stream": stream,
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
    client_started_ns = time.perf_counter_ns()
    first_text_at: float | None = None
    text_parts: list[str] = []
    status = 0
    response_payload: Any = None
    response_headers: dict[str, str] = {}
    finish_reason = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            if stream:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if "error" in event:
                        response_payload = event
                        continue
                    choice = event["choices"][0]
                    content = choice.get("delta", {}).get("content")
                    if content:
                        if first_text_at is None:
                            first_text_at = time.perf_counter()
                        text_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
            else:
                response_payload = json.loads(response.read().decode("utf-8"))
                choice = response_payload["choices"][0]
                text_parts.append(choice["message"]["content"])
                finish_reason = choice.get("finish_reason")
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = {key.lower(): value for key, value in exc.headers.items()}
        try:
            response_payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            response_payload = {"error": {"type": "invalid_http_error", "message": str(exc)}}
    except Exception as exc:
        response_payload = {"error": {"type": type(exc).__name__, "message": str(exc)}}

    finished = time.perf_counter()
    text = "".join(text_parts)
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    if not stream and isinstance(response_payload, dict):
        usage = response_payload.get("usage")
        if isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")
            prompt_tokens = usage.get("prompt_tokens")
    elif stream and token_count is not None and status == 200:
        completion_tokens = token_count(text)

    e2e_ms = (finished - started) * 1000
    ttft_ms = (first_text_at - started) * 1000 if first_text_at is not None else None
    tpot_ms = None
    if ttft_ms is not None and completion_tokens is not None and completion_tokens > 1:
        tpot_ms = max(0.0, e2e_ms - ttft_ms) / (completion_tokens - 1)
    tokens_per_second = (
        completion_tokens / (e2e_ms / 1000)
        if completion_tokens is not None and e2e_ms > 0
        else None
    )
    queue_wait = response_headers.get("x-minillm-queue-wait-ms")
    return {
        "request_index": request_index,
        "client_started_ns": client_started_ns,
        "status": status,
        "success": status == 200 and _error_code(response_payload) is None,
        "error_code": _error_code(response_payload),
        "request_id": response_headers.get("x-minillm-request-id"),
        "queue_wait_ms": float(queue_wait) if queue_wait is not None else None,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "tokens_per_second": tokens_per_second,
        "finish_reason": finish_reason,
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
    }


def summarize_case(
    runs: list[dict[str, Any]],
    *,
    wall_seconds: float,
    stream: bool,
    concurrency: int,
    target_prompt_tokens: int,
    max_tokens: int,
    kv_cache_state: str,
) -> dict[str, Any]:
    successful = [run for run in runs if run["success"]]
    error_codes: dict[str, int] = {}
    for run in runs:
        if not run["success"]:
            key = run["error_code"] or f"http_{run['status']}"
            error_codes[key] = error_codes.get(key, 0) + 1
    e2e = distribution(run["e2e_ms"] for run in successful)
    ttft = distribution(run["ttft_ms"] for run in successful)
    tpot = distribution(run["tpot_ms"] for run in successful)
    queue_wait = distribution(run["queue_wait_ms"] for run in successful)
    token_rate = distribution(run["tokens_per_second"] for run in successful)
    return {
        "stream": stream,
        "kv_cache": kv_cache_state,
        "target_prompt_tokens": target_prompt_tokens,
        "max_tokens": max_tokens,
        "client_concurrency": concurrency,
        "requests": len(runs),
        "successful": len(successful),
        "failed": len(runs) - len(successful),
        "error_rate": (len(runs) - len(successful)) / len(runs) if runs else 0.0,
        "error_codes": error_codes,
        "request_throughput_rps": len(successful) / wall_seconds if wall_seconds > 0 else None,
        "e2e_p50_ms": e2e["p50"],
        "e2e_p95_ms": e2e["p95"],
        "e2e_p99_ms": e2e["p99"],
        "ttft_p50_ms": ttft["p50"],
        "ttft_p95_ms": ttft["p95"],
        "tpot_p50_ms": tpot["p50"],
        "tokens_per_second_p50": token_rate["p50"],
        "queue_wait_p50_ms": queue_wait["p50"],
        "queue_wait_p95_ms": queue_wait["p95"],
        "wall_seconds": wall_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM-Serve bounded HTTP benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer", help="Tokenizer name/path; defaults to --model")
    parser.add_argument("--prompt-lengths", type=parse_positive_ints, default=parse_positive_ints("32,256"))
    parser.add_argument("--generation-lengths", type=parse_positive_ints, default=parse_positive_ints("16,64"))
    parser.add_argument("--concurrency", type=parse_positive_ints, default=parse_positive_ints("1,2,4,8"))
    parser.add_argument("--mode", choices=("stream", "non-stream", "both"), default="both")
    parser.add_argument("--kv-cache-state", choices=("on", "off", "unknown"), default="unknown")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-total-requests", type=int, default=500)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.timeout <= 0:
        parser.error("warmup >= 0, repeats >= 1 and timeout > 0 are required")
    if max(args.concurrency) > 64:
        parser.error("client concurrency is capped at 64")

    modes = [True, False] if args.mode == "both" else [args.mode == "stream"]
    cases = len(args.prompt_lengths) * len(args.generation_lengths) * len(args.concurrency) * len(modes)
    planned_requests = cases * (args.warmup + args.repeats)
    if planned_requests > args.max_total_requests:
        parser.error(
            f"planned requests ({planned_requests}) exceed --max-total-requests ({args.max_total_requests})"
        )

    base_url = args.base_url.rstrip("/")
    health = get_json(f"{base_url}/health", args.timeout)
    scheduler_before = get_json(f"{base_url}/scheduler/status", args.timeout)
    if health.get("model") != args.model:
        parser.error(f"server model is {health.get('model')!r}, not {args.model!r}")
    actual_kv_state = "on" if health.get("use_kv_cache") else "off"
    if args.kv_cache_state == "unknown":
        args.kv_cache_state = actual_kv_state
    elif args.kv_cache_state != actual_kv_state:
        parser.error(
            f"--kv-cache-state={args.kv_cache_state} disagrees with server /health ({actual_kv_state})"
        )
    tokenizer = load_local_tokenizer(args.tokenizer or args.model, args.allow_model_download)
    token_count = (
        (lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        if tokenizer is not None
        else None
    )

    summaries: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    case_number = 0
    for prompt_length in args.prompt_lengths:
        prompt = make_prompt(tokenizer, prompt_length)
        for generation_length in args.generation_lengths:
            for stream in modes:
                for concurrency in args.concurrency:
                    case_number += 1
                    common = {
                        "base_url": base_url,
                        "model": args.model,
                        "prompt": prompt,
                        "max_tokens": generation_length,
                        "stream": stream,
                        "seed": args.seed,
                        "timeout": args.timeout,
                        "token_count": token_count,
                    }
                    for warmup_index in range(args.warmup):
                        request_once(**common, request_index=-(warmup_index + 1))

                    started = time.perf_counter()
                    runs: list[dict[str, Any]] = []
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futures = [
                            pool.submit(request_once, **common, request_index=index)
                            for index in range(args.repeats)
                        ]
                        for future in as_completed(futures):
                            runs.append(future.result())
                    wall_seconds = time.perf_counter() - started
                    runs.sort(key=lambda run: run["request_index"])
                    case_id = f"case-{case_number:03d}"
                    for run in runs:
                        run.update(
                            {
                                "case_id": case_id,
                                "stream": stream,
                                "client_concurrency": concurrency,
                                "target_prompt_tokens": prompt_length,
                                "max_tokens": generation_length,
                            }
                        )
                    all_runs.extend(runs)
                    summary = summarize_case(
                        runs,
                        wall_seconds=wall_seconds,
                        stream=stream,
                        concurrency=concurrency,
                        target_prompt_tokens=prompt_length,
                        max_tokens=generation_length,
                        kv_cache_state=args.kv_cache_state,
                    )
                    summary["case_id"] = case_id
                    summaries.append(summary)
                    print(
                        f"{case_id}: stream={stream} prompt={prompt_length} max={generation_length} "
                        f"concurrency={concurrency} success={summary['successful']}/{summary['requests']} "
                        f"p95={summary['e2e_p95_ms']} ms"
                    )

    scheduler_after = get_json(f"{base_url}/scheduler/status", args.timeout)
    metadata = runtime_metadata()
    metadata.update(
        {
            "timestamp": datetime.now().astimezone().isoformat(),
            "metric_scope": "HTTP client observed",
            "base_url": base_url,
            "model": args.model,
            "server_health": health,
            "kv_cache_state": args.kv_cache_state,
            "warmup_per_case": args.warmup,
            "repeats_per_case": args.repeats,
            "seed": args.seed,
            "planned_requests": planned_requests,
            "tokenizer_available": tokenizer is not None,
        }
    )
    payload = {
        "metadata": metadata,
        "scheduler_before": scheduler_before,
        "scheduler_after": scheduler_after,
        "summaries": summaries,
        "runs": all_runs,
    }
    prefix = args.output_prefix or Path("benchmarks/results") / datetime.now().strftime(
        "http-benchmark-%Y%m%d-%H%M%S"
    )
    write_json(prefix.with_suffix(".json"), payload)
    write_csv(prefix.with_suffix(".csv"), summaries)
    columns = [
        "case_id",
        "stream",
        "kv_cache",
        "target_prompt_tokens",
        "max_tokens",
        "client_concurrency",
        "successful",
        "failed",
        "request_throughput_rps",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "e2e_p99_ms",
        "ttft_p50_ms",
        "tpot_p50_ms",
        "tokens_per_second_p50",
        "queue_wait_p95_ms",
    ]
    write_markdown(
        prefix.with_suffix(".md"),
        title="MiniLLM-Serve HTTP Benchmark",
        metadata=metadata,
        rows=summaries,
        columns=columns,
        notes=[
            "HTTP metrics include client/network overhead; engine metrics are reported separately.",
            "TTFT is the first non-empty assistant content delta, not the role-only SSE chunk.",
            "TPOT = (E2E - TTFT) / (completion_tokens - 1) for successful streaming requests.",
            "SSE chunks are reconstructed and tokenized; chunk count is never used as token count.",
            "Non-streaming TTFT and TPOT are N/A because the client receives one completed JSON body.",
            "Client concurrency is not GPU execution concurrency; see server_health and scheduler settings.",
        ],
    )
    print(f"Reports: {prefix.with_suffix('.json')}, {prefix.with_suffix('.csv')}, {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
