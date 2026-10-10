"""Mission 06：比较 KV Cache 开关下的单请求生成性能。

指标口径：
- TTFT：从开始调用生成器到第一个可发送 chunk；
- Prefill：首个完整 Prompt 模型步（含采样同步）；
- TPOT：后续 Decode 模型步的平均耗时；
- Tokens/s：实际 completion tokens / 总生成时间；
- CUDA peak：PyTorch 统计的峰值已分配显存及相对模型基线的增量。

模型加载和预热不计入正式结果。脚本不下载模型的保证由 --offline 控制。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch

from minillm.config import Settings
from minillm.engine import GenerationEngine, GenerationMetrics

try:
    from benchmark_utils import runtime_metadata, write_csv, write_json, write_markdown
except ModuleNotFoundError:  # imported as benchmarks.benchmark_kv_cache in unit tests
    from benchmarks.benchmark_utils import runtime_metadata, write_csv, write_json, write_markdown


def parse_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("长度必须是逗号分隔的正整数")
    return lengths


def make_prompt(tokenizer, target_tokens: int) -> str:
    seed = "KV Cache benchmark prompt. 大模型推理需要区分预填充与逐 token 解码。"
    text = (seed + " ") * max(2, target_tokens)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_once(engine: GenerationEngine, prompt: str, max_tokens: int, seed: int) -> dict:
    metrics = GenerationMetrics()
    is_cuda = engine.device.startswith("cuda")
    if is_cuda:
        torch.cuda.empty_cache()
        synchronize(engine.device)
        baseline_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_bytes = 0

    started = time.perf_counter()
    first_chunk_at = None
    final = None
    for chunk in engine.stream_generate(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=seed,
        metrics=metrics,
    ):
        if first_chunk_at is None and chunk.text:
            synchronize(engine.device)
            first_chunk_at = time.perf_counter()
        if chunk.finish_reason is not None:
            final = chunk
    synchronize(engine.device)
    total_seconds = time.perf_counter() - started

    if final is None:
        raise RuntimeError("生成未正常结束")

    if is_cuda:
        peak_bytes = torch.cuda.max_memory_allocated()
    else:
        peak_bytes = 0

    return {
        "prompt_tokens": final.prompt_tokens,
        "completion_tokens": final.completion_tokens,
        "finish_reason": final.finish_reason,
        "ttft_ms": (first_chunk_at - started) * 1000 if first_chunk_at is not None else None,
        "prefill_ms": metrics.prefill_seconds * 1000,
        "tpot_ms": (metrics.decode_seconds / metrics.decode_steps * 1000) if metrics.decode_steps else 0.0,
        "tokens_per_second": final.completion_tokens / total_seconds,
        "total_ms": total_seconds * 1000,
        "cuda_peak_mb": peak_bytes / 1024**2,
        "cuda_peak_increment_mb": max(0, peak_bytes - baseline_bytes) / 1024**2,
    }


def aggregate(runs: list[dict]) -> dict:
    result = {
        "prompt_tokens": runs[0]["prompt_tokens"],
        "completion_tokens_median": statistics.median(run["completion_tokens"] for run in runs),
        "finish_reasons": sorted({run["finish_reason"] for run in runs}),
    }
    for key in ("ttft_ms", "prefill_ms", "tpot_ms", "tokens_per_second", "total_ms"):
        samples = [run[key] for run in runs if run[key] is not None]
        result[key] = statistics.median(samples) if samples else None
    result["cuda_peak_mb"] = max(run["cuda_peak_mb"] for run in runs)
    result["cuda_peak_increment_mb"] = max(run["cuda_peak_increment_mb"] for run in runs)
    return result


def print_results(results: list[dict]) -> None:
    columns = (
        "cache",
        "prompt",
        "max_new",
        "actual_new",
        "TTFT ms",
        "Prefill ms",
        "TPOT ms",
        "tok/s",
        "total ms",
        "peak MB",
        "peak +MB",
    )
    print(" | ".join(f"{column:>10}" for column in columns))
    print("-" * 142)
    for item in results:
        values = (
            "on" if item["use_kv_cache"] else "off",
            item["prompt_tokens"],
            item["requested_max_tokens"],
            item["completion_tokens_median"],
            f"{item['ttft_ms']:.2f}" if item["ttft_ms"] is not None else "N/A",
            f"{item['prefill_ms']:.2f}",
            f"{item['tpot_ms']:.2f}",
            f"{item['tokens_per_second']:.2f}",
            f"{item['total_ms']:.2f}",
            f"{item['cuda_peak_mb']:.1f}",
            f"{item['cuda_peak_increment_mb']:.1f}",
        )
        print(" | ".join(f"{str(value):>10}" for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM-Serve KV Cache Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto", help="auto / cuda / cuda:0 / cpu")
    parser.add_argument("--dtype", default="auto", help="auto / float16 / bfloat16 / float32")
    parser.add_argument("--prompt-lengths", type=parse_lengths, default=parse_lengths("64,256"))
    parser.add_argument("--generation-lengths", type=parse_lengths, default=parse_lengths("32,64"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="只使用本地 Hugging Face 缓存")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup 必须 >= 0，--repeats 必须 >= 1")

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    max_model_len = max(args.prompt_lengths) + max(args.generation_lengths) + 64
    settings = Settings(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        max_model_len=max_model_len,
        use_kv_cache=True,
    )
    engine = GenerationEngine(settings)
    print(f"加载模型: {args.model}")
    engine.load()
    print(f"设备: {engine.device}, dtype: {engine.dtype}, repeats: {args.repeats}, warmup: {args.warmup}")

    results = []
    for target_prompt_tokens in args.prompt_lengths:
        prompt = make_prompt(engine.tokenizer, target_prompt_tokens)
        for max_tokens in args.generation_lengths:
            for use_kv_cache in (False, True):
                engine.settings.use_kv_cache = use_kv_cache
                for _ in range(args.warmup):
                    run_once(engine, prompt, max_tokens, args.seed)
                runs = [run_once(engine, prompt, max_tokens, args.seed) for _ in range(args.repeats)]
                summary = aggregate(runs)
                summary.update(
                    {
                        "model": args.model,
                        "device": engine.device,
                        "dtype": str(engine.dtype),
                        "use_kv_cache": use_kv_cache,
                        "target_prompt_tokens": target_prompt_tokens,
                        "requested_max_tokens": max_tokens,
                        "repeats": args.repeats,
                        "runs": runs,
                    }
                )
                results.append(summary)

    print_results(results)
    metadata = runtime_metadata()
    metadata.update(
        {
            "timestamp": datetime.now().astimezone().isoformat(),
            "metric_scope": "model engine",
            "model": args.model,
            "device": engine.device,
            "dtype": str(engine.dtype),
            "warmup_per_case": args.warmup,
            "repeats_per_case": args.repeats,
            "seed": args.seed,
            "offline": args.offline,
        }
    )
    prefix = args.output_prefix or Path("benchmarks/results") / datetime.now().strftime(
        "engine-kv-%Y%m%d-%H%M%S"
    )
    json_path = args.json_output or prefix.with_suffix(".json")
    csv_path = args.csv_output or prefix.with_suffix(".csv")
    markdown_path = args.markdown_output or prefix.with_suffix(".md")
    write_json(json_path, {"metadata": metadata, "results": results})
    summary_rows = [{key: value for key, value in result.items() if key != "runs"} for result in results]
    write_csv(csv_path, summary_rows)
    write_markdown(
        markdown_path,
        title="MiniLLM-Serve Engine KV Cache Benchmark",
        metadata=metadata,
        rows=summary_rows,
        columns=(
            "use_kv_cache",
            "prompt_tokens",
            "requested_max_tokens",
            "completion_tokens_median",
            "ttft_ms",
            "prefill_ms",
            "tpot_ms",
            "tokens_per_second",
            "total_ms",
            "cuda_peak_mb",
            "cuda_peak_increment_mb",
        ),
        notes=(
            "TTFT is time to the first non-empty generated text chunk.",
            "TPOT is the mean measured model-step time after Prefill; it excludes client/network overhead.",
            "Tokens/s is actual completion tokens divided by total engine generation time.",
            "CUDA peak is reset per measured run and excludes model loading; peak increment is relative to the loaded-model baseline.",
            "Results are local samples, not universal hardware claims.",
        ),
    )
    print(f"报告已写入: {json_path}, {csv_path}, {markdown_path}")


if __name__ == "__main__":
    main()
