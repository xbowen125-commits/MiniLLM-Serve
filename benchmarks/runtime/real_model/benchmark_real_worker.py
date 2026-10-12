"""Opt-in local-cache benchmark of real prefill and serial token decode."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import argparse
import json
from pathlib import Path
import time

from minillm.engine import GenerationConfig
from minillm.runtime import RuntimeFacade
from minillm.workers import TransformersModelLoader, TransformersWorker
from .metrics import cache_storage_bytes, calculate_metrics


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cpu"
    prompt: str = "Continue the number sequence: 1, 2, 3,"
    max_tokens: int = 32
    warmups: int = 1
    repeats: int = 10

    def __post_init__(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if not 1 <= self.max_tokens <= 4096 or not 0 <= self.warmups <= 100 or not 1 <= self.repeats <= 100:
            raise ValueError("invalid generation or repetition bounds")


def run_benchmark(config: BenchmarkConfig, *, worker=None):
    """Return JSON-ready engine observations; injected workers are mock scope."""
    import torch
    real = worker is None
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if worker is None:
        worker = TransformersWorker(loader=TransformersModelLoader(
            config.model_name, device=config.device,
            dtype=torch.float16 if config.device == "cuda" else torch.float32,
            local_files_only=True,
        ))
        worker.load()  # Loading is deliberately outside request timing.

    def synchronize():
        if config.device == "cuda":
            torch.cuda.synchronize()

    def allocated():
        return torch.cuda.memory_allocated() if config.device == "cuda" else None

    runs = []
    try:
        for index in range(config.warmups + config.repeats):
            runtime = RuntimeFacade(worker, max_batch_size=1)
            try:
                # Warm lazy scheduler imports without executing the model.
                runtime.step()
                runtime.snapshot()
                context = runtime.submit(config.prompt, GenerationConfig(
                    max_tokens=config.max_tokens, temperature=0, seed=0,
                ), request_id=f"benchmark-{index}")
                synchronize()
                before = allocated()
                start = time.perf_counter()
                runtime.step()
                synchronize()
                after_prefill = allocated()
                cache_bytes = cache_storage_bytes(context.request.kv_cache_ref)
                first = None
                decode_seconds = 0.0
                for _ in range(config.max_tokens):
                    if context.request.is_terminal:
                        break
                    synchronize()
                    decode_start = time.perf_counter()
                    runtime.step()
                    synchronize()
                    decoded_at = time.perf_counter()
                    decode_seconds += decoded_at - decode_start
                    if first is None and context.request.generated_tokens:
                        first = decoded_at
                synchronize()
                end = time.perf_counter()
                if not context.request.is_terminal:
                    raise RuntimeError("request did not terminate within token budget")
                result = runtime.get_result(context.request_id)
                if result.error is not None or result.status.value != "finished":
                    raise RuntimeError(f"benchmark request failed: {result.error}")
                metrics = calculate_metrics(
                    start=start, first_token=first, end=end,
                    decode_seconds=decode_seconds,
                    generated_tokens=len(result.generated_tokens),
                )
                runtime.close()
                release = getattr(worker, "release_request", None)
                if release is not None:
                    release(context.request_id)
                synchronize()
                after = allocated()
                if index >= config.warmups:
                    runs.append({
                        "run_index": index - config.warmups,
                        "prompt_length": context.decode_state.prefill_tokens,
                        "generated_tokens": metrics.generated_tokens,
                        "metrics": metrics.to_dict(),
                        "finish_reason": result.finish_reason.value if result.finish_reason else None,
                        "cuda_memory": {
                            "allocated_memory_before_bytes": before,
                            "allocated_memory_after_bytes": after,
                            "allocated_memory_after_prefill_bytes": after_prefill,
                            "prefill_allocated_delta_bytes": None if before is None else after_prefill - before,
                            "kv_cache_memory_delta_bytes": cache_bytes if config.device == "cuda" else None,
                        },
                    })
            finally:
                runtime.close()
                release = getattr(worker, "release_request", None)
                if release is not None:
                    release(f"benchmark-{index}")
    finally:
        # Injected workers are caller-owned, while real workers belong to this run.
        if real:
            worker.close()

    return {
        "schema_version": 1,
        "scope": "real-transformers-worker" if real else "mock-worker-format-validation",
        "model_name": config.model_name, "device": config.device,
        "dtype": "float16" if real and config.device == "cuda" else "float32" if real else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {k: v for k, v in asdict(config).items() if k != "prompt"},
        "versions": {name: version(name) for name in ("torch", "transformers")},
        "runs": runs,
    }


def main():
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-slow", action="store_true", help="explicitly enable real-model execution")
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prompt", default=defaults.prompt)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_slow:
        parser.error("real execution requires --run-slow and a complete local model cache")
    result = run_benchmark(BenchmarkConfig(args.model, args.device, args.prompt,
                                         args.max_tokens, args.warmups, args.repeats))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
