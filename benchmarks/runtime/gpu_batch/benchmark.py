"""Bounded opt-in real CUDA batch benchmark; no downloads or CPU fallback."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import hashlib
import platform
import subprocess
import os
import uuid

import torch
import transformers

from minillm.continuous import BatchExecutor, IterationScheduler
from minillm.engine import GenerationConfig, RequestState
from minillm.runtime import RuntimeRequestContext
from minillm.workers import BatchedTransformersWorker, TransformersModelLoader
from .analysis import request_timing, execution_order, summarize, markdown


def gpu_telemetry():
    """Optional boundary snapshot, outside measured intervals; no process data."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return {"fields": ["temperature_c", "power_w", "sm_mhz", "memory_mhz", "gpu_utilization_percent"],
                "values": result.stdout.strip() if result.returncode == 0 else None}
    except (OSError, subprocess.TimeoutExpired):
        return None


class DiagnosticWorker(BatchedTransformersWorker):
    """Benchmark-local instrumentation: worker calls model.forward directly.

    Module __call__ hooks therefore cannot count this backend's forwards.
    """

    def __init__(self, *args, diagnostics=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.forwards = []
        self.diagnostics = diagnostics

    def _forward(self, ids, mask, cache):
        row = {"input_shape": list(ids.shape)}
        if self.diagnostics:
            row["events"] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            row["events"][0].record()
        self.forwards.append(row)
        try:
            return super()._forward(ids, mask, cache)
        finally:
            if self.diagnostics:
                row["events"][1].record()


def measure(loaded, batch_size, max_tokens, *, prompt="The capital of France is", diagnostics=False):
    worker = DiagnosticWorker(loaded, slot_capacity=batch_size, diagnostics=diagnostics)
    executor = BatchExecutor(worker, scheduler=IterationScheduler(max_batch_size=batch_size))
    prompts = [prompt] * batch_size
    contexts = [RuntimeRequestContext(RequestState(request_id=str(i), prompt=p,
        generation_config=GenerationConfig(max_tokens=max_tokens, temperature=0))) for i, p in enumerate(prompts)]
    first, finished, steps = {}, {}, []
    forwards = worker.forwards
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        for c in contexts:
            executor.add(c)
        for _ in range(max_tokens + 1):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            forward_offset = len(forwards)
            step = executor.step()
            torch.cuda.synchronize()
            now = time.perf_counter()
            steps.append({"decoded": list(step.decoded), "prefilled": list(step.prefilled),
                          "duration_s": now - tick})
            steps[-1].update({"start_s": tick - start, "end_s": now - start,
                "forward_call_count": len(forwards) - forward_offset,
                "generated_counts": {c.request_id: len(c.request.generated_tokens) for c in contexts},
                "slots": worker.pool.snapshot()})
            for c in contexts:
                if c.request.generated_tokens:
                    first.setdefault(c.request_id, now)
                if c.request.is_terminal:
                    finished.setdefault(c.request_id, now)
            if all(c.request.is_terminal for c in contexts):
                break
        if any(c.request.status.name != "FINISHED" for c in contexts):
            raise RuntimeError("batch failed or exceeded bounded step count")
        elapsed = time.perf_counter() - start
        rows = []
        for c in contexts:
            n = len(c.request.generated_tokens)
            rows.append({"request_id": c.request_id, "generated_tokens": n,
                         "token_ids": list(c.request.generated_tokens),
                         "finish_reason": executor.coordinator.get_state(c.request_id).finish_reason.value,
                         **request_timing(start, first[c.request_id], finished[c.request_id], n)})
        peak = torch.cuda.max_memory_allocated()
    finally:
        executor.close()
        worker.close()
        torch.cuda.synchronize()
    tokens = sum(r["generated_tokens"] for r in rows)
    for forward in forwards:
        if "events" in forward:
            begin, end = forward.pop("events")
            forward["cuda_elapsed_ms"] = begin.elapsed_time(end)
    return {"batch_size": batch_size, "requests": rows, "latency_s": elapsed,
            "forward_call_count": len(forwards), "forwards": forwards,
            "all_requests_reached_max_tokens": all(r["generated_tokens"] == max_tokens for r in rows),
            "cleanup_slots": worker.pool.snapshot(),
            "generated_tokens": tokens, "tokens_per_second": tokens / elapsed,
            "steps": steps, "gpu_memory_bytes": {"allocated_before": before,
                "reserved_before": reserved_before, "peak_allocated": peak,
                "peak_increment": peak - before, "allocated_after_cleanup": torch.cuda.memory_allocated(),
                "reserved_after_cleanup": torch.cuda.memory_reserved()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-slow", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default=None, help="cached Hugging Face commit; no downloads")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/runtime/results/gpu_batch/validity_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"))
    args = parser.parse_args()
    session_started = datetime.now(timezone.utc).isoformat()
    if not args.run_slow:
        parser.error("--run-slow required for real GPU execution")
    if not 1 <= args.max_tokens <= 128 or not 1 <= args.repeats <= 20 or not 0 <= args.warmups <= 5:
        parser.error("bounds: tokens 1..128, repeats 1..20, warmups 0..5")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; no CPU fallback or model download")
    paths = [args.output, args.output.with_suffix('.jsonl'), args.output.with_suffix('.md')]
    if any(p.exists() for p in paths):
        parser.error("refusing to overwrite an existing result, trace or report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    from huggingface_hub import snapshot_download
    if Path(args.model).is_dir():
        model_path, revision = args.model, None
    else:
        model_path = snapshot_download(args.model, revision=args.revision, local_files_only=True)
        revision = Path(model_path).name
    loaded = TransformersModelLoader(model_path, device="cuda", dtype=torch.float32,
                                     local_files_only=True).load()
    prompt_ids = loaded.tokenizer.encode(args.prompt, add_special_tokens=True)
    runs, warmups = [], []
    with args.output.with_suffix('.jsonl').open('x', encoding='utf-8') as journal:
        for phase, count, target in (("warmup", args.warmups, warmups), ("measured", args.repeats, runs)):
            for index, size in execution_order(count):
                telemetry_before = gpu_telemetry() if args.diagnostics else None
                row = {"repeat": index, "phase": phase, **measure(loaded, size, args.max_tokens,
                    prompt=args.prompt, diagnostics=args.diagnostics)}
                row.update({"telemetry_before": telemetry_before,
                            "telemetry_after": gpu_telemetry() if args.diagnostics else None})
                target.append(row)
                journal.write(json.dumps(row) + '\n')
                journal.flush()
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root, capture_output=True, text=True).stdout.strip()
    result = {"scope": "real-cuda-batch-worker", "timestamp": datetime.now(timezone.utc).isoformat(),
              "session_id": str(uuid.uuid4()), "pid": os.getpid(), "started_at": session_started,
              "model_revision": revision, "prompt_token_count": len(prompt_ids),
              "prompt_token_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
              "model": args.model, "dtype": "float32", "torch": torch.__version__,
              "transformers": transformers.__version__, "cuda": torch.version.cuda,
              "schema_version": 2, "python": platform.python_version(), "git_sha": commit,
              "dirty_worktree": bool(dirty), "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(), "configuration": {
                  "max_tokens": args.max_tokens, "warmups": args.warmups, "repeats": args.repeats,
                  "prompt": args.prompt, "temperature": 0, "seed": 0, "order": "alternating AB/BA pairs",
                  "diagnostics": args.diagnostics}, "runs": runs, "warmup_runs": warmups, "summary": summarize(runs)}
    with args.output.open('x', encoding='utf-8') as output:
        output.write(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix('.md').write_text(markdown(result), encoding='utf-8')
    print(args.output)


if __name__ == "__main__":
    main()
