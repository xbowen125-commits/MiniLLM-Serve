# Experimental TransformersWorker benchmark

This opt-in benchmark drives RuntimeFacade with a real TransformersWorker,
executing sequential per-request prefill and token decode. It requires a complete
local model/tokenizer cache; loading uses local_files_only=True and never downloads.
Model loading and empty-runtime import warming are excluded from request timing.
Default warmup count is 1 and measured repetition count is 10. Greedy sampling,
fixed seed 0, identical prompt, device, and precision are used within each run.

From the project root in PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m benchmarks.runtime.real_model.benchmark_real_worker `
  --run-slow --model Qwen/Qwen2.5-0.5B-Instruct --device cpu `
  --max-tokens 32 --warmups 1 --repeats 10 `
  --output benchmarks/runtime/results/real_worker_cpu.json
```

Change --device to cuda for optional GPU execution (FP16). CPU uses FP32;
do not treat CPU/CUDA comparisons as precision-controlled experiments. CUDA
unavailability is an error. A local model directory or cached
HuggingFaceTB/SmolLM2-135M is also accepted. Real execution requires --run-slow.
Output JSON includes model/device/dtype, version metadata, timestamp, repeat
configuration, per-run actual prompt length and generated token count. Prompt
contents and local cache directories are not separately logged.

Metric scope is engine runtime, not HTTP:

- TTFT: request timing start before prefill to the first observed generated token.
- Decode latency: sum of decode runtime.step durations, including first-token
  sampling from prefill logits; not TPOT or post-first-token-only latency.
- Total latency: prefill plus decode and loop overhead through terminal state.
- Generated tokens: length of the authoritative request token list, including EOS.
- Tokens/s: generated tokens / total latency seconds, including prefill.

CUDA synchronizes before/after measured steps. Memory values are bytes:

- allocated_memory_before: after model load, before request prefill.
- allocated_memory_after: after request cleanup, before model disposal.
- allocated_memory_after_prefill and prefill_allocated_delta: all allocated
  tensors including logits and K/V; this delta is not labelled pure KV memory.
- kv_cache_memory_delta: unique live K/V tensor storage bytes after prefill minus
  the initial empty request-cache size (zero). This is a prefill-time cache
  footprint, not final decode growth, peak memory, or allocator reservation.
  Unsupported cache layouts report null; CPU CUDA fields are null.

Normal tests inject FakeWorker and validate JSON/timing formulas without models.
An opt-in pytest smoke test is marked slow:

```powershell
.\.venv\Scripts\python.exe -m pytest -m slow tests/test_real_worker_benchmark.py -v
```

Use MINILLM_RUNTIME_TEST_MODEL and MINILLM_RUNTIME_TEST_DEVICE for that smoke
test. Its single repetition is correctness validation, not a performance study.
No GPU tensor batching, Continuous Batching, or PagedAttention is implemented.
No real benchmark result is claimed until this command runs on a cached model.
