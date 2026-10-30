# Experimental real CUDA batch worker

Status: real GPU correctness and diagnostic experiments ran in Mission 23.6.
See [the validity audit](../../../docs/gpu_benchmark_validity.md) for provenance,
results and unresolved historical stratification. No universal speedup is claimed.

Cross-process evidence: [Mission 23.7 replication](../../../docs/gpu_cross_session_replication.md).
Run three sequential fresh model-loading processes using
`python -m benchmarks.runtime.gpu_batch.replication --run-slow --sessions 3`.
Each session retains its own JSON/JSONL/Markdown and process identity. The runner
validates fixed configuration and explicit files; it does not glob/merge old trials.

The new backend reuses row split/merge logic, moving input IDs, masks and position
IDs to the loaded model device. SlotKVPool owns request-local cache references.
Batch size is 1 or 2. Cache rows are cloned and merged by copying, not paged
attention; full-attention DynamicCache layers are required. FP32 greedy generation
is used for the initial correctness study. Floating-point batch shapes can still
affect near-tied logits; the slow test compares exact tokens rather than assuming equality.

Use an existing CUDA PyTorch environment and complete cached Qwen model; these
commands never download a model and never fall back to CPU:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_real_batched_transformers_worker.py -m slow -v
.\.venv\Scripts\python.exe -m benchmarks.runtime.gpu_batch.benchmark `
  --run-slow --model Qwen/Qwen2.5-0.5B-Instruct --max-tokens 16 --warmups 1 --repeats 10
```

The slow test explicitly skips if CUDA is absent; a skip is not GPU validation.
On CUDA, missing local models or token disagreement fails rather than skipping.
The benchmark fails immediately without CUDA. Timestamped output refuses overwrites and is Git-ignored under
`benchmarks/runtime/results/gpu_batch/`.

Metrics synchronize CUDA at step boundaries. TTFT starts before admission/prefill
and ends when the first token result is observed. TPOT is `(finish - first) / (n-1)`
or null for one token. Latency includes prefill and runtime overhead. Throughput
is total tokens of the batch / elapsed time. Both shapes now use the same fixed
Prompt, repeated in batch-2. Measured AB/BA pairs alternate order. Model loading
and warmups are excluded; warmup rows are retained separately in the JSONL journal.
Use --diagnostics for CUDA forward events and boundary telemetry. JSON and Markdown
include full distributions; GPU batch throughput and request latency are separate.

Memory reports allocated baseline/peak/cleanup and reserved baseline/cleanup;
peak increment includes logits, cache copying and other tensors, not pure KV bytes.
Releasing slots drops references, but does not force the CUDA allocator to return
reserved memory. No custom kernels, PagedAttention, FlashInfer or tensor parallelism.
