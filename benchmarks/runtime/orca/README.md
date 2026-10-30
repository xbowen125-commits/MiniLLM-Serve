# Orca-inspired Iteration Scheduling Benchmark

This experiment reuses the existing offline `ContinuousBatch`, `BatchExecutor`
and `IterationScheduler`. It evaluates logical iteration scheduling with the
unchanged virtual-clock Fake Worker. It is not a GPU performance test or a
reproduction of Orca's selective batching optimizations.

## Run

From the project root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.orca.report
```

Default outputs (Git-ignored):

- `benchmarks/runtime/results/orca-validity/orca_iteration.json`
- `benchmarks/runtime/results/orca-validity/orca_iteration.md`
- `benchmarks/runtime/results/orca-validity/case_N_iterations.jsonl`

The new directory preserves prior results under `results/orca` and `results/orca-v2`.
Use `--output-dir` for another destination; repeated runs replace files there.

## Workload and trace

Four deterministic cases cover simultaneous/staggered long-plus-short requests
and mixed prompt/output lengths. Each request records synthetic prompt length,
generation length, arrival time and stable zero-based arrival rank. Prompt length
is descriptive metadata, not actual tokenizer output or a calibrated prefill
cost. Output length controls fake generation steps.

Each iteration records selected Decode/Prefill IDs, newly observed terminal IDs,
actual completions, arrivals, remaining groups and virtual start/end times.
`finished_request_ids` is observed when selecting a plan; execution completion
is separately recorded as `completed_this_iteration`. The last completion need
not appear in a subsequent plan because the experiment does not add empty steps.

## Metrics

- TTFT and total latency are arrival-relative and include waiting.
- Throughput is tokens or completed requests divided by elapsed virtual time,
  including idle gaps, not wall-clock Python speed.
- Batch utilization is group size/capacity weighted by busy phase durations.
- Average batch size is mean Decode+Prefill selections per iteration.
- Iteration count includes both prefill-only and decode-bearing cycles.
- Decode fairness is Jain's index of tokens per eligible active decode iteration;
  it excludes admission waiting and normally equals 1 with this scheduler.

Each group costs `base + per_request * group_size`. Configurations, per-request
metrics, environment metadata and full traces are saved in schema-v3 JSON. Numeric results reproduce;
timestamps and output hashes need not. No production speedup, starvation freedom,
GPU utilization or comparison against vLLM is claimed. No CUDA, PagedAttention,
Chunked Prefill or Prefix Cache is implemented.

Metric formulas: [METRICS.md](../../../docs/METRICS.md).
Environment provenance: [environment.md](../../../docs/environment.md).
JSONL `finished_request_ids` records actual cycle retirements, including the last
cycle; selection-time observations use `scheduler_finished_request_ids`.
Shared WorkloadRequest preserves prompt/generation lengths and virtual arrival
time. Legacy workloads with unknown prompt length report null, not a guess.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_orca_benchmark.py -q
```
