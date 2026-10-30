# Mission 23.6 — GPU Benchmark Validity and Repeatability

## Evidence and scope

Inspected on 2026-10-20 (Asia/Shanghai): RTX 5060 Laptop GPU, PyTorch
2.11.0+cu128, CUDA 12.8, Transformers 5.19.0, Qwen2.5-0.5B-Instruct, FP32.
No RuntimeFacade, HTTP API, IterationScheduler or Worker source changes.

Original local artifact: `benchmarks/runtime/results/gpu_batch/results.json`.
SHA-256 before and after investigation:
`AA4C8CFD6A232115D36E3F3085EDE95B0398F70CE72042C617DA112C95A6D273`.
It is unchanged. Configuration: max_tokens=16, warmups=1, repeats=10 per shape.
The historical artifact does not contain a source commit SHA.

Corrected experiment: `benchmarks/runtime/results/gpu_batch/validity_20261020T051421146085.json`,
plus same-stem `.jsonl` and `.md`. These local artifacts remain Git-ignored.
Recorded HEAD is `dc9db6e9cb2585bd7eb141717206586ec0e01fb3`, with dirty worktree.
JSON also records benchmark source SHA-256; retain uncommitted sources alongside
artifacts because HEAD alone does not identify them.

## Timing and metric audit

- Old code synchronizes CUDA before the run and before/after every executor step.
  No evidence supports an unsynchronized-timing explanation for the old split.
- Model loading and warmups occur outside measured calls. Peak allocation resets
  after loading, before each run. Model baseline is included in absolute peaks;
  peak increment subtracts that baseline, not all non-KV allocations.
- Both shapes start before admission/prefill and observe completion after step
  synchronization. Python work, transfers, KV split/merge and sync overhead remain.
- TTFT = first synchronized token observation - run start. TPOT =
  `(request_finish - first_token) / (actual_tokens - 1)`, null for one token.
  This is runtime TPOT, not pure model-forward latency.
- Batch tokens/s = sum of actual request tokens / batch elapsed time. Mean of
  per-run rates and pooled total tokens / total time differ; both are now reported.
- All old requests generated 16 tokens; each group had one Prefill and 16 Decode
  runtime steps. First Decode samples Prefill logits. Source predicts 16 forwards,
  but old records cannot verify actual forward calls retrospectively.
- Cache rows are cloned and slots are independent. The real GPU batch-1 vs batch-2
  greedy token equality slow test passed during this audit, for its test prompts.

## Historical performance stratification

Every batch-2 run generated 32 tokens, executed 17 runtime steps and reported
the same peak allocation of 2,002,088,448 bytes. Decode averages include the
first no-forward decode step.

| Repeat | Latency (s) | Tokens/s | Mean decode-step duration (s) |
| --- | ---: | ---: | ---: |
| 0 | 0.635923 | 50.320558 | 0.038327 |
| 1 | 0.880036 | 36.362136 | 0.051789 |
| 2 | 0.833720 | 38.382214 | 0.049004 |
| 3 | 0.818452 | 39.098200 | 0.048278 |
| 4 | 0.880525 | 36.341934 | 0.051671 |
| 5 | 0.740688 | 43.203105 | 0.043101 |
| 6 | 0.364662 | 87.752518 | 0.021339 |
| 7 | 0.371496 | 86.138093 | 0.021704 |
| 8 | 0.331886 | 96.418534 | 0.019379 |
| 9 | 0.331557 | 96.514296 | 0.019338 |

Request TTFT ranged roughly 22–54 ms and TPOT 20.6–55.2 ms. Both rows share
observation times because results are observed at one synchronized boundary.
The split is present in step durations, not merely throughput aggregation.
Token count, early termination and allocation peak do not explain it.

**Historical root cause is unresolved.** No clocks, power, thermals, CPU scheduling,
competing-load trace or actual forward events were recorded. GPU operating state,
driver/library paths and host overhead remain hypotheses, not proven causes.
Wall-clock steps alone cannot distinguish these possibilities.

## Experimental design changes

Original execution ran all batch-1 measurements before batch-2 and added a
different Prompt for batch-2. New controls use identical repeated Prompt, FP32,
greedy sampling, seed 0 and max_tokens=16, with one warmup per shape. Ten measured
pairs alternate AB on even repeats, BA on odd repeats. No samples are filtered.

Every warmup and measured run is flushed to JSONL. Timestamped filenames and
overwrite refusal preserve prior records. Each step includes generated counts,
slot state, relative timing and actual forward count; token IDs/finish reasons
are retained. Optional forward CUDA events measure device timeline spans, not
isolated kernel compute. Optional boundary nvidia-smi snapshots are outside timed
intervals and record temperature, power, clocks and utilization, not process data.

Reports retain every run and give mean, median, sample standard deviation (n-1),
min, max and linearly interpolated P95. Batch-2 request observations are correlated,
not twenty independent repetitions. New diagnostics and changed Prompt mix mean
old/new performance is not a clean speedup comparison.

### Retained instrumentation trial

`validity_20261020T051339210065.json` and companions are retained but have invalid
zero forward counts: Module hooks did not fire because the backend calls
`model.forward()` directly instead of Module `__call__`. This tooling incident
is explicitly excluded from forward-count evidence, not deleted as a bad sample.
The corrected benchmark-local subclass wraps `_forward`; a Mock test verifies
Prefill and later Decode counting. Worker source is unchanged.

## Corrected real GPU results

20 measured groups plus two warmups completed. Every request generated exactly
16 tokens; each measured group made 16 actual forwards and ended with zero slots.
Full per-run timings, forward events, telemetry and memory are in the JSON/JSONL;
the generated Markdown contains distributions and the chronological run table.

| Batch | Mean tokens/s | Median | Sample SD | Min | Max | P95 | Pooled tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 46.927790 | 46.977693 | 1.476277 | 44.181650 | 49.037306 | 48.634596 | 46.885378 |
| 2 | 85.586600 | 86.617585 | 3.978645 | 76.855071 | 89.868758 | 89.793044 | 85.412029 |

| Batch | Median request TTFT (ms) | Median request TPOT (ms) | Median batch latency (ms) |
| --- | ---: | ---: | ---: |
| 1 | 22.362100 | 21.215027 | 340.594100 |
| 2 | 24.347550 | 22.992317 | 369.453550 |

This session shows higher aggregate throughput for batch-2 but higher per-request
TTFT/TPOT. The old six/four split did not recur; this neither explains it nor
establishes universal stability. New boundary snapshots show changing GPU clocks,
so hardware conditions were not fixed.

## Tests and reproduction

Five new Mock/metric tests cover exact boundaries, alternating order, retained
slow values, sample statistics, pooled throughput and direct-forward counting.
Real GPU greedy equality test passed. Full pytest: **378 passed**; compileall and
git diff --check passed (line-ending warnings only).

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_benchmark_validity.py tests/test_batched_transformers_worker.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_real_batched_transformers_worker.py -m slow -v
.\.venv\Scripts\python.exe -m benchmarks.runtime.gpu_batch.benchmark `
  --run-slow --max-tokens 16 --warmups 1 --repeats 10 --diagnostics
```

No downloads; model load excluded; results use fresh timestamped paths. Existing
explicit output paths are refused. Next: repeat independent sessions with recorded
laptop power and competing-load conditions; profile CPU/CUDA timelines if the
split recurs. Keep diagnostic/non-diagnostic results separate. No kernel or
Runtime changes are warranted by current evidence.
