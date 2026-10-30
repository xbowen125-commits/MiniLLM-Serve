# Mission 23.7 — Cross-session GPU Benchmark Replication

## Scope and session definition

Completed on 2026-10-20 (Asia/Shanghai): **three distinct, serial Python processes**.
Each process loads the model afresh, creates its own CUDA context/allocator state,
runs two warmup groups and twenty measured groups, then exits. Sessions share the
same machine, driver, OS, cache and closely spaced time window. This is not a
reboot, cross-day or controlled-power replication study.

All 60 measured groups and six warmup groups are retained. Each shape has ten
measurements per session; no Mission 23.6 runs or invalid instrumentation trials
are merged into this sample.

## Tool audit and provenance

The existing `execution_order` yields AB, BA, AB, BA pairs. JSONL contains warmup
and measured rows separately; reading it back exactly matches `warmup_runs + runs`
in JSON. CUDA synchronizes before the group and before/after each step. Loading,
warmups and hardware queries are outside formal timing. Forward events are recorded
by a benchmark-local wrapper, not Module hooks. New output refuses existing files.

Mission 23.6's validated source has 20 measured rows, two warmups, 22 journal rows,
16 forwards per group, 16 tokens per request and zero final slots. Its invalid
zero-forward instrumentation trial remains separate. Explicit session paths, not
directory globs, define this analysis, preventing accidental trial/history merging.

The only benchmark instrumentation additions for replication are process UUID/PID,
start/end time, pinned local model revision, input-token count/hash and a separate
process runner/validator. No inference Runtime, Scheduler or Worker changes.

### Windows launcher incident

The first process completed successfully, but initial aggregation rejected a PID
comparison: Windows `.venv` launcher PID 36140 differed from actual Python PID
15012. Exit code was zero and all 22 journal rows matched the complete JSON.
The checker now records launcher and interpreter PIDs separately. That first
session was explicitly retained and combined with **two new processes**, not
discarded or rerun for a better number.

## Fixed experiment configuration

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Python | 3.13.5 |
| PyTorch / CUDA build | 2.11.0+cu128 / 12.8 |
| Transformers | 5.19.0 |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Cached model revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Prompt | `The capital of France is` repeated for batch=2 |
| Input length | 5 tokens per request, tokenizer add_special_tokens=True |
| Input-token SHA-256 | `9ee874ffc7ef6f6382c038283d6bcb24d52acf8e26d784a16a2f3d58815b2e39` |
| Output | 16 actual tokens per request |
| Generation | Greedy, temperature=0, seed=0; other options unchanged defaults |
| Dtype / device | float32 / cuda |
| Warmups / repetitions | 1 per shape / 10 per shape |
| Order | Repeat 0: 1,2; repeat 1: 2,1; alternating through repeat 9 |
| Diagnostics | CUDA forward events and before/after hardware snapshots enabled |
| Git HEAD | `dc9db6e9cb2585bd7eb141717206586ec0e01fb3` |
| Source state | Dirty, uncommitted; HEAD alone does not identify working sources |
| Benchmark source SHA-256 | `c794be5e46dcf6de16ad10d87d0e07717122d95729f4974ce25b3ef54e519433` |

TTFT/TPOT/throughput definitions remain unchanged from
[Mission 23.6](gpu_benchmark_validity.md). Group wall time includes orchestration,
cache copies and synchronization, not loading. Pooled throughput is total tokens
divided by total measured group time; the mean of run rates is also reported.

## Actual sessions and retained artifacts

All paths below are project-relative and Git-ignored.

| Session | Actual Python PID | Local time window (+08:00) | Raw JSON |
| --- | ---: | --- | --- |
| S1 | 15012 | 13:20:18–13:20:48 | `benchmarks/runtime/results/gpu_batch/replication_20261020T052014918527/session_1.json` |
| S2 | 2200 | 13:21:24–13:21:58 | `benchmarks/runtime/results/gpu_batch/replication_20261020T052120044185/session_2.json` |
| S3 | 13824 | 13:22:03–13:22:31 | `benchmarks/runtime/results/gpu_batch/replication_20261020T052120044185/session_3.json` |

Each JSON has companion JSONL/Markdown; logs retain process output.
`replication_20261020T052120044185/manifest.json` identifies all three sessions,
launcher PIDs and SHA-256 hashes. That directory also contains `summary.json` and
`summary.md` with full per-session distributions, memory and cross-session statistics.

## Results: throughput

Units: actual generated tokens/second. SD uses n-1; P95 uses linear interpolation.
Each row contains ten measured groups, all retained.

| Session | Batch | Mean | Median | SD | Min | Max | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 1 | 20.684117 | 20.227752 | 2.104852 | 17.844908 | 24.288480 | 23.837092 |
| S1 | 2 | 43.814582 | 38.282149 | 12.430681 | 31.467705 | 67.672520 | 64.344726 |
| S2 | 1 | 17.918389 | 16.666426 | 3.089069 | 14.355231 | 23.537401 | 22.811847 |
| S2 | 2 | 34.039696 | 33.571887 | 8.226740 | 21.193984 | 49.371775 | 46.457066 |
| S3 | 1 | 20.792244 | 19.928184 | 3.035629 | 16.770954 | 26.708938 | 25.486757 |
| S3 | 2 | 37.277914 | 36.589192 | 5.068140 | 31.705789 | 45.767487 | 45.489681 |

Across 30 groups per shape: batch=1 mean 19.798250, median 19.913177, SD 3.004346;
batch=2 mean 38.377397, median 36.723760, SD 9.696887. Pooled tokens/s are
19.356453 and 36.357350 respectively. Run pooling does not make 30 independent
sessions; there are only three. Complete min/max/P95 and latency distributions
are retained in the summary JSON/Markdown.

## Request latency and relative change

Medians in milliseconds; change is batch-2 median / batch-1 median minus one.
Batch-2 request rows are correlated, not independent repetitions.

| Session | Batch | TTFT | TPOT | Request end-to-end latency |
| --- | ---: | ---: | ---: | ---: |
| S1 | 1 | 30.888650 | 50.110700 | 790.989750 |
| S1 | 2 | 25.710500 | 53.267703 | 835.904200 |
| S2 | 1 | 49.113750 | 59.602933 | 960.112000 |
| S2 | 2 | 33.048750 | 61.437283 | 954.608000 |
| S3 | 1 | 28.265550 | 49.835600 | 802.876150 |
| S3 | 2 | 32.174650 | 56.620380 | 876.351000 |

| Session | Throughput change | TTFT change | TPOT change | Request latency change |
| --- | ---: | ---: | ---: | ---: |
| S1 | +89.26% | -16.76% | +6.30% | +5.68% |
| S2 | +101.43% | -32.71% | +3.08% | -0.57% |
| S3 | +83.61% | +13.83% | +13.61% | +9.15% |

Pooled request medians: TTFT 34.822500/29.999750 ms, TPOT
51.739317/56.259537 ms, end-to-end 803.478850/871.521350 ms (batch=1/2).
Batch=2 has a TPOT cost in every session. TTFT and total latency costs are not
uniform across sessions; reporting only aggregate throughput would hide that.

## Token, forward and memory checks

The validator accepts only the exact AB/BA order and measured count. Every
measured group had 16 forwards; each request generated 16 tokens; all final KV
slot counts were zero. Per-step counts and token IDs remain in raw results.

Every recorded peak was identical within each shape across all three sessions:

| Batch | Peak allocated bytes | Peak increment bytes |
| --- | ---: | ---: |
| 1 | 1,990,313,472 | 3,892,224 |
| 2 | 1,994,205,696 | 7,784,448 |

Increment includes logits and KV copying, not only KV tensors. Allocated/reserved
before and after cleanup remain in each raw row; zero slots is not proof that
the CUDA allocator returned reserved bytes to the OS.

## Stratification and unresolved evidence

The original six-slow/four-fast sequence did **not** recur as that temporal
pattern. Significant interleaved variation and much slower absolute execution
did recur. The Mission 23.6 medians of 46.98/86.62 tokens/s did not replicate.

In S1 batch=2, repeat 2 achieved 67.67 tokens/s versus repeat 6 at 31.47. Their
forward CUDA-event sums were 372.08 versus 775.89 ms; mean Decode wall steps
were 27.99 versus 62.22 ms. Both used 16 forwards, 32 total tokens and the same
peak bytes. In S2, repeat 1 versus repeat 4: 49.37 versus 21.19 tokens/s,
forward-event sums 483.03 versus 1,145.59 ms, mean Decode steps 38.85 versus
88.46 ms. S3 also varied (31.71–45.77 tokens/s for batch=2).

Thus variation is visible in wall-clock steps and device-timeline event spans,
not only summary arithmetic or token counts. CUDA event spans can include stream
idle gaps/host submission delay; they do not isolate kernel compute. Hardware
snapshots show changing clocks/power and cannot establish causal attribution.

Confound explicitly retained: a CPU-only related-test/compile check was run while
S2's process was active. Its exact overlap with measured intervals was not recorded,
so host contention cannot be ruled out. No sample is excluded for this reason.
Other desktop/background load and laptop power conditions were not controlled.
No driver, thermal, CPU or GPU-root-cause claim is justified.

## Validation, reproduction and next stage

Related Mock/statistics tests: **8 passed**. After all timing processes exited,
non-slow regression: **371 passed, 10 deselected**. Compileall and git diff --check passed
(existing LF/CRLF warnings only). Historical Mission 23.6 hashes were checked
before the experiment; final preservation checks are recorded in the task result.
No new greedy correctness claim is made by these timing sessions alone.

For a fresh three-process replication (preserve all old output):

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_replication.py tests/test_gpu_benchmark_validity.py -q
# Finish tests before launching timing sessions.
.\.venv\Scripts\python.exe -m benchmarks.runtime.gpu_batch.replication --run-slow --sessions 3
```

The optional `--existing-session` accepts an explicit complete JSON plus matching
journal; it counts toward the requested total, never searches historical globs.
The reproduction runner fixes the cached revision before launching children.

**Conclusion:** aggregate batch-2 advantage reproduced in these three processes
for this five-token Prompt and sixteen-token output. Absolute speed and uniform
latency improvements did not. Mission 24.0 can proceed as bounded engineering
research with these limitations; a claim of stable production GPU performance
should remain blocked pending quiet-host, cross-day/power-controlled replication
and host/device timeline profiling. No automatic commit or push was performed.
