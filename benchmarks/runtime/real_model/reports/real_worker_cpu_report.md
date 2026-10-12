# Real TransformersWorker Benchmark Report

## Experiment Setup

Source artifact: `real_worker_cpu.json` (local JSON; Git-ignored).
Recorded run timestamp: 2026-10-12T12:32:43.217305+00:00

Source SHA-256: `3b23e8a69629612c112980f6b3815c8acca9e10b90b00e3f8fb3702045ad268a`

| Parameter | Recorded value |
| --- | --- |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Device | cpu |
| dtype | float32 |
| max_tokens | 32 |
| Measured repeats | 3 |
| Warmups | 1 |
| torch version | 2.14.1 |
| transformers version | 5.19.0 |

## Metrics

Arithmetic means are calculated from the recorded per-run values.
Unavailable values are excluded with explicit sample counts; they are not zero.

| Metric | Average | Unit | Valid samples |
| --- | ---: | --- | ---: |
| TTFT | 94.744533 | ms | 3 |
| Decode latency | 2276.634500 | ms | 3 |
| Total latency | 2370.888233 | ms | 3 |
| Tokens/s | 13.502615 | tokens/s | 3 |

TTFT runs from prefill start to the first observed token. Decode latency sums
all decode steps, including first-token sampling. Total latency includes prefill
and loop overhead. Tokens/s divides generated tokens by total latency; the table
reports the arithmetic mean of per-run ratios, not an aggregate throughput ratio.
Model loading and warmup samples are excluded by the benchmark producer.

## Run Summary

| Run | Prompt tokens | Generated tokens | TTFT (ms) | Decode (ms) | Total (ms) | Tokens/s | Finish |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 14 | 32 | 95.299500 | 2343.852400 | 2438.647100 | 13.122030 | length |
| 1 | 14 | 32 | 92.066700 | 2236.698900 | 2328.288100 | 13.744004 | length |
| 2 | 14 | 32 | 96.867400 | 2249.352200 | 2345.729500 | 13.641812 | length |

## Limitations

- This is a single-request sequential worker experiment, with no tensor batching.
- Recorded device is `cpu`; this report is not GPU validation.
- For CPU execution, CUDA memory measurements are N/A; no memory values are inferred.
- Tokens/s is per-request generation rate, not production request throughput.
- A small number of repeats does not establish stable tail latency or capacity.
- CPU thread count, hardware details, and prompt text are absent from the source
  artifact, limiting exact reproduction. No commit SHA is recorded.
- No HTTP, concurrency, Continuous Batching, or PagedAttention is measured.
- This document summarizes an existing artifact and does not rerun inference.

Report reproduction from the project root:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.real_model.report_generator
```
