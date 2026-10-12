# Runtime Benchmark v2 — framework

## Configured scheduler experiments and reports

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.report_generator
```

Parameters live in `configs/decode_flood.json`, `configs/mixed_length.json`, and
`configs/continuous_arrival.json`. Each group defines count, ID prefix, arrival
start/interval, phase cycle, and synthetic service-duration cycle. The total
workload is capped at 1000 unique requests. No model or external service is used.

The command saves two policy records per scenario under `results/*.json`, and
generates [the consolidated report](reports/runtime_benchmark_report.md).
Local JSON results are ignored by Git. Each record includes the full workload
configuration, policy, metrics, UTC timestamp, schema version, and configuration
SHA-256. Metrics and config hashes reproduce exactly; timestamps vary.
Use `--config-dir`, `--results-dir`, and `--report` to select alternative paths.
Rerunning replaces the selected output files; historical benchmark reports are
not touched. Report generation uses saved structured records rather than copied
numeric tables. The report contains full configs so ignored results can be
regenerated on another machine.

This offline tool tests RuntimeFacade selection and request execution behavior
with FakeWorker. It loads no model, allocates no GPU resources, and performs no
HTTP requests. These measurements are Python runtime simulation observations;
they cannot establish real model latency, GPU throughput, or batching speedup.

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.runner --scenario single
.\.venv\Scripts\python.exe -m benchmarks.runtime.runner --scenario mixed
```

Both commands print structured JSON to stdout and create no result files.
For programmatic use, pass a `Scenario` to `RuntimeBenchmarkRunner.run()`;
`BenchmarkResult.to_dict()` is JSON serializable. A custom fake worker may be
injected through the constructor using ExecutionWorkerProtocol.

An empty runtime step and snapshot warm lazy imports before measured submission.
All requests are submitted before the first execution cycle. Scenario controls
request token limits, synthetic prompt word count, batch selection limit, and
maximum cycles. Built-in scenarios use deterministic IDs, fixed prompts, and
one fixed token per decode step. No random sampling or artificial sleeps occur.

Metric definitions:

- Queue delay: request start minus creation, in milliseconds; this approximates
  time before first admission, not cumulative queue time between decode steps.
- TTFT: first token observation minus request start, including fake prefill.
- Total latency: terminal time minus creation, including queue delay. This differs
  from RuntimeMetrics.total_latency_ms, which starts at request execution.
- Decode tokens: actual per-request token observations, not selected batch count.
- Scheduler decision: cycle, incoming execution phase, selected request IDs,
  and monotonic elapsed time at cycle completion. WAITING selections run prefill.

Request timestamps use the runtime wall clock; unavailable or backwards duration
measurements are `null`. Wall duration and decision timestamps use perf_counter.
The current facade chooses decode before waiting and runs homogeneous batches.
No fairness-policy comparison or real Continuous Batching is implemented here.

Runs are bounded by request count, token limits, batch size, and maximum cycles.
If the cycle budget expires, remaining requests are cancelled and the result
sets `step_budget_exhausted=true`. Cleanup runs even on worker exceptions.
The cycle budget cannot interrupt a custom synchronous worker that hangs; use
only bounded fake workers. No measurements are copied into historical reports.
