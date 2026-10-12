# Unified Runtime Benchmark Report

## Experimental Scope

This report consolidates the saved Mission 19 experiments. All three are
**offline simulations**, not real model, HTTP, GPU, or production throughput
measurements. No comparison with vLLM performance is made. See
[architecture.md](architecture.md) for the separation from online serving.

The source JSON artifacts remain Git-ignored under `benchmarks/runtime/results/`.
Paths below identify local evidence; they are not links to published raw files.
This report was assembled from existing artifacts, without rerunning experiments.

| Experiment | Local source | Scope | Recorded date (Asia/Shanghai) |
| --- | --- | --- | --- |
| Paged KV memory | `benchmarks/runtime/results/paged_memory/paged_memory.json` | Simulated token capacity | 2026-10-12 |
| Continuous batch | `benchmarks/runtime/results/continuous_batch/continuous_batch.json` | Fake Worker, virtual clock | 2026-10-12 |
| Scheduler research | `benchmarks/runtime/results/scheduler_research/scheduler_research.json` | Virtual-clock admission | 2026-10-12 |

SHA-256 of the source bytes inspected for this report:

```text
paged_memory.json
F58963B9368134961706EF321096E70E9BC4BD525628054F4D1B56C724429E1A
continuous_batch.json
1A7C932AD543AE0F2B43E87E5A07CBF2A1F2D90ED9F386D997F289930C73BA0A
scheduler_research.json
6938C7530457113FB5561D78B0ADFD2B6F6A08596A99005E0C10B013FB75C873
```

These hashes identify local artifacts, not a source-code revision. New runs have
new timestamps and byte hashes even when deterministic metrics match. Hardware,
model and precision are not applicable to these simulations.

## Methodology

### 1. Paged KV Memory Experiment

[Implementation](../benchmarks/runtime/paged_memory/experiment.py) uses the
existing `PagedKVCacheManager` with one simulated byte per token. Thirty lengths
are drawn with seed 42 from `[32, 64, 128, 256, 512, 1024]`. Each request has a
maximum capacity of 1024 tokens; pages and growth steps are both 16 tokens.
All requests remain resident until the final measurement, then are released.

- Contiguous reserves the maximum token capacity once per request.
- Paged capacity grows as logical tokens cross page boundaries.
- Requested tokens are total occupied sequence positions, not just generated tokens.
- Allocated tokens are reserved capacity; wasted = allocated - requested.
- Utilization = requested / allocated.
- Contiguous block count means one allocation per request; paged block count
  means fixed-size pages. Counts therefore represent different allocation sizes.

This is one deterministic workload, not repeated GPU allocation measurements.

### 2. Continuous Batch Experiment

[Runner](../benchmarks/runtime/continuous_batch/runner.py) drives the existing
offline `BatchExecutor` with a Fake Worker. Sequential execution allows one
active request; continuous execution allows four. Each cycle decodes existing
requests before admitting new Prefill requests. Arrivals are admitted at cycle
boundaries, not during a worker call.

Virtual phase cost is `base + per_request * group_size`:

| Phase | Base cost (simulated s) | Per-request cost (simulated s) |
| --- | ---: | ---: |
| Prefill | 0.02 | 0.01 |
| Decode | 0.01 | 0.005 |

| Scenario | Requests per strategy | Generated length | Arrival pattern |
| --- | ---: | --- | --- |
| short_requests | 8 | 2 each | All at time 0 |
| mixed_length | 8 | 2, 16, 4, 32, 2, 8, 16, 4 | All at time 0 |
| continuous_arrival | 8 | 8 each | Every 0.04 simulated seconds |

TTFT is first token completion minus arrival; queue delay is Prefill start minus
arrival; total latency is finish minus arrival. Token throughput divides actual
Fake Worker token count by elapsed simulated time from first arrival to last
completion, including idle gaps. Batch size is decode-call group size, averaged
over calls rather than weighted by elapsed time. Each scenario/strategy has one
deterministic simulation, not a statistical sample of hardware runs.

### 3. Scheduler Research

[Runner](../benchmarks/runtime/scheduler_research/runner.py) reuses `RequestQueue`,
`PhaseAwareBatchPolicy` and `FairPhaseBatchPolicy`. The model is a single-slot,
non-preemptive admission simulation; phase labels remain fixed and every selected
request runs to completion. It does not implement per-token scheduling.

- FIFO preserves arrival order.
- Phase Priority selects DECODE before PREFILL before WAITING; FIFO within phases.
- Fair Aging scores weights 100/50/10 plus creation age multiplied by 5 per second.
- Start time is 0; starvation threshold is 10 simulated seconds, strictly exceeded.
- Synthetic TTFT = queue delay + 0.1 * service time; total latency = wait + service.
- P50/P95/P99 use linear interpolation over completed request waits.
- Fairness is the Jain index on inverse slowdown `x = service / total_latency`:
  `(sum(x))^2 / (n * sum(x^2))`. Equal slowdown scores 1, even with excessive waits.

| Scenario | Requests per policy | Workload |
| --- | ---: | --- |
| decode_flood | 11 | Nine Decode jobs (5 s each), one later Waiting job (1 s), one Prefill job (2 s); later Decode jobs arrive every 3 s |
| mixed_length | 9 | All arrive at 0; WAITING/PREFILL/DECODE repeat; service lengths 1, 8, 2, 16, 3, 1, 12, 4, 2 s |
| continuous_arrival | 18 | Every 2 s; repeating phases; service time `3 + index % 4` s |

These are different workloads and timing definitions from the continuous-batch
experiment, despite shared scenario names. Their latencies are not interchangeable.

## Results

Values below are extracted from the saved artifacts; decimals are rounded to six places.

### Paged KV capacity

| Strategy | Requested tokens | Allocated tokens | Wasted tokens | Blocks | Utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contiguous | 12,288 | 30,720 | 18,432 | 30 | 40% |
| Paged | 12,288 | 12,288 | 0 | 768 | 100% |

Allocated token capacity after cleanup: **0**.

### Continuous batch (virtual seconds)

| Scenario | Strategy | Avg TTFT | Avg queue | Avg total | Tokens | Simulated tokens/s | Avg/max decode batch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| short_requests | Sequential | 0.255000 | 0.210000 | 0.270000 | 16 | 33.333333 | 1.000/1 |
| short_requests | Continuous | 0.150000 | 0.060000 | 0.180000 | 16 | 66.666667 | 4.000/4 |
| mixed_length | Sequential | 0.705000 | 0.660000 | 0.847500 | 84 | 56.000000 | 1.000/1 |
| mixed_length | Continuous | 0.206250 | 0.128750 | 0.491250 | 84 | 93.333333 | 2.625/4 |
| continuous_arrival | Sequential | 0.430000 | 0.385000 | 0.535000 | 64 | 53.333333 | 1.000/1 |
| continuous_arrival | Continuous | 0.171875 | 0.115625 | 0.436875 | 64 | 84.210526 | 3.200/4 |

Both strategies completed all eight requests in each scenario; every saved
`idle_after_test` value is true.

### Scheduler research (virtual seconds)

| Scenario | Policy | Avg queue | P50 | P95 | P99 | Avg TTFT | Avg total | Starvation count | Fairness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_flood | FIFO | 9.454545 | 9.000000 | 18.000000 | 18.800000 | 9.890909 | 13.818182 | 5 | 0.744874 |
| decode_flood | Phase Priority | 14.636364 | 10.000000 | 44.500000 | 45.700000 | 15.072727 | 19.000000 | 5 | 0.673605 |
| decode_flood | Fair Aging | 12.545455 | 12.000000 | 27.500000 | 34.300000 | 12.981818 | 16.909091 | 6 | 0.661603 |
| mixed_length | FIFO | 22.111111 | 27.000000 | 45.400000 | 46.680000 | 22.655556 | 27.555556 | 6 | 0.500916 |
| mixed_length | Phase Priority | 13.000000 | 13.000000 | 30.600000 | 35.720000 | 13.544444 | 18.444444 | 5 | 0.674233 |
| mixed_length | Fair Aging | 13.000000 | 13.000000 | 30.600000 | 35.720000 | 13.544444 | 18.444444 | 5 | 0.674233 |
| continuous_arrival | FIFO | 20.055556 | 20.500000 | 40.150000 | 40.830000 | 20.494444 | 24.444444 | 13 | 0.563042 |
| continuous_arrival | Phase Priority | 18.944444 | 15.000000 | 49.000000 | 49.000000 | 19.383333 | 23.333333 | 10 | 0.590596 |
| continuous_arrival | Fair Aging | 19.833333 | 20.500000 | 43.000000 | 43.000000 | 20.272222 | 24.222222 | 12 | 0.553953 |

Every policy completed 11/11, 9/9 and 18/18 requests respectively, and all final
idle flags are true. The threshold counts are not permanent starvation events.

## Analysis

1. **Paging avoids maximum reservation in this workload.** Paged accounting reserves
   only occupied, page-rounded capacity. Zero waste results from lengths all being
   multiples of 16; it is not a general property of paging. No real GPU byte
   reduction is demonstrated.
2. **Continuous coordination benefits from the configured cost model.** Grouping
   amortizes a nonzero per-call base cost, improving simulated throughput and mean
   latency in these three workloads. This does not predict real tensor-batch speedup;
   purely per-request linear costs need not produce a throughput improvement.
3. **Aging involves tradeoffs.** In decode flood, Fair Aging reduces P99 versus
   Phase Priority (45.7 to 34.3 s), but threshold exceedances increase (5 to 6)
   and fairness decreases (0.673605 to 0.661603). FIFO has the lowest P99 here.
4. **Means and tails can disagree.** For continuous arrival, Phase Priority has
   lower mean queue delay than FIFO but higher P99 (49 versus 40.83 s). Fair Aging
   reduces that P99 to 43 s, still above FIFO, and scores lower on this fairness metric.
5. **Aging is not always a ranking change.** All mixed-length jobs arrive together,
   so their aging bonus is equal. Fair Aging and Phase Priority consequently match.

## Limitations

- One deterministic workload per case; no confidence intervals or independent
  real-hardware repetitions. Small request sets limit tail interpretation.
- No GPU, model, attention kernel, network or tokenizer performance is measured.
- Memory simulation omits tensor metadata, allocator overhead, fragmentation,
  prefix sharing and actual device capacity. Paging can waste more than an exact
  contiguous reservation when the final page is only partly occupied.
- Continuous coordination uses assumed phase costs; no production GPU continuous
  batching or speedup over vLLM is claimed.
- Scheduler simulations are finite and non-preemptive with fixed phase labels;
  completion does not prove starvation freedom under sustained unbounded arrivals.
- Jain fairness measures relative slowdown, not token service shares or guaranteed QoS.
- These artifacts contain no source commit SHA. Hashes aid provenance but do not
  establish correspondence with a published source revision.
- These results must not be mixed with the separate historical real CUDA/HTTP
  measurements in [the evidence summary](../benchmarks/reports/benchmark-summary.md).

## Reproduction

From the project root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.paged_memory.report
.\.venv\Scripts\python.exe -m benchmarks.runtime.continuous_batch.report
.\.venv\Scripts\python.exe -m benchmarks.runtime.scheduler_research.report
```

These commands regenerate the local JSON/Markdown files and replace prior output
at the same paths. Preserve artifacts first if their recorded hashes are needed.
No model download or GPU connection is required. Reproduction commands were not
executed as part of this documentation-only consolidation.
