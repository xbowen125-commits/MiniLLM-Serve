# Runtime Scheduler Stress Comparison

## Design and reproduction

Results below were obtained by running the deterministic virtual-clock workloads
on 2026-10-12. Source: `benchmarks/runtime/stress_scenarios.py` and the shared
`SchedulerComparisonRunner` in `scheduler_compare.py`.

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.stress_scenarios
```

The command prints JSON with all request queue delays, selected order, and
service-count distributions; it does not modify historical reports or results.
Each policy receives fresh request states and identical input. Virtual time
starts at 0, aging_factor=1, starvation threshold=10 seconds, one non-preemptive
service slot. Every request is selected once and then completes after a fixed
synthetic service duration. FIFO orders eligible requests by arrival; Fair uses
the actual project FairPhaseBatchPolicy. No model, GPU, HTTP, or token decode
executes. A request phase is a fixed workload label, not a real model transition.

### Workloads

- Decode Flood: 20 DECODE requests arriving at `2*i` seconds, each serviced for
  20 seconds; 4 WAITING requests arriving at `5+i` seconds, serviced for 1 second.
- Mixed Length: 12 requests, arrivals `i` seconds, cycling WAITING/PREFILL/DECODE.
  Durations `[1,20,2,15,3,10,4,8,5,6,2,12]` seconds represent generation-length
  heterogeneity; actual generated token counts are not measured.
- Continuous Arrival: 30 requests arriving every 0.5 seconds, cycling the same
  phases, service duration `1 + i % 4` seconds. This is a finite arrival stream.

## Recorded results

All queue delays and durations below are synthetic seconds. Values are rounded
to six decimal places; JSON output retains full precision.

| Scenario | Policy | Completed | Average queue | Max queue | P50 queue | P95 queue | Starvation (>10s) | Completed/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode Flood | FIFO | 24 | 157.666667 | 346 | 139 | 325.3 | 23 | 0.059406 |
| Decode Flood | Fair | 24 | 208.333333 | 395 | 207 | 395 | 23 | 0.059406 |
| Mixed Length | FIFO | 12 | 37.083333 | 65 | 40.5 | 64.45 | 10 | 0.136364 |
| Mixed Length | Fair | 12 | 36.416667 | 73 | 36.5 | 72.45 | 10 | 0.136364 |
| Continuous Arrival | FIFO | 30 | 27.783333 | 56.5 | 27.25 | 54.425 | 23 | 0.410959 |
| Continuous Arrival | Fair | 30 | 27.016667 | 56 | 28.75 | 55.5 | 19 | 0.410959 |

Simulation durations are 404, 88, and 73 seconds respectively for both policies.
The per-request `service_count_distribution` is one service for each request:
24 entries for Flood, 12 for Mixed Length, and 30 for Continuous Arrival, under
both policies. This distribution checks completion coverage; it cannot measure
token-step fairness in this non-preemptive model.

## Metric definitions

- Queue delay: selection time minus arrival time; average/max include every
  completed request, including all phases.
- P50/P95: linear interpolation at `(n-1)*p/100` in sorted queue delays.
- Completed per simulated second: completed requests divided by elapsed virtual
  time from scenario start through final completion, including any idle gaps.
- Starvation count: number of unique requests with queue delay strictly above
  the threshold. All requests complete; this is not proof of indefinite starvation.
- Service count distribution: request ID to number of whole-request selections.

## Analysis and limits

Fair worsens average and long-tail waiting in Decode Flood: short WAITING jobs
are postponed behind long DECODE jobs. Mixed Length has a slightly lower average
but a higher maximum and P95. Continuous Arrival reduces threshold exceedances
and maximum delay, while increasing P50/P95. No uniform Fair improvement follows.

Equal throughput is expected because the same work completes on one continuously
busy slot; priority changes order, not service cost. With linear aging shared by
all requests, relative scores of already-queued requests do not change merely
because time advances. Arrival-age differences can overcome phase weights, but
aging is not a strict waiting-time guarantee.

There is one deterministic input per scenario per policy. Repeating it reproduces
logical values, not independent statistical measurements. Tail percentiles for
12–30 requests are descriptive, not production capacity evidence. Requests are
never resumed between tokens, and there are no memory constraints or concurrency
effects. These bounded admission stress simulations do not implement Continuous
Batching and are distinct from historical engine/GPU/HTTP performance evidence.
