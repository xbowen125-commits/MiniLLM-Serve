# Scheduler Policy Benchmark — virtual-clock simulation

## Experiment design and reproduction

Source: `benchmarks/runtime/scheduler_compare.py`, default workload and runner.
The results below were produced by executing the command on 2026-10-12.

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.scheduler_compare
```

This compares selection behavior, not real model performance. Each policy uses
fresh request states and the same virtual clock. One non-preemptive execution
slot completes an entire synthetic request before another is selected. Phases
are fixed input labels; no actual prefill, KV cache, or token decode occurs.
FIFO sorts eligible requests by arrival time. Fair calls the project's actual
FairPhaseBatchPolicy with aging_factor=1.0. Selection starts at virtual t=100 s.

| Request | Phase | Arrival (s) | Synthetic service duration (s) |
| --- | --- | ---: | ---: |
| waiting-old | WAITING | 0 | 5 |
| prefill | PREFILL | 90 | 3 |
| decode | DECODE | 99 | 1 |

Queue delay is selection time minus arrival time. Average and maximum include
all three completed requests. `starvation_count` counts unique requests whose
queue delay is strictly greater than 10 s; it does not mean indefinite starvation.
The deliberate initial backlog contributes to these values.

## Recorded results

| Policy | Completed | Average queue delay (s) | Max queue delay (s) | Starvation count (>10 s) |
| --- | ---: | ---: | ---: | ---: |
| FIFO | 3 | 41.333333 | 100 | 2 |
| FairPhaseBatchPolicy | 3 | 40.666667 | 100 | 2 |

| Policy | Selection order | Queue delays in selection order (s) |
| --- | --- | --- |
| FIFO | waiting-old → prefill → decode | 100, 15, 9 |
| FairPhaseBatchPolicy | waiting-old → decode → prefill | 100, 6, 16 |

## Difference analysis and limitations

At t=100, scores are WAITING=110, PREFILL=60, DECODE=101. Aging therefore keeps
the oldest waiting request first. Fair then selects decode ahead of prefill.
Compared with FIFO, decode delay decreases from 9 to 6 s, while prefill delay
increases from 15 to 16 s. Average delay decreases by 2/3 s, maximum delay is
unchanged, and starvation count at the default threshold is unchanged.

With a threshold of 15 s, FIFO has one threshold exceedance and Fair has two.
This sensitivity is covered by tests and demonstrates that Fair is not uniformly
better. The simulation does not prove a maximum waiting-time bound or benefits
under sustained arrivals. There is one deterministic scenario per policy;
repeating it reproduces the same logical values, not independent performance
samples. No GPU, HTTP, continuous batching, or token-level scheduling is tested.
All durations are synthetic seconds, not measured wall-clock latency. This
report is separate from historical engine, HTTP, and GPU benchmark evidence.
