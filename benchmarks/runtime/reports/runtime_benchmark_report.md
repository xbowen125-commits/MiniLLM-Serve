# Runtime Benchmark v2 Report

## Experiment setup

Deterministic virtual-clock admission simulation; one non-preemptive slot.
Each row represents one finite workload run. No real model or GPU executes.
Complete configurations and UTC run timestamps are recorded in JSON.

Reproduce from the project root:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.report_generator
```

## Runtime architecture

Configuration → StressScenario → SchedulerComparisonRunner → FIFO / FairPhaseBatchPolicy.
This admission simulator uses request and phase contracts; it does not drive RuntimeFacade
or model execution. The separate FakeWorker benchmark exercises RuntimeFacade.

## Metrics definition

Queue delay = selection minus arrival, in virtual seconds. P50/P95 use linear interpolation.
Starvation count = unique delays strictly above the configured threshold.
Completed/s = completions / simulated duration (including idle gaps).
Service distribution counts whole-request selections, not token steps.

## Scenario description

### continuous_arrival

Run timestamp (UTC): 2026-10-12T11:51:06.673927+00:00

Config SHA-256: `89e4d6a43c258d3e9a2c2a156e6d2daca5b2c4437522c2289dd9132760014f2f`

```json
{
  "scenario": "continuous_arrival",
  "start_time": 0,
  "aging_factor": 1,
  "starvation_threshold": 10,
  "groups": [
    {
      "prefix": "arrival",
      "count": 30,
      "arrival_start": 0,
      "arrival_interval": 0.5,
      "phases": [
        "waiting",
        "prefill",
        "decode"
      ],
      "service_seconds": [
        1,
        2,
        3,
        4
      ]
    }
  ]
}
```

### decode_flood

Run timestamp (UTC): 2026-10-12T11:51:06.686275+00:00

Config SHA-256: `6d81ae22ad357352da3954d580155b98d850a737c4cdee8fc96ef5b4cff11c66`

```json
{
  "scenario": "decode_flood",
  "start_time": 0,
  "aging_factor": 1,
  "starvation_threshold": 10,
  "groups": [
    {
      "prefix": "decode",
      "count": 20,
      "arrival_start": 0,
      "arrival_interval": 2,
      "phases": [
        "decode"
      ],
      "service_seconds": [
        20
      ]
    },
    {
      "prefix": "waiting",
      "count": 4,
      "arrival_start": 5,
      "arrival_interval": 1,
      "phases": [
        "waiting"
      ],
      "service_seconds": [
        1
      ]
    }
  ]
}
```

### mixed_length

Run timestamp (UTC): 2026-10-12T11:51:06.692705+00:00

Config SHA-256: `ca4560ade2676d8dacbc88b2e6621a8ecb14c736ce18395cd5c4d9ef25130e8b`

```json
{
  "scenario": "mixed_length",
  "start_time": 0,
  "aging_factor": 1,
  "starvation_threshold": 10,
  "groups": [
    {
      "prefix": "mixed",
      "count": 12,
      "arrival_start": 0,
      "arrival_interval": 1,
      "phases": [
        "waiting",
        "prefill",
        "decode"
      ],
      "service_seconds": [
        1,
        20,
        2,
        15,
        3,
        10,
        4,
        8,
        5,
        6,
        2,
        12
      ]
    }
  ]
}
```

## Results summary

| Scenario | Policy | Completed | Avg queue (s) | Max queue (s) | P50 (s) | P95 (s) | Starvation | Completed/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| continuous_arrival | FIFO | 30 | 27.783333 | 56.500000 | 27.250000 | 54.425000 | 23 | 0.410959 |
| continuous_arrival | FairPhaseBatchPolicy | 30 | 27.016667 | 56.000000 | 28.750000 | 55.500000 | 19 | 0.410959 |
| decode_flood | FIFO | 24 | 157.666667 | 346.000000 | 139.000000 | 325.300000 | 23 | 0.059406 |
| decode_flood | FairPhaseBatchPolicy | 24 | 208.333333 | 395.000000 | 207.000000 | 395.000000 | 23 | 0.059406 |
| mixed_length | FIFO | 12 | 37.083333 | 65.000000 | 40.500000 | 64.450000 | 10 | 0.136364 |
| mixed_length | FairPhaseBatchPolicy | 12 | 36.416667 | 73.000000 | 36.500000 | 72.450000 | 10 | 0.136364 |

## Limitations

All durations are synthetic, not measured serving latency. Phases are fixed labels.
No continuous batching, GPU work, token scheduling, cache pressure, or HTTP is modeled.
Fair may worsen long-tail waiting. Identical repeated inputs reproduce logical metrics
rather than independent performance samples. No universal fairness improvement is claimed.
Timestamps vary between runs; configuration hashes and metrics should match.
Small finite samples do not establish production capacity or permanent starvation.
