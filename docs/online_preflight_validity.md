# Mission 24.4.2 — Online correctness preflight and validity evidence

The subsequent [workload shape proof](workload_shape_coverage.md) specifies the
versioned manifest and independent replay. This historical evidence is unchanged
and does not acquire new shape-coverage status retroactively.

## Scope

This document separates historical CPU/Fake transport evidence from GPU
correctness preflight and from a future formal performance experiment. The
historical run below is **not an accepted performance comparison**. It contains
deliberate failures and exceeds the frozen **25 ms** client send-lateness
tolerance. Neither records nor thresholds were rewritten during this audit.

The [frozen protocol](online_benchmark_protocol.md) and
[online HTTP framework](../benchmarks/runtime/online_http/README.md) define the
common trajectory, modes and metric boundaries. CPU/Fake results do not establish
CUDA performance, production capacity or a speedup.

## Historical evidence and integrity

Audited on 2026-10-20, using the existing ignored directory:

`benchmarks/runtime/results/online_http/run-20261020T151132-0cbe5afef462/`

The run records:

- Timestamp: `2026-10-20T15:11:33.100739+08:00`.
- HEAD: `dc9db6e9cb2585bd7eb141717206586ec0e01fb3`; dirty worktree: `true`.
- Source-tree fingerprint:
  `dfca21fbfeb7f93f236f9ca9cfe109b6eb7649429f74fa1d35a0703b2f239ff6`.
- Canonical configuration SHA-256:
  `ebc98e1e861753178b03363e07f057d28aef60981cb22ee36e77636cf717c8f0`.
- Prepared trace SHA-256:
  `14f303081fa2dd47e5bbb0318a08eac7c330e6ebe6156dd17e754b7f37a9d804`.
- `evidence_kind=cpu_fake`, `device=cpu`, `dtype=float32`, client concurrency 2,
  admission limit 2, one warmup per trial, and LC/CL process order.
- All **37** artifact hashes listed by `artifacts.json` matched the existing
  files when rechecked. `source_unchanged=true` is the historical run's recorded
  source-consistency result, not a claim that the current worktree is clean.

Selected file SHA-256 values, for independently locating the evidence:

| Relative artifact | File SHA-256 |
| --- | --- |
| `summary.json` | `229cdc1947059ec836e2a9bfef5cc199eba136086c89d6f2e3b1a287e4c85158e` |
| `config.json` | `ed70c3eeda7514893a78187453324ba37a1b0670ed32779cd5df08bfcfa192e2` |
| `trace.json` | `901163637a5e5162ccc7f99b5dbce8bc2a93854a0903dffc3a3e702e935eed32` |
| `trial-000-legacy/requests.jsonl` | `a9dd1e224093039fb8fab9f785df12e9a6ade9efc5e8a597f0b75243e31c2f58` |
| `trial-001-continuous/requests.jsonl` | `004d00900bee0c214111d8892d47ca1002ca649599e07d828811bae9ffe5841f` |
| `trial-002-continuous/requests.jsonl` | `b2adeba74faa490eed8c05fc8d98fb2b03f306ea662a6444bdd18d41c82f7889` |
| `trial-003-legacy/requests.jsonl` | `d1ee050a408c460be6f86178018c7b79769d95e3169712c120fed7829b30b857` |

The file hash of the formatted `config.json` is intentionally different from
the canonical configuration identity. The artifact manifest binds the former;
the protocol binds the latter.

## Phase separation

Durations are from the existing root `summary.json`, in milliseconds. Each
replay has its own phase-relative monotonic origin. Warmup timestamps cannot be
subtracted from formal timestamps as though they shared one origin.

| Trial | Startup | Warmup replay | Warmup drain | Formal replay | Formal drain | Shutdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `000-legacy` | 2446.505 | 83.967 | 19.264 | 203.014 | 26.435 | 474.368 |
| `001-continuous` | 2448.470 | 59.588 | 15.152 | 96.691 | 11.070 | 419.120 |
| `002-continuous` | 2429.505 | 45.962 | 0.872 | 140.999 | 1.499 | 423.433 |
| `003-legacy` | 2459.987 | 68.479 | 1.353 | 176.463 | 25.191 | 429.154 |

All four warmups completed successfully. Their send-lateness values were
respectively **0.565, 0.637, 0.718 and 1.030 ms**, each below 25 ms. The failed
send-lateness gate therefore concerns formal request replay, not a model-load,
warmup or drain duration accidentally used as request send time.

## Formal request chronology

Values below are the existing JSONL fields converted from seconds to
milliseconds and rounded for display. The raw timestamps remain authoritative.
JSONL is in completion order; `trace_ordinal` reconstructs planned arrival order.

| Trial | Request | Planned send | Actual send start | First visible content | Terminal event | Send lateness | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `000-legacy` | `short-01` | 0 | 1.798 | 17.323 | 79.595 | 1.798 | completed |
| `000-legacy` | `long-01` | 0 | 2.855 | 33.105 | 103.886 | 2.855 | completed |
| `000-legacy` | `short-02` | 20 | 79.838 | 107.339 | 170.307 | **59.838** | completed |
| `000-legacy` | `failure-01` | 30 | 104.191 | N/A | 203.014 | **74.191** | error |
| `001-continuous` | `short-01` | 0 | 0.534 | 50.041 | 54.467 | 0.534 | completed |
| `001-continuous` | `long-01` | 0 | 1.105 | 36.670 | 50.350 | 1.105 | completed |
| `001-continuous` | `short-02` | 20 | 50.670 | 96.406 | 96.691 | **30.670** | completed |
| `001-continuous` | `failure-01` | 30 | 54.752 | N/A | 84.483 | 24.752 | error |
| `002-continuous` | `short-01` | 0 | 0.993 | 33.266 | 40.627 | 0.993 | completed |
| `002-continuous` | `long-01` | 0 | 1.614 | 58.371 | 75.021 | 1.614 | completed |
| `002-continuous` | `short-02` | 20 | 40.994 | 83.098 | 89.613 | 20.994 | completed |
| `002-continuous` | `failure-01` | 30 | 75.438 | N/A | 140.999 | **45.438** | error |
| `003-legacy` | `short-01` | 0 | 1.294 | 7.049 | 68.306 | 1.294 | completed |
| `003-legacy` | `long-01` | 0 | 2.108 | 30.753 | 91.301 | 2.108 | completed |
| `003-legacy` | `short-02` | 20 | 68.961 | 96.025 | 161.009 | **48.961** | completed |
| `003-legacy` | `failure-01` | 30 | 92.096 | N/A | 176.463 | **62.096** | error |

The four maxima are **74.191, 30.670, 45.438 and 62.096 ms**. Thus **all four**
trials fail the 25 ms gate, not only the legacy trials. Six of the sixteen formal
requests exceed the threshold. All sixteen formal rows record
`queue_wait_ms=0.0`; that server Scheduler admission observation is distinct from
the nonzero client-side delay before a POST was started.

Each trial also retains the deliberately injected `failure-01`: legacy records
`internal_error`, and continuous records `generation_failed`. These are failures
even though the SSE HTTP status is 200. There are twelve completed formal
requests and four errors, not sixteen successful performance samples.

## What the available evidence can explain

The benchmark client uses a bounded `ThreadPoolExecutor` with two workers. Each
worker remains occupied while it observes its HTTP response through DONE or
error and performs post-response accounting. The first two requests start near
time zero and have not reached their terminal events at 20 or 30 ms. Therefore
there is no free transport worker when the two later requests are due.

Implementation sources are [`replay()`](../benchmarks/runtime/online_http/runner.py)
and [`observe_request()`](../benchmarks/runtime/online_http/client.py). At audit
time both files matched their historical `source-manifest.json` hashes, so this
interpretation is tied to the code recorded for the run, not just to filenames.

The observed dispatch sequence is consistent with that capacity constraint:

| Trial | First two terminal times, chronological order (ms) | Later actual send starts (ms) | Send after preceding terminal (ms) |
| --- | --- | --- | --- |
| `000-legacy` | 79.595, 103.886 | 79.838, 104.191 | 0.242, 0.304 |
| `001-continuous` | 50.350, 54.467 | 50.670, 54.752 | 0.319, 0.285 |
| `002-continuous` | 40.627, 75.021 | 40.994, 75.438 | 0.366, 0.417 |
| `003-legacy` | 68.306, 91.301 | 68.961, 92.096 | 0.655, 0.795 |

This identifies a necessary, observable client-capacity bottleneck. It does
**not** determine every component of the total lateness: no generator-submit or
worker-start timestamps were saved in these old rows. The timestamps cannot
prove exactly when the main replay thread woke or submitted each task, or how
much additional delay came from OS scheduling, thread-pool dispatch, serialization
or post-response tokenization. The full causal decomposition remains **UNKNOWN**.

The legacy rows also show visible-content-to-DONE tails (for example,
`short-01`: last content 25.459 ms, DONE 79.595 ms in trial 000). These tails hold
the client transport task open. The historical observations do not localize that
tail to a particular server finalizer, network mechanism or client read operation.
It would be speculation to attribute it to timer granularity, CUDA, an event-loop
stall or a specific operating-system mechanism.

### Missing evidence

The historical rows do not record:

- Replay-generator wake and pool-submit timestamps.
- Per-task worker-start and body-serialization start/end timestamps.
- Client clock implementation/resolution at runtime or sleep overshoot samples.
- Server TCP ingress, HTTP parser arrival or per-request runtime execution times.
- A complete thread/event-loop timeline or per-read socket timing.

The present audit does not retrofit any of these into existing records. Future
instrumentation may measure them, but new observations cannot identify missing
historical events. `sent_at_s` remains the original send-call timing boundary;
planned arrivals are not moved to actual worker starts to make lateness disappear.

## Resource and process evidence

Every trial's post-replay status has Scheduler `running_requests=0` and
`queued_requests=0`. Both continuous trials have zero allocated slots, pending
requests, runtime active requests, contexts, claims, released/finished ID history,
pending commands, stream consumers and finalizers after drain. Their owner thread
is alive while the server is running, as expected; each `server-stopped.json`
records `owner_thread_alive=false` and zero finalizers/consumers after shutdown.
All four server processes exited with code zero.

Legacy has no continuous SlotKVPool telemetry; unavailable slot metrics must
remain N/A rather than being invented as zero. Continuous lifetime peak slots
were 2 and 1 respectively. These include warmup and do not prove formal batch
shape coverage; periodic samples are lower bounds and may miss brief overlap.

There were 6, 4, 4 and 6 resource-sampling rows, with no recorded probe errors.
This sampling frequency is not sufficient to reconstruct every request or thread
transition. All four server metadata records identify the CPU device, and all
status GPU records say `available=false`, so this audit is not a GPU experiment.

## Consequence for the next validity gate

The old smoke configuration is intentionally adversarial to gate logic and
remains invalid. A new formal configuration must not mix deliberate failure
injection into accepted performance samples. It must preserve the 25 ms rule,
retain every failed/late row and provide the separate preflight and warmup
coverage evidence required by the updated framework. Increasing a client limit,
altering a workload or changing a warmup rule requires a new frozen config hash;
none of those changes can rehabilitate this historical experiment.

No GPU preflight or performance result is asserted by the historical evidence
above. Actual new preflight execution and its machine-readable report must be
reported separately, with the precise checks that passed, failed or were not run.

## Independent correctness entry point

`benchmarks.runtime.online_http.preflight` defaults to plan-only. Explicit
`--execute --allow-gpu --scope single` starts two independent real HTTP service
processes, one per mode, and sends only the first frozen request once per mode.
It preserves the original prompt, immutable revisions, float32/CUDA and original
16-Token limit; EOS can still end generation early. No throughput workload is
replayed. The full workload contains six requests, so this scope cannot prove
all six references or concurrency two.

Read-only, process-local observers copy terminal RequestState token IDs, including
EOS, and bind each HTTP result to a fresh backend state. Old terminal events and
duplicate request identities cannot substitute for a new result. Raw records
and the summary carry identical token annotations. Legacy cache-reference release
is observed after the generation runner's finally block. GPU slot/Context cleanup
comes from genuine owner/status snapshots, not clearing dictionaries.

Forward attempts and completions have correlated call IDs. Failed attempts remain
visible but do not count as successful observed shapes. Python forward-return
time is not CUDA kernel completion time; this observer adds no synchronization
and is not a GPU timing instrument. Exact generated IDs are read after normal
request completion. The phase classifier based on input width is ambiguous for a
one-token prompt; frozen chat prompts exceed one token.

Single mode leaves concurrency two, injected errors, disconnect cancellation and
active shutdown `UNKNOWN`. Passive shutdown has its own check; it is not relabelled
as an active-shutdown test. Full mode implements those bounded probes and has
CPU/Fake coverage, but must not be described as real GPU evidence unless run.
Every fault row is explicitly marked `performance_sample=false`.

## Warmup coverage gate

The frozen formal runner sends warmups sequentially. Two successful requests
with one client do not prove Continuous batch=2 warmup. Required logical shape
categories distinguish mode, Prefill/Decode and batch cardinality. Successful
diagnostic shape observations are retained separately from actual formal-process
warmup. Input/mask dimensions are recorded; exhaustive padded-width and per-request
KV-length coverage remains `UNKNOWN`. Coarse B=1/B=2 coverage is necessary but
not sufficient, so the gate also requires explicit full tensor-shape proof and
`formal_warmup_coverage=true`. The current configuration cannot meet this gate.

## Machine-readable validity gates

Every check records `{code, status, evidence, detail}` where applicable. Missing
evidence is UNKNOWN, not PASS. The principal reason codes are:

| Reason code | Requirement |
| --- | --- |
| `PREFLIGHT_MISSING` / `PREFLIGHT_DECLARED_PASS` | A complete PASS report must exist |
| `PREFLIGHT_IDENTITY_BOUND` | Exact config, prepared trace, HEAD, dirty flag and source-tree fingerprint |
| `PREFLIGHT_ENVIRONMENT_BOUND` / `SERVER_MODEL_PROVENANCE` | Current controls and actual loaded server provenance agree |
| `PREFLIGHT_ARTIFACT_INTEGRITY` | Report, frozen config and raw sidecar checksums match; paths cannot escape |
| `PREFLIGHT_<check>` | CUDA, pinned assets, output correctness, both concurrency levels and all lifecycle checks pass individually |
| `PREFLIGHT_WARMUP_COVERAGE` / `TRIAL_WARMUP_COVERAGE` | Real process-local logical and tensor shapes are warmed |
| `ALL_OUTPUT_REFERENCES_FROZEN` / `OUTPUT_REFERENCE_MATCH` | Every request has its own retained expected output/count/finish reason |
| `RAW_FIELDS_COMPLETE` / `RAW_IDENTITY_BOUND` / `RAW_TIMELINE_VALID` | All required records, identities and monotonic stage boundaries exist |
| `CLIENT_SEND_ON_TIME` | Actual recorded send lateness does not exceed 25ms |
| `ALL_REQUESTS_COMPLETED` / `WIRE_PROTOCOL_VALID` | No hidden failed, timed-out, cancelled or truncated success sample |
| `AUTHORITATIVE_OUTPUT_TOKENS_UNAVAILABLE` / `..._MISMATCH` | Actual output counts must exist and match the protocol; visible text is insufficient |
| `FINAL_RESOURCES_ZERO` | Lease/Runtime/slot cleanup and process shutdown are confirmed |
| `APPEND_ONLY_ARTIFACTS` / `LOCAL_AND_PAIR_VALIDATION` | Exclusive output, complete journal, unchanged sources and paired checks hold |

Admission runs before any formal service process/request. Missing or partial
preflight, a changed dirty-source fingerprint, absent warmup proof, or current SSE
usage limitations block launch. `admission.json`, root `summary.json` and Markdown
record the rejection and zero executed trials. The API helper enforces this too;
calling the Python runner directly does not bypass the CLI gate.

Post-run validation does not delete any request. Diagnostic summaries remain
visible, but `formal_statistics` is populated only for an entirely eligible
trial. The current CPU/Fake smoke and frozen SSE configuration cannot populate
that field. SHA manifests prevent accidental missing/stale evidence, not a
maliciously forged report; no signed attestation is claimed.

## New dispatch evidence and reproduction

The original timing fields are unchanged. Future rows additionally record:

```text
planned arrival -> generator submit -> transport worker start
  -> body build start/end -> original send-start -> HTTP call -> response events

send lateness = generator lateness + client pool wait + client preparation
```

`http_call_at_s` identifies the actual Python HTTP call after client setup; the
original `sent_at_s` measurement boundary is not silently moved. Clock metadata
records implementation and nominal resolution, not actual scheduler wakeup
granularity. Client timings do not establish server ingress or GPU execution time.

```powershell
Set-Location D:\AI_Lab\MiniLLM-Serve
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Plan only, then the authorized bounded single-request correctness scope.
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.preflight `
  --config benchmarks/runtime/online_http/configs/qwen_protocol.json
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.preflight `
  --config benchmarks/runtime/online_http/configs/qwen_protocol.json `
  --execute --allow-gpu --scope single

# Non-PASS intentionally returns exit code 1 and retains all evidence.
# This is NOT permission to start the formal benchmark.
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
.\.venv\Scripts\python.exe -m compileall -q src tests benchmarks
git diff --check
```

New preflight reports use distinct ignored directories under
`benchmarks/runtime/results/online_http/preflight-*`. Inspect `preflight.json`,
per-mode `requests.jsonl`/`observer.jsonl`/`probe.json` and `artifacts.json` for
actual outcomes. Do not interpret the existence of this entry point as a real GPU
PASS. No full GPU preflight or formal online throughput experiment is authorized
or claimed by this document. Hosted CI, long soak and independent signal handling
remain separate unverified work.

## Local regression and bounded GPU probe scope

The final CPU/Fake non-slow regression on 2026-10-20 executed **591 passed,
13 deselected** in 55.36 seconds. It includes 59 pure validity-gate cases,
14 correctness-preflight cases, new dispatch/admission checks and existing HTTP
regressions. These CPU cases do not attest to real GPU concurrency or fault
cleanup. Compileall and `git diff --check` returned zero; the three protocol/audit
documents had no broken local links. The production source tree is not edited.

The environment identity query found Python 3.13.5, PyTorch 2.11.0+cu128,
CUDA runtime 12.8 and an available NVIDIA GeForce RTX 5060 Laptop GPU. Environment
availability is not itself a model correctness result.

The authorized GPU scope is the single-request command below. The new target is
exclusive; it is never reused or overwritten. Its machine-readable report, raw
observations and checksums determine actual outcomes; this document does not
predeclare a GPU PASS. A non-PASS report must continue to block formal execution.

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.preflight `
  --config benchmarks/runtime/online_http/configs/qwen_protocol.json `
  --execute --allow-gpu --scope single `
  --output-dir benchmarks/runtime/results/online_http/preflight2442-single-gpu-20261020-a
```

The GPU run produces no performance samples. Full-workload references,
concurrency-two and injected lifecycle faults require separate execution before
they can be promoted from UNKNOWN. Complete process-local warmup proof and
authoritative streamed Token usage remain mandatory even if single-request
greedy output and normal cleanup pass.
