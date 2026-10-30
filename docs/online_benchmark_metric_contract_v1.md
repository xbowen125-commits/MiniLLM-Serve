# Online benchmark metric evidence contract v1

Specification version: **1.0**, Mission 24.4.2.45, 2026-10-30.
Source baseline: `21b130fc954348b33f8f286644d8a9fce6cdf3ab`.
This is a documentation specification, not an implemented schema migration,
an admission certificate, or experimental evidence. Formal online GPU Benchmark
remains **NOT_ADMITTED**. Existing PREFLIGHT FAIL/UNKNOWN, research-contract
obligations and the frozen 25 ms dispatch-lateness threshold remain effective.

## Implementation and evidence boundaries

Implemented: the [benchmark host](../benchmarks/runtime/online_http/server.py)
installs the opt-in [usage adapter](../benchmarks/runtime/online_http/usage.py).
The [client](../benchmarks/runtime/online_http/client.py) requests
`X-MiniLLM-Benchmark-Usage: 1` and parses `event: minillm.usage` independently
of content. [Existing gate checks](../benchmarks/runtime/online_http/gate.py)
validate count source, terminal status, identifiers and inclusion flags.
Ordinary application responses and standard chat schema are unchanged.

The adapter reads terminal RequestState counters; it does not generate tokens.
`generated_output_tokens` includes generated EOS, stop and hidden tokens present
in that list. It is not the visible text token count or the generation limit.
Failed/cancelled terminal counts do not become completed throughput.
Disconnected clients may receive no final event. Their accounting requires an
independent identity-bound terminal journal; absent evidence stays UNKNOWN.
Usage supplies neither per-token timestamps nor resource-release confirmation.

Proposed: explicit metric aliases, the evidence envelopes and reason codes below,
complete negative validation, and interval-scoped memory evidence. These are not
claims that current code implements every requirement. Current missing evidence
includes exact per-generated-token timing and complete current-process admission
proof. Historical artifacts retain their original schemas and limitations.

## Common evidence representation and compatibility

Each proposed field has a value plus status (`PASS`, `FAIL`, `UNKNOWN`, or
`NOT_APPLICABLE`), reason code and raw source references (file hash, event/line).
Include clock domain, PID/device, phase and measurement scope where applicable.
Unavailable or inapplicable values are JSON `null`, never invented zero; zero
is valid only when directly evidenced. Preserve malformed raw values separately.
Contradictory evidence is FAIL; missing evidence is UNKNOWN. These statuses do
not override the existing gate or its result precedence.

Existing [online metrics](../benchmarks/runtime/online_http/metrics.py) remain
unchanged: `ttft_ms` is client-visible TTFT, `tpot_ms` is visible-text TPOT proxy.
The proposed aliases below do not rename existing fields in saved reports.
Older HTTP/engine benchmark TPOT definitions are not silently reinterpreted.
Readers must select the artifact's schema and metric definitions explicitly.

The existing implementation field `completed_generated_output_tokens_per_s`
maps semantically to proposed contract field `completed_output_tokens_per_s`:
both mean authoritative completed generated tokens divided by the declared trial
window. This is a documented versioned alias, not an implemented output-schema
rename. Existing report consumers retain their interpretation; no historical
report migration occurs. Formula, population and token-count provenance must
accompany the alias rather than be inferred from the shorter name.

Let s, f, l, e be client send-start, first tokenizer-valid visible content,
last visible content and terminal observation, in seconds from one monotonic
phase origin. N is authoritative generated count; V is final visible text
retokenized by the pinned tokenizer. W is the formal replay origin to the last
client terminal observation, including arrival gaps and request queueing but
excluding model loading, warmup and subsequent resource drain/shutdown.

## Field contract

The eligibility column is conditional on all unchanged admission gates passing.
It never independently permits an experiment.

| Field / unit | Clock and boundary | Raw evidence and validity | Missing / invalid behavior | Formal eligibility |
| --- | --- | --- | --- | --- |
| `generated_output_tokens` / token | Counter, no clock; terminal generation list | Server JSON usage or validated benchmark SSE usage; nonnegative integer, not bool; identity, terminal source and inclusion flags verified | null/UNKNOWN if absent; conflict or invalid type FAIL | Completed throughput only with valid complete accounting |
| `visible_output_tokens` / token | Counter, no clock; complete visible response | Final text, tokenizer revision and tokenization rules; nonnegative integer | null/UNKNOWN; invalid count FAIL | Visible-text proxies only, never substitutes for N |
| `client_visible_ttft_ms` / ms | Client monotonic; 1000(f-s) | Raw send and first valid text timestamps; same clock, ordered, finite | No content or nonstream: null with reason; invalid ordering FAIL | Comparable client-visible streaming boundary, not internal TTFT |
| `client_visible_tpot_proxy_ms` / ms | Client monotonic; 1000(l-f)/(V-1) | Valid content timestamps and V; excludes role/finish/usage/DONE | V<=1: null/NOT_APPLICABLE; missing evidence UNKNOWN; invalid ordering FAIL | Proxy-labelled comparison only |
| `server_generated_tpot_ms` / ms | Server monotonic with documented CUDA-completion boundaries; 1000(t_last-t_first)/(N-1) | Identity-bound first/last generated-token completion records and count; no cross-clock subtraction | N<=1: null/NOT_APPLICABLE; current missing timing UNKNOWN; contradictions FAIL | Only with exact timing evidence; usage alone insufficient |
| `e2e_ms` / ms | Client monotonic; 1000(e-s) | DONE, full JSON receipt or recorded error/timeout/cancellation; finite ordered timestamps | Missing UNKNOWN; impossible timing FAIL | Separate completed and all-attempted populations |
| `completed_requests_per_s` / request/s | Client phase monotonic; completed length-valid count/W | All planned rows, protocol outcome and exact W endpoints; W>0 | Missing window/rows UNKNOWN; invalid window FAIL | Full valid trial, retain failures in outcome denominators |
| `completed_output_tokens_per_s` / token/s | Same W; sum(N for completed length-valid requests)/W | Valid authoritative counts for the entire completed population | Any required N missing: null/UNKNOWN; conflicts FAIL; no known-only substitute | Full valid trial, not sum of per-request rates |
| `queue_wait_ms` / ms | Server monotonic; queue entry to admission | Authority-provided queue-wait header/journal and request mapping | Missing null/UNKNOWN; negative/nonfinite FAIL | Same queue boundary; never client dispatch lateness |
| `gpu_peak_allocated_bytes` / byte | CUDA process/device peak counter; formal phase | Reset/read records, process/device identity, synchronization and allocator API | Missing reset: interval peak UNKNOWN; preserve labelled lifetime peak separately | Matching phase and scope only |
| `gpu_peak_reserved_bytes` / byte | Same phase and device; reserved allocator counter | Independent reserved counter with same reset/read evidence | Missing UNKNOWN; invalid counter FAIL | Never relabel as allocated or KV size |
| `device_utilization_pct_samples` / percent sequence | Collector monotonic and device sample interval | Device-level sampler, timestamps, interval, GPU ID, finite values in [0,100] | Missing samples null/UNKNOWN, not zero; invalid values FAIL | Contextual device telemetry, not per-request or kernel attribution |
| `measurement_provenance` / object | Multiple named clocks, never implicitly equated | Commit/tree and dirty-source manifest, config/trace hashes, PID/run, model/tokenizer revisions, dtype/device, versions, collector version, artifact hashes | Missing required identity UNKNOWN; conflicting identity FAIL | Mandatory for every formal comparison |

Report count and latency populations separately. P50/P95/P99 use linear
interpolation at sorted index (n-1)p, p in [0,1]; empty populations are null.
Sample standard deviation uses n-1 and is null for fewer than two samples.
Always preserve sample counts, missing counts, errors, timeouts, cancellations,
not-sent records and length mismatches. Freeze SLOs before execution, never after
observing results. Successful-only distributions do not erase failed requests.

## Cross-trial and session aggregation (specification, not implemented claims)

Every trial retains trial ID, session ID, source/workload identity, mode,
validity and failure reasons, window duration W, completed request count C,
authoritative completed generated-token count G, C/W, G/W, per-request latency
observations, and missing/invalid sample counts. Unknown G remains null.
Retain session identifiers and execution ordering, including alternating mode
order. State whether a summary is within one session or across sessions/trials.

Freeze comparison groups and eligibility rules before execution. Group by exact
source identity, model/tokenizer identity, device/dtype, mode, workload/config/
trace identity and metric semantics; do not pool incomparable workloads or
different definitions. Existing admission gates remain prerequisites. All trials
remain in the report with eligible/excluded counts and exclusion reasons.
Never exclude a trial merely because its performance is unfavorable.

For eligible trials, report a distribution of per-trial rates: each trial
contributes one equally weighted observation unless a different weighting policy
was separately declared in advance. Names include `trial_request_rate_mean`,
`trial_request_rate_median`, `trial_output_token_rate_mean` and
`trial_output_token_rate_median`. Report sample count and variability alongside
them. Do not silently weight by request count, duration or session size.

Pooled rates are separate metrics over declared eligible trials j:

- `pooled_completed_requests_per_s = sum(C_j) / sum(W_j)`.
- `pooled_completed_output_tokens_per_s = sum(G_j) / sum(W_j)`.

Require finite positive windows and a nonempty eligible trial set. An arithmetic
mean of trial rates is not a pooled rate. Missing required G in any trial within
the declared eligible population makes pooled token throughput null/UNKNOWN;
do not drop that trial or substitute zero. No eligible trials or invalid/missing
windows make the corresponding pooled rate unavailable. A valid positive window
with directly evidenced zero completions can legitimately yield zero throughput;
that is distinct from unavailable evidence.

Distinguish per-trial request-latency percentiles, the distribution of trial-level
percentiles, and pooled request-level percentiles. Compute pooled P50/P95/P99
directly from eligible individual request observations using the frozen linear
interpolation rule. Never average trial P95 values and call it pooled P95.
Specify the latency population (for example completed-only versus all-attempted),
retain request/session associations and report missing/invalid observations;
never silently omit them. Pooled request percentiles weight individual eligible
observations, whereas trial-percentile distributions weight trials equally.

Preserve failed, cancelled, timed-out and not-sent request records regardless of
aggregation eligibility. Across-session summaries must list constituent sessions
and ordering and retain per-session results; do not present pooled observations
as independent session replications. These are versioned contract requirements,
not a claim that the current report generator implements every aggregation.
No frozen workload or historical report is changed by these definitions.

## Usage and negative acceptance requirements (proposed checks)

Require one valid named usage event when the configuration demands it. Match
response ID, lease ID and engine request identity through direct associations;
validate terminal status, `count_source=terminal_request_state`, inclusion flags
and `generated_count_source=terminal_request_state_sse`. When all counts are
known, total must equal prompt plus completion. Unknown prompt keeps total null.
An identity match alone does not prove token values, timing or resource safety.

The following are specification reason codes, not claims of existing emitted codes.
Preserve original events and diagnostics, including conflicting duplicates.

| Negative case | Proposed reason code | Required fail-closed result |
| --- | --- | --- |
| Required usage absent | `USAGE_MISSING` | Count UNKNOWN; required accounting prevents admission |
| Duplicate, even identical | `USAGE_DUPLICATE` | Protocol FAIL; no silent deduplication |
| Conflicting counts/total | `USAGE_COUNT_CONFLICT` | Accounting FAIL; no preferred-value selection |
| Malformed payload, negative/bool/noninteger count | `USAGE_INVALID_TYPE` | FAIL; preserve raw payload, do not coerce |
| Response/lease/engine identity mismatch | `USAGE_IDENTITY_MISMATCH` | FAIL; do not attach to a nearby request |
| No first visible text | `VISIBLE_FIRST_MISSING` | TTFT null; missing required SLO evidence cannot qualify |
| V<=1 or N<=1 | `TPOT_NOT_APPLICABLE` | Corresponding TPOT null, not zero or division by zero |
| Chunk count substituted for token count | `TOKEN_SOURCE_INVALID` | FAIL; content-event counts stay diagnostic |
| CUDA execution unobserved | `CUDA_EXECUTION_UNKNOWN` | UNKNOWN; no real-GPU comparison eligibility |
| Batch above backend limit 2 | `BATCH_UNSUPPORTED` | FAIL; no silent fallback or concurrency-based inference |
| Required prefill/decode shapes missing | `SHAPE_COVERAGE_INCOMPLETE` | FAIL/UNKNOWN according to evidence; no admission |
| Interval memory reset unproved | `MEMORY_RESET_UNVERIFIED` | Interval peak UNKNOWN; lifetime/sampled data labelled separately |
| Cancellation cleanup not confirmed | `CLEANUP_UNCONFIRMED` | Drain not successful; preserve lease/slot/reference evidence |
| Model/source identity mismatch | `PROVENANCE_MISMATCH` | FAIL; historical certificate cannot authorize current run |

These cases require future synthetic acceptance tests before implementation can
be claimed complete. This document does not execute those tests or replace any
existing reason code, whitelist, preflight verdict or research proof obligation.

## CUDA execution and shape evidence

A future authorized preflight must bind actual model execution to current
run/PID/source/config/model/tokenizer identities and input/model CUDA devices.
Observe real Forward tensor batch dimensions for batch=1 and batch=2, never
infer them from client concurrency. Match runtime Forward UUID and observer call
ID explicitly. Capture prefill/decode input/mask shapes, padding widths, physical
KV tensor dimensions and per-request effective KV lengths separately.

Map actual request objects and IDs to rows, Slots and cache ownership. Demonstrate
initialization, decode growth and regrouping without treating shape equality as
content equality. Required manifest completeness and matched observations are
distinct obligations. UNKNOWN dimensions or unclosed shape space remain UNKNOWN.
Observe cancel intent/effect, Slot release and original RequestState reference
clearing independently. Terminal state, usage delivery or an empty dictionary
alone does not prove safe retirement. Existing contracts and open F01/F02/F08
findings are not resolved by this specification.

## Memory scope and future measurement procedure

Proposed interval-peak instrumentation must identify the service process and
CUDA device, exclude other model processes from its attribution, complete
process-local warmup and reach a safe idle boundary before synchronization and
peak reset. Record baseline allocated/reserved, reset time and the formal phase
start. Read synchronized peaks at the formal end before post-request drain;
document any unavoidable finalization work already included during replay.
Record post-drain and post-worker-close resource states separately.

PyTorch allocator peaks are not whole-device VRAM peaks or CUDA kernel timings.
Reserved memory differs from live allocated tensor memory. Allocation deltas are
not exact KV size. Device samples may include other processes and miss short
peaks; sampled maxima are lower bounds over the sampled points, not exact peaks.
Missing telemetry remains null. Normal completion, cancellation, failures and
cleanup retries require independent final resource evidence; no unconditional
dictionary clear or allocator cache flush can manufacture drain success.

## References and scope

See the [online protocol](online_benchmark_protocol.md) and
[runner README](../benchmarks/runtime/online_http/README.md).
No historical results, frozen traces/configurations or legacy metric formulas
are migrated by this document. Formal GPU Benchmark remains **NOT_ADMITTED**.
