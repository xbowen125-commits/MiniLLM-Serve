# Mission 24.4.1 — Online benchmark protocol freeze

Mission 24.4.2 adds a [fail-closed correctness/validity gate](online_preflight_validity.md)
without changing these frozen controls. The current SSE configuration cannot
enter formal performance statistics until complete correctness, process-local
warmup/tensor-shape and authoritative generated-token accounting requirements
are met. A partial single-request GPU probe is not a formal admission PASS.

## Status and permitted scope

This milestone freezes an auditable protocol and implements a bounded real TCP
HTTP runner. CPU/Fake smoke validation is not real model performance evidence.
No full online GPU experiment is authorized or claimed here. Existing Worker,
Runtime, Scheduler, HTTP generation and historical benchmark results are unchanged.
The working tree remains uncommitted; a Git HEAD SHA alone cannot identify its
current behavior, so artifacts also record dirty status and a source-tree hash.

The [full protocol and commands](../benchmarks/runtime/online_http/README.md)
define fixed controls, metrics, artifact scope and limitations. The frozen inputs
are [CPU/Fake smoke](../benchmarks/runtime/online_http/configs/smoke.json) and
[future Qwen CUDA protocol](../benchmarks/runtime/online_http/configs/qwen_protocol.json).

## Audit evidence motivating a separate framework

| Existing behavior | Consequence for the frozen protocol |
| --- | --- |
| Existing HTTP benchmark submits requests through a thread pool without scheduled arrival offsets | Preserve one immutable nominal trace and actual dispatch lateness for each mode |
| Existing streamed output is reconstructed then retokenized | Label visible-text counts separately from authoritative generated counts |
| Existing SSE schema contains no usage or Token ID field | Actual streamed generated tokens/s and exact internal Token TPOT stay N/A |
| `[DONE]` can follow an explicit SSE error | Require a valid finish, stable response ID, DONE and no error for success |
| Legacy HTTP TPOT includes the end-of-stream tail | Freeze a distinct visible-content TPOT proxy, excluding finish/DONE tail, for both modes |
| Health does not itself prove revision, process identity or execution mode | Record launch metadata and pinned local snapshots in addition to health/status |
| CUDA and KV peak status counters are lifetime maxima | Label lifetime peaks; sampled formal maxima are lower bounds, not exact peaks |
| Runtime internal request metrics are not exposed by HTTP | Do not infer those execution metrics from client transport timestamps |

Sources: [existing HTTP benchmark](../benchmarks/benchmark_http.py),
[Chat/SSE schema](../src/minillm/schemas.py),
[HTTP server](../src/minillm/server.py),
[single runtime owner](../src/minillm/runtime/http_owner.py) and
[prior TCP correctness report](continuous_sse_validation.md).

## Fixed experimental contract

Both modes use the same immutable model/tokenizer revisions, messages, dtype,
device, sampling parameters, KV-cache setting, token limits, admission/queue
settings, nominal arrival trajectory and predeclared SLOs. The full normalized
configuration and trace receive SHA-256 identities. Actual client send offsets
are measured independently; exact server ingress identity is not observable.

Each mode trial is a new server process. Pairs alternate legacy/continuous and
continuous/legacy. Startup/loading, warmup, measured replay, resource drain and
shutdown are timed separately. Only the measured replay window contributes to
formal completed rates. The same frozen SLO thresholds apply to every trial;
changing thresholds or any control requires a new configuration hash.

Warmups cycle through preflight-valid frozen trace entries using separate IDs,
one client and phase-labelled raw rows. Startup, per-request absolute deadlines,
replay budget, resource drain and shutdown grace have separate scopes; no claim
is made that `max_run_seconds` covers all model loads and all pairs. An unconfirmed
server stop aborts subsequent process launches instead of force-killing and
claiming cleanup success.

All failures, timeouts, cancellations, length mismatches, malformed/truncated
streams, excessive dispatch lateness and failed drains remain recorded. Invalid
trial/pair labels are diagnostic, not a mechanism for silently filtering samples.
Result directories and raw artifacts are exclusively created, never overwritten.

`config.json` is the frozen input. Canonical configuration identity is recorded
in surrounding metadata, and `artifacts.json` binds filenames to actual file
SHA-256 values. Immutable per-trial summaries have `trial_local_only` validity
scope. Final-verdict files are appended after cross-mode and source validation;
those files and the root summary define final validity. Reports explicitly list
planned/executed/unrun trials, aborted experiments, source consistency and pair
verdicts. A locally successful but unmatched trial is not an accepted comparison.

## Metric and evidence boundaries

HTTP TTFT is actual send-start to first visible, tokenizable content at the
client, not the Runtime first Token timestamp. Visible TPOT is the first-to-last
content interval divided by retokenized visible count minus one. End-to-end
latency includes the final transport event. Chunks are never treated as Tokens.

Generated counts are accepted only from server usage. With current SSE, those
counts and true per-Token/internal execution timing are N/A. Visible-text rates
remain explicitly labelled proxies. This is a deliberate observability gap, not
permission to equate visible and generated Tokens. Per-request token limits are
not assumed to be exact generation lengths because EOS and stop may finish early.

GPU/KV lifetime peaks include warmup; periodically sampled measured maxima may
miss short peaks. Final resources must be checked after execution and consumers
finish, rather than made zero by clearing state. A CPU/Fake run can validate
retention, protocol and reporting, but not real CUDA concurrency or speed.

## Gate for a future GPU experiment

1. Review/freeze exact protocol JSON and SLO before execution.
2. Verify local cached immutable revisions and actual dtype/device.
3. Run a bounded legacy-versus-continuous greedy correctness preflight at the
   same admission limit, including concurrent legacy CUDA execution.
4. Acknowledge that actual streamed generated counts, exact interval memory
   peaks and Runtime execution timestamps remain unavailable with the current API.
5. Retain every trial, phase, raw record, process identity and dirty-source hash.
6. Check dispatch lateness, pair consistency and final resources before drawing
   any configuration-specific conclusion. Do not claim universal speedup.

The current fixed warmups use one client and cycle through valid trace entries;
they do not prove that every concurrent GPU batch shape is fully warmed. Review
that coverage in a separately authorized preflight before accepting formal GPU
results. Changing a warmup rule requires a new frozen configuration/protocol,
never relabelling old measurements as already warmed.

Long soak, separate-process signals, forced termination and production-scale
online throughput remain outside this protocol-freeze milestone.

## Executed validation, 2026-10-20

- New protocol/configuration/metrics/report/CPU TCP tests: **65 passed** within the full regression.
- Complete non-slow regression: **515 passed, 13 deselected** (39.71 seconds).
- `python -m compileall -q src tests benchmarks`: exit 0.
- `git diff --check`: exit 0; new files also checked for trailing whitespace.
- Both new Markdown documents: zero broken local links.
- A standalone CPU/Fake CLI smoke used four fresh Uvicorn processes in LC/CL
  order. It retained four warmups and sixteen formal records: twelve completed
  requests and four intentionally failing requests. All processes stopped;
  scheduler leases drained, and continuous Contexts, claims and KV slots were zero.
- That smoke also flagged excessive actual send lateness in all four trials.
  Failed/late comparisons remain invalid and inspectable; the threshold was not
  relaxed. Protocol validation is not an accepted performance comparison.
- Mission 23.6 original/corrected GPU JSON hashes and all three Mission 23.7
  session hashes were rechecked without modification.

Reproduction commands are in the linked protocol README. Raw outputs remain
under ignored `benchmarks/runtime/results/online_http/` with distinct run IDs.
The GPU configuration was inspected in plan-only mode; **no new real-model or
GPU benchmark was executed**. Hosted CI, GPU correctness under the new online
protocol, exact arrival matching, long soak and signal tests are not established
by these CPU/Fake checks. The worktree remains dirty and uncommitted.
