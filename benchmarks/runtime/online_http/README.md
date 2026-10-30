# Frozen online HTTP benchmark protocol

**Current restriction (2026-10-28): formal online GPU Benchmark is not admitted.**
Read [current research status](../../../docs/research/CURRENT_STATUS.md).
This document retains descriptions from earlier protocol milestones, including
the original lack of streamed usage. The later opt-in `usage.py` experiment
adapter exports authoritative benchmark metadata separately from standard
content; the default Chat API is not thereby a standard streaming-usage API.
That addition does not close shape/model/evidence-contract obligations. Commands
below are documentation, not authorization to run a blocked GPU experiment.

Mission 24.4.1 provides an auditable comparison framework, not a performance
result. It reuses the existing HTTP application, RequestScheduler, legacy engine
and opt-in continuous owner. There is no benchmark-specific scheduler or Runtime.
CPU/Fake tests validate the framework; they are never GPU performance evidence.

See the [protocol freeze](../../../docs/online_benchmark_protocol.md),
[HTTP/SSE correctness evidence](../../../docs/continuous_sse_validation.md),
[GPU timing audit](../../../docs/gpu_benchmark_validity.md) and
[cross-process replication](../../../docs/gpu_cross_session_replication.md).
Those older experiments have different scopes and are not new online samples.

## Current opt-in accounting and versioned semantics

The benchmark host installs `usage.install(app)`; clients explicitly send
`X-MiniLLM-Benchmark-Usage: 1`. A named `minillm.usage` event before `[DONE]`
contains terminal metadata, not standard chat content. Ordinary HTTP responses
and the default chat schema are unchanged. Validated counts come from the
terminal RequestState token list, including generated EOS/stop/hidden tokens;
they are separate from final visible-text retokenization. Disconnects may prevent
delivery, so an unknown count stays unknown unless independently journaled and
identity-bound. An event is not proof of KV retirement.

See [metric evidence contract v1](../../../docs/online_benchmark_metric_contract_v1.md)
for proposed field names, validation rules, memory scope and negative cases.
Existing `ttft_ms` and `tpot_ms` keep their meanings: client-visible TTFT and
visible-text TPOT proxy. Usage adds no per-token timing. This specification does
not implement new checks or grant admission; historical outputs retain their
original schema and evidence. The 25 ms lateness gate remains frozen.

## Frozen controls

`configs/smoke.json` is a bounded CPU/Fake TCP check, including an intentional
failed request. `configs/qwen_protocol.json` is a **proposed future GPU protocol**;
its existence does not mean the GPU experiment ran. Parameters are validated and
canonicalized before execution. Every artifact carries the SHA-256 of the full
canonical configuration, including SLO thresholds, and a separate trace SHA-256.

For each mode, fix all of the following before collecting formal samples:

- Model and tokenizer immutable revisions, exact input messages and chat template.
- Explicit device and dtype; no `auto` device, precision change or silent CPU fallback.
- Greedy generation or the same frozen temperature, top-p, top-k, seed and stop configuration.
- Per-request generation limits, prompt lengths, maximum model length and KV-cache setting.
- Client concurrency, server admission limit, queue capacity, queue/inference/client timeouts.
- Nominal arrival offsets, equal-arrival input order, output-channel capacity and send timeout.
- Warmup count, independent trial count, order, resource-drain deadline and SLO thresholds.

Both modes use the same admission limit; client concurrency and GPU batch size
are distinct concepts. The sample protocol permits two admitted requests and a
continuous batch size of two. Its six-request arrival trace allows six client
connections, so a client pool of two does not systematically postpone arrivals
behind completed responses. Server admission is still two, with bounded waiting.
This is a protocol choice, not an observed performance improvement. Legacy concurrent GPU execution must pass an
independent correctness preflight before a formal GPU comparison is accepted.
Do not serialize legacy to one admitted request while retaining two for
continuous and call that an otherwise identical configuration.

For real models, the launch wrapper resolves both revisions from existing local
cache only, then loads pinned snapshots. No model downloads are allowed. Server
metadata must prove the actual model/tokenizer revisions, process identity,
device and dtype; merely recording requested settings is insufficient. Prompt
token counts include the actual chat template and tokenizer special-token rules,
not word counts or requested length labels. Optional declared expected lengths
are checked; absence of an expectation is not proof of exact output length.
`max_tokens` is a limit, not a promise to generate that many tokens: EOS/stop can
end a request early. Original output hashes and counts are retained for comparison.

## Arrival replay and AB/BA order

Each trial starts a fresh server process and loads one model. A zero-based pair
index defines alternating order:

```text
pair 0: legacy, continuous
pair 1: continuous, legacy
pair 2: legacy, continuous
...
```

The exact same serialized request trace is replayed in each mode. Arrival offsets
are relative to a new monotonic phase origin, not completion of the previous
request. The client concurrency cap can defer actual dispatch. Every raw row
therefore preserves nominal arrival, actual send time and dispatch lateness.
Lateness beyond the frozen tolerance is a comparison issue, not erased by moving
the arrival timestamp. Equal nominal arrivals do not guarantee exact OS or
server-ingress simultaneity. The API does not expose server ingress timestamps;
the framework cannot claim identical actual server arrivals.

Startup/model loading, warmup, formal replay, resource drain and shutdown are
separate phases with separate wall times. Formal rates include arrival gaps and
client terminal observations, but exclude startup, warmup and post-request drain.
Warmup is journaled with its own phase label. Never merge rows by a wildcard over
old directories or pool warmup, prior experiments and formal samples together.

The original sequential warmup cycled through preflight-valid entries in trace
order using a single client and warmup-only IDs. Current `warmup.py` supports
`sequential_and_pairs`, selected by `configs/qwen_admission_v2.json`: individual
requests are followed by request-pair groups. Pairs do not independently prove
actual GPU batch=2 or complete prefill/decode Tensor and KV shape coverage.
Warmup inputs are not synthesized
as different prompts, and the fake fixture's intentionally invalid input is not
used to pretend a successful warmup. Formal replay still retains that failed input.

Timeout controls have distinct scopes: startup has `startup_timeout_s`; each
request has an absolute `request_timeout_s`; replay is bounded by
`max_run_seconds`; each drain has `drain_timeout_s`. These are not one shared
whole-experiment deadline. Shutdown waits a bounded grace period and, if the
server has not exited, records an aborted experiment and starts no next model
process. The runner does not force-kill a process and report a successful drain;
manual intervention may be required for an unconfirmed shutdown.

## Metric definitions

For request `i`, use one client monotonic clock and these observations:

- `a_i`: nominal arrival offset; `s_i`: actual send-start offset.
- `f_i`: arrival of first nonempty, tokenizer-valid visible text delta.
- `l_i`: arrival of last nonempty visible text delta.
- `z_i`: `[DONE]` receipt, complete JSON body receipt, or observed failure/timeout/cancellation.
- `V_i`: complete visible response text retokenized with the common pinned tokenizer.
- `G_i`: actual generated tokens from authoritative server usage, if available.

`f_i` is a **client-visible TTFT proxy**. Unicode/word/stop holdback can delay it
relative to internal generation. No SSE chunk count is interpreted as token count.

| Metric | Frozen definition / population |
| --- | --- |
| Client TTFT | `1000 * (f_i - s_i)`; non-streaming or no visible content: N/A |
| Visible TPOT proxy | `1000 * (l_i - f_i) / (V_i - 1)` for `V_i > 1`; otherwise N/A |
| HTTP end-to-end latency | `1000 * (z_i - s_i)`; includes the terminal transport tail |
| Formal window `W` | `max(z_i) - phase_origin`, including scheduled idle gaps |
| Completed generated output tokens/s | `sum(G_i of completed, length-valid requests) / W`; unknown usage means N/A |
| Completed visible output tokens/s proxy | `sum(V_i of completed, length-valid requests) / W` |
| Completed requests/s | Number of completed, length-valid requests divided by `W` |
| SLO goodput requests/s | Completed, length-valid requests satisfying all applicable frozen SLOs divided by `W` |
| Error/timeout/cancellation/mismatch rate | Corresponding retained outcome count divided by all planned requests |
| P50/P95/P99 | Linear interpolation at index `(N-1)*p`; populations and sample counts labelled |
| Sample standard deviation | Uses `N-1`; N/A for fewer than two observations |

The first token is excluded from the TPOT denominator; the role, finish and DONE
events are excluded from its numerator. Because visible deltas can bundle multiple
tokens, TPOT remains an inter-content proxy, not precise device per-token latency.
The same rule applies to legacy and continuous. Successful-only TTFT/TPOT/E2E
distributions and all-attempted E2E distributions are reported separately. The
client queue delay before send is not silently added to request TTFT; nominal
arrival-to-send lateness is separately retained.

Receipt timestamps are captured by the HTTP observer, not a packet capture.
Socket/JSON work and client scheduling are part of these measurements. The
tokenizer confirms first visible content and retokenizes final text; it does not
retokenize every prefix after the first event. SSE prompt counts without
authoritative prompt usage retain `pinned_tokenizer_chat_template_preflight`;
validated opt-in usage can supply `prompt_count_source=server_usage`.
JSON usage also uses `server_usage`.

SLO thresholds are mandatory configuration values established before execution.
Missing required observations fail SLO qualification rather than being invented;
TPOT is not applicable for at most one visible token. Threshold edits produce a
different configuration hash and a different experiment. Never select thresholds
after reading outcomes to improve reported goodput.

### Explicit observability gaps

Ordinary Chat API SSE has no benchmark usage extension or generated Token IDs.
The opt-in benchmark host can supply validated terminal `G_i`, but not exact
internal-token TPOT or Runtime per-request execution timestamps. Without that
evidence the corresponding fields remain **N/A**. Visible counts can omit EOS,
special or stop tokens and differ from generation. JSON usage and validated
benchmark usage are separate from visible proxies; historical missing counts
are never backfilled. Default production API semantics are unchanged.

The status endpoint exposes process-lifetime CUDA allocation peaks and owner KV
slot peaks. These include model startup and warmup; they are not exact formal
phase peaks. Periodic formal-phase sampling supplies only lower-bound observed
maxima. Legacy does not expose a SlotKVPool: missing legacy slot telemetry is
N/A, not an invented zero. Snapshots include final scheduler leases, runtime
active/pending requests, slots, Contexts, ID claims, consumers, commands and
finalizers when available. Cleanup failure prevents an accepted drain result.

## Raw records and failures

An abbreviated **schema illustration, not an executed sample**, is:

```json
{
  "trial_id": "trial-000-legacy",
  "pair_index": 0,
  "mode": "legacy",
  "phase": "measured",
  "trace_ordinal": 0,
  "config_sha256": "<canonical configuration SHA-256>",
  "trace_sha256": "<prepared workload SHA-256>",
  "git_sha": "<HEAD SHA>",
  "git_dirty": true,
  "server_pid": 12345,
  "request_id": "short-01",
  "scheduled_at_s": 0,
  "sent_at_s": 0.001,
  "first_content_at_s": 0.101,
  "last_content_at_s": 0.201,
  "terminal_at_s": 0.211,
  "http_status": 200,
  "response_id": "chatcmpl-<server identity>",
  "lease_id": "<scheduler identity>",
  "outcome": "completed",
  "success": true,
  "visible_output_tokens": 4,
  "generated_output_tokens": null,
  "generated_count_source": "unavailable_sse",
  "finish_reason": "length",
  "saw_done": true,
  "protocol_errors": [],
  "runtime_metrics": null
}
```

Rows also retain derived timing metrics, expected/observed prompt lengths,
optional visible-length expectations, output SHA-256, dispatch lateness,
effective absolute timeout budget and error details. JSONL is flushed in
completion order; `trace_ordinal` reconstructs the original admission intent
without rewriting the recorded completion order.

Every planned request has a JSONL record, including not-sent, rejected, failed,
timed-out, cancelled, malformed/truncated SSE and output-length mismatch cases.
SSE success requires stable response ID, ordered assistant/content/finish events,
an accepted finish reason, `[DONE]` and no error envelope. `[DONE]` alone does
not assert success. Raw timing observations and error details are retained even
when derived metrics are unavailable or invalid.

Each trial has provenance, phase durations, before/after resource snapshots,
periodic observations and immutable request records. The root report includes
every trial and comparison-validity issues, not only the best results. An invalid
comparison is diagnostic evidence, not permission to drop bad samples.

The frozen `config.json` is an input document, not a self-hashing metadata wrapper.
Its canonical SHA-256 is carried by the trace, trial records and result metadata;
the final `artifacts.json` additionally binds each artifact filename to its file
SHA-256, including configuration and raw journals. The source manifest records
HEAD, dirty status and source-tree identity separately from generated results.
Per-trial `summary.json` is immutable with `validity_scope=trial_local_only`:
it precedes cross-mode pair and final source validation. Appended final-verdict
files and the root `summary.json` carry the authoritative final validity. The
Markdown report exposes planned/executed/unrun trials, experiment aborts, source
consistency and pair validation. An incomplete pair never becomes a completed
comparison merely because one local trial succeeded.

The runner creates a new output directory exclusively and refuses an existing
target. New JSONL/JSON/Markdown artifacts are not written over old files. Partial
records remain on launch failure, timeout or unsuccessful shutdown. Results are
under ignored `benchmarks/runtime/results/online_http/`; intentional publication
requires a separate privacy/provenance review. Logs may contain local diagnostic
paths and are not automatically public reports.

## Windows PowerShell

Only the small CPU/Fake run belongs to Mission 24.4.1:

```powershell
Set-Location D:\AI_Lab\MiniLLM-Serve
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Inspect the frozen plan without starting a model or server.
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.runner `
  --config benchmarks/runtime/online_http/configs/smoke.json

# Explicitly execute only the bounded CPU/Fake TCP smoke protocol.
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.runner `
  --config benchmarks/runtime/online_http/configs/smoke.json --execute

.\.venv\Scripts\python.exe -m pytest tests/test_online_http_config.py `
  tests/test_online_http_metrics.py tests/test_online_http_report.py `
  tests/test_online_http_runner.py tests/test_online_http_gate.py `
  tests/test_online_http_preflight.py -q
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
.\.venv\Scripts\python.exe -m compileall -q benchmarks/runtime/online_http tests
git diff --check
```

## Mission 24.4.2: correctness preflight and fail-closed admission

The [evidence audit and gate design](../../../docs/online_preflight_validity.md)
separate diagnostic probes from formal performance samples. `preflight.py`
defaults to a plan; `--scope single` sends the first frozen request once to each
fresh mode process. It copies exact terminal RequestState Token IDs and observes
forward attempts/completions without changing generation. This is correctness
instrumentation, never a performance measurement. Full mode additionally offers
bounded concurrency-two, fault, disconnect and active-shutdown probes; execution
of single mode is not evidence that those full checks passed.

Every required check has `PASS`, `FAIL` or `UNKNOWN` plus evidence. Non-PASS
preflight exits with code 1 and cannot authorize a formal run. Report/config/
trace/environment/Git dirty-source identities and all artifact hashes must agree.
Artifact checks detect missing/stale/modified files, not a maliciously forged
manifest; they are not signed attestation. All paths must stay under the report
directory. A fresh output directory is mandatory for every retry.

`gate.py` enforces the fixed 25ms tolerance, complete request records, expected
outputs, successful drain/shutdown and process-local warmup proof. Logical
batch sizes alone are insufficient: input/mask/KV-length tensor shape coverage
must be attested. The original sequential warmup cannot establish Continuous B=2
coverage; current paired dispatch alone cannot establish it either. Actual
forward observations and complete required-shape evidence remain necessary.
Diagnostic B=2 probes are not warmup of a future process.

The original streaming configuration lacked authoritative generated-token usage.
Later opt-in accounting does not close the remaining identity, shape and
evidence obligations. Formal performance statistics remain **blocked** until
all existing admission checks pass. No new admission is granted here;
`formal_statistics` remains null for blocked runs;
raw diagnostic summaries and rejected records are still retained. Visible-text
proxy eligibility never substitutes for generated-token eligibility. Deliberate
fault probes have `performance_sample=false` and cannot enter formal samples.

New records add `generator_scheduled_at_s`, `worker_started_at_s`, body build
start/end and `http_call_at_s`. Original `sent_at_s` and metric boundaries are
unchanged. Send lateness decomposes into generator lateness + client pool wait +
client preparation; `queue_wait_ms` is independently observed server admission.
Clock implementation/resolution is recorded, not treated as proof of actual
sleep/wakeup accuracy. Historical missing timestamps are never synthesized.

GPU correctness commands (not a throughput benchmark):

```powershell
# Review the complete frozen protocol first (no GPU/server execution).
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.runner `
  --config benchmarks/runtime/online_http/configs/qwen_protocol.json

# Low-cost correctness only; default single scope does not cover all checks.
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.preflight `
  --config benchmarks/runtime/online_http/configs/qwen_protocol.json `
  --execute --allow-gpu --scope single

# FUTURE ONLY: needs a complete matching PASS report and observable token usage.
# Current frozen SSE configuration is blocked; this is not a bypass command.
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.runner `
  --config benchmarks/runtime/online_http/configs/qwen_protocol.json `
  --execute --allow-gpu --preflight <matching-preflight-report.json>
```

Changing prompts, arrival rate, concurrency, token lengths, SLO, versions or
model revisions creates a new experiment, not an update to historical data.
This framework does not establish production throughput, resolve historical GPU
performance stratification, or replace long-soak and independent signal tests.
