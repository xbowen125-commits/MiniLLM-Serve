# Mission 24.4.2.1 — authoritative SSE counters and correctness coverage

For the subsequent versioned shape inventory, exact matching rules and read-only
raw-proof replay, see [Workload shape coverage](workload_shape_coverage.md).

This is a bounded correctness/lifecycle task, not an online throughput experiment.
All historical results remain immutable. The original six requests, revisions,
generation parameters, arrival offsets and 25ms tolerance are unchanged.

## Versioned opt-in transport

The original `qwen_protocol.json` is unchanged. The new
`benchmarks/runtime/online_http/configs/qwen_admission_v2.json` adds explicit
`authoritative_sse_usage=true` and `warmup_strategy=sequential_and_pairs` controls.
It has a new canonical configuration hash; old experiments are not relabelled.

Only the benchmark process host installs an adapter. Ordinary production FastAPI,
legacy defaults, non-streaming JSON, Worker algorithms and scheduling are not
modified. A client opts in using `X-MiniLLM-Benchmark-Usage: 1`. A separate
`event: minillm.usage` appears before DONE and carries no content delta:

```text
event: minillm.usage
data: {"object":"minillm.benchmark.usage","id":"chatcmpl-...","request_id":"...","engine_request_id":"...","count_source":"terminal_request_state","usage":{"prompt_tokens":37,"completion_tokens":16,"total_tokens":53}}
```

The example illustrates schema, not a new measured result. Actual events also
record terminal status and inclusion semantics. The count is the length of the
existing terminal RequestState generated-token list. It includes generated EOS,
the whole Token triggering a stop string, and buffered/hidden generated Tokens;
it does not include prompt Tokens or Tokens never generated. Visible text
retokenization remains a distinct proxy. Legacy binds its actual runner state
through the lease cancellation event; Continuous reuses the shared lease state.
No second generation loop or lifecycle registry is introduced. Weak session
references and per-stream removal avoid retained historical request states.

Missing or unconfirmed terminal state yields null, never guessed zero. Error or
cancelled requests do not become successful because a usage event exists. A
disconnected TCP client cannot receive a final event: its client count remains
N/A, while the independent terminal observer journal records server-side partial
counts. Unknown resource cleanup also prevents claiming a confirmed count.

Both response and lease identities must match before client/gate accepts a count.
The event is explicitly benchmark metadata, not a standard OpenAI content chunk
or a claim that the OpenAI stream_options contract was implemented.

## Fixed-set correctness and process-local warmup

Full correctness scope uses two independent service processes, one per mode.
All six fixed requests run independently and in bounded pairs, retaining exact
terminal Token IDs, prompt/generated counts, response/lease/backend identities,
and resource snapshots. Forward-error injection, partial-stream disconnection
and active shutdown are separate fault probes with performance_sample=false.
Fault generation is capped; socket probes have absolute deadlines.

The same server/model process first warms every fixed request sequentially and
then in pairs. Immutable warmup start/end markers include PID, mode, configuration
hash and model revision. Each successful Forward records input, attention mask,
KV sequence length and initialized KV tensor dimensions. Failed attempts do not
count as warmup. A batch=2 assertion requires an observed tensor first dimension
of two, not two HTTP callers or two occupied slots.

The formal runner reuses the recipe and writes its live-process coverage record.
It must stop before formal replay if that process lacks complete coverage.
Preflight-process warmup is never evidence that a new future process is already
warmed. Observed successful shapes are not an exhaustive declaration of every
possible future padded-width/KV-length combination. Missing exhaustive coverage
remains UNKNOWN and cannot unlock admission. The 25ms gate and original planned
arrivals remain unchanged; correctness request times are not performance samples.

## Reproduction and evidence

```powershell
Set-Location D:\AI_Lab\MiniLLM-Serve
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.preflight `
  --config benchmarks/runtime/online_http/configs/qwen_admission_v2.json `
  --execute --allow-gpu --scope full
```

Every execution creates an exclusive new directory. Inspect preflight.json,
per-mode requests.jsonl, observer.jsonl, probe.json, warmup markers and artifacts.json.
Non-PASS returns code 1 and does not authorize a formal run. A complete PASS still
requires all original identity, cleanup, tensor-coverage and artifact gates; this
milestone does not lower them. GPU outcomes must be read from actual new artifacts,
not inferred from these commands or CPU tests. No commit/push is performed.
