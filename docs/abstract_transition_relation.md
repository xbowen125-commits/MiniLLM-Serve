# Abstract transition relation and frozen workload — Mission 24.4.2.8

Result: **PARTIALLY CLOSED**. Model/evidence work only. No production source,
generation/scheduling algorithm, frozen configuration or 25ms threshold changes.
No GPU or formal performance execution. No existing admission flag is promoted.

## Identity and inputs

HEAD remains `dc9db6e9cb2585bd7eb141717206586ec0e01fb3`; worktree is dirty.
The before-source inventory is saved in
`benchmarks/runtime/results/online_http/transition244428-a/source-before.json`.
Each audit saves its own current-source inventory and exact input identity.

The existing fingerprint algorithm hashes bytes of Git-visible tracked and
non-ignored untracked files, then SHA-256 hashes the sorted compact JSON of
`relative_path → file_sha256`. Git HEAD equality is not dirty-tree equality.
Ignored environments/results are excluded; no line-ending normalization is used.

Frozen configuration: `benchmarks/runtime/online_http/configs/qwen_admission_v2.json`.

| Identity | SHA-256 / value |
| --- | --- |
| Canonical config | `4145413187aa1ddaeae52cf5e1943e0a313788d730bf0c00aacddf436544ea29` |
| Pristine six-request arrival trace | `ff722d85aff3b3b0eabb1828d14c3b4c229383cb8361cd798cf39871709241c0` |
| Historical prepared trace | `97e3a89ea03ca60adb11d31e5801c46eebdbcc1aac73a90eca17d25d5e501ff7` |
| Pinned model/tokenizer revision | `7ae557604adf67be50417f59c2c2f167def9a775` |

The prepared trace includes actual tokenizer observations. Its hash is purposely
different from the pristine workload hash. The audit authenticates the old
manifest, recomputes its prepared hash, and compares its original workload fields
to the frozen configuration. This consistency check does not itself prove the
tokenizer's correctness or current-source GPU compatibility.

Historical CUDA input:
`benchmarks/runtime/results/online_http/preflight244424-trace-gpu-20261021-a/continuous/`.
Its original source fingerprint differs from current source. It remains historical
evidence, not a current certificate. New CPU fixtures bind their actual current
source/run/PID but explicitly use synthetic model/config identities, not Qwen.

## Extension, not another Runtime ledger

`transition_relation.py` is a pure offline projection/guard layer over the existing
raw trace, resource model and composition checks. It never submits, cancels,
schedules or owns requests. `taxonomy.py::transitions()` remains unchanged.

Observation variables include separate lifecycle and Context phase; actual
operation phase; original request-object identity; Lease identity/state; request
to Slot to cache-object relationships; actual cache length; resource retry and
confirmation fields; Owner running/draining/stopped observations; and backend
ownership. Queue counts do **not** supply queue member identity. Unobserved
per-request queue state stays UNKNOWN.

`before`/`after` in the mapping are last-direct-observation projections with
per-field source lines, **not complete concrete pre/post states**. Missing fields
are not reconstructed from the preceding timestamp. Registry or Slot absence
never clears the original RequestState reference by inference.

## Rules and proof obligations

| Rule | Required direct evidence / local predicate |
| --- | --- |
| BIND | Explicit request ID and original object; handoff Context still waiting with zero generated tokens |
| PLAN_SELECT | Actual post-retirement active/waiting lists, original identities, created times, capacity; stable same-key FIFO |
| PLAN_TRIM | Explicit iteration link; decode unchanged; prefill trimmed by observed free Slots |
| EXECUTOR_DISPATCH | Phase ID and exact selected phase members |
| WORKER_DISPATCH | Same phase ID, phase and explicit member IDs |
| WORKER_RESULT | Distinct known IDs, one result per member, correct phase |
| PHASE_APPLY | Explicit phase submit/result/end; token application exactly once; request/whole-phase errors terminalize affected requests |
| FORWARD | Explicit forward/phase/member IDs; mask/input/cache width equation and observed KV growth |
| SLOT_UPDATE | Actual pool before/after Slot and inventory; allocation/update/release do not imply request-ref clearance |
| CANCEL | Linked cancel call; terminal lifecycle and unchanged generated count |
| RETIRE_BEGIN | Explicit release ID and original object; attempt does not imply success |
| RETIRE_END | Same release ID; Slot absence, original reference None and pending-record absence checked separately |
| ACK | Linked operation; terminal request and confirmed retirement before successful forget |
| LEASE | Actual original request object and released flag; no first finalization before confirmed retirement |
| OWNER | Direct closing/thread observations; no inference from runtime_active_requests alone |
| CLOSE | Actual borrowed/owned relation; owned close cannot leave pending request resources |
| OBSERVE | Supporting raw observation, not a newly asserted execution transition |
| IDEMPOTENT | Identical event-ID/payload reapplication has no second effect; changed payload fails |
| UNMODELED | Unknown event kind remains UNKNOWN, not an allowlist exemption |

Read-only collector additions supply per-call operation/release IDs and actual
batch selection inputs. They do not influence decisions. No historical trace is
backfilled. Worker/Executor/Forward association uses IDs, not row position or time.
Original IDs are scoped to the actual PID/run; abstract `request-0/1` symbols in
the finite graph are not substituted for concrete request identities.

Every mapping row records raw line number, event/sequence IDs, rule/schema hash,
whole raw-file hash, canonical event hash, linked begin/input lines, partial
before/after observations and classified issues. An initial test caught mutable
projection aliasing; after-states now have independent snapshots and the original
counterexample assertion is retained.

## Frozen workload comparison

Historical client records are authenticated against the historical artifacts
index. Explicit `authoritative_usage.engine_request_id` links to actual Runtime
submit IDs; no temporal association is used. All 26 supplied historical client
records link, but they are warmup/correctness/fault probes, with **0 measured
records**. Six prompt counts are observed as 37, 72, 34, 42, 83 and 39; generation
limits are 16, 48, 16, 32, 48 and 16. These are provenance facts, not new GPU runs.
Reverse coverage is incomplete: 27 Runtime submit IDs exist, but ID `26` has no
corresponding authoritative client row. CLIENT_RUNTIME_COVERAGE stays UNKNOWN;
the missing row is not assigned to a disconnect/fault probe by inference.

The audit retains probe groups and all failures. Warmup/sequential/grouped arrivals
cannot be relabeled as a replay of arrivals 0, 0, .25, .5, .75 and 1 seconds.
Even a complete measured trace would only establish observed conformance; queue
arrival/removal identities, tied/network arrival semantics, Owner command/future
boundaries and a complete refinement relation would still need proof.

Explicit measured planned-arrival/generation-limit changes or lateness above
25ms are PROTOCOL_MISMATCH, not excused as noise. Wall-clock duration differences
alone are not request-order violations. Observed grant order is exported using
explicit engine/client links, but a tied HTTP/network-arrival order not defined
by the frozen contract stays PROTOCOL_SCOPE_UNDEFINED/OBSERVATION_INSUFFICIENT.

Findings are classified as IMPLEMENTATION_COUNTEREXAMPLE, MODEL_RULE_MISSING,
OBSERVATION_INSUFFICIENT, PROTOCOL_SCOPE_UNDEFINED/PROTOCOL_MISMATCH, or historical
identity mismatch. A current-source implementation counterexample is not inferred
merely because a historical trace lacks newer observer fields.

## Search, SCCs and bounds

The stable composition graph is reused, not replaced or given a larger budget:

* Initial states: 2 (borrowed / owned).
* Explored: 5,746; transitions: 12,792; stopped-owner states: 242.
* Unexplored: 0; duplicate successor visits: 7,048.
* SCCs: 4,306; cyclic SCCs / retry witnesses: 720.

Scope: two ordered requests, one running Lease, one Slot, one queue entry and
at most two Decode tokens each. Request lifecycle, resource stage, Lease,
Context/completion, reference/Slot ownership, cleanup target and ownership mode
are explicit. Retry count 2 means 2+ and is not a retry cap. Cache symbols denote
request-local ownership lineages, not an unchanging concrete cache object across
Decode row cloning. Context phases annotate **safe boundaries**; Prefill is an
atomic edge, not an exhaustive model of its partial-installation interior.

SCC traversal is iterative and reproducible. All graph nodes/edges and symbolic
Slot/reference-owner relations are saved in search.json. This exhausts that graph
only, not the six-request/capacity-two frozen workload or all Runtime behavior.
Tensor data, clocks, arbitrary failures/clients, unbounded diagnostic histories,
and forced termination are outside the graph. No complete abstraction/implementation
bisimulation has been established. The finite width bound in taxonomy v2 remains
an upper bound, not an exhaustive reachable-state proof.

## Execution and reports

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_abstract_transition_relation.py -q
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.transition_audit `
  --input benchmarks/runtime/results/online_http/preflight244424-trace-gpu-20261021-a/continuous `
  --output benchmarks/runtime/results/online_http/transition244428-a/manual-new-historical `
  --config benchmarks/runtime/online_http/configs/qwen_admission_v2.json `
  --clients benchmarks/runtime/results/online_http/preflight244424-trace-gpu-20261021-a/continuous/requests.jsonl `
  --artifact-root benchmarks/runtime/results/online_http/preflight244424-trace-gpu-20261021-a
```

Output directories are exclusive. Raw/index/client/prepared hashes are verified
before writing. Reports include event-mapping.jsonl, rules.json, search.json,
frozen-equivalence.json, inconsistencies.json, audit.json, source-inventory.json
and a new artifact hash index. Input files and saved old verdicts are never edited.

Safety conclusions concern the stated observed/local predicates. Liveness needs
fair retry, eventual cleanup observability/success and consumer finalization.
Permanent cleanup failure admits infinite executions; unconditional termination
is not proven. Old CACHE_OWNERSHIP, OBSERVED_ABSTRACT_STATE_DOMAIN,
V2_CLEANUP_RETRY_REPRESENTATION and full TAXONOMY_V2_STEP_RELATION,
FROZEN_SCHEDULE_EQUIVALENCE, FINITE_STATE_EXHAUSTION UNKNOWNs are preserved.
Historical PREFLIGHT_* blockers are archived with hashes, not promoted or reused
as current certificates. Formal GPU Benchmark remains prohibited.
