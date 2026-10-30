# Frozen workload replay and identity closure — Mission 24.4.2.9

Scope: bounded correctness replay and independent identity reconstruction, not
an online performance experiment or a full Runtime equivalence proof. Earlier
admission blockers remain in force; no historical certificate is promoted.

## Frozen inputs and provenance

The configuration remains `benchmarks/runtime/online_http/configs/qwen_admission_v2.json`:

* Config SHA-256: `4145413187aa1ddaeae52cf5e1943e0a313788d730bf0c00aacddf436544ea29`.
* Pristine workload SHA-256: `ff722d85aff3b3b0eabb1828d14c3b4c229383cb8361cd798cf39871709241c0`.
* Model/tokenizer revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
* Six original requests at 0, 0, .25, .5, .75 and 1 seconds; generation limits
  16, 48, 16, 32, 48 and 16; admission/batch limit 2; lateness tolerance 25ms.

`artifacts.git_snapshot()` hashes Git-visible file bytes and then the canonical
path/hash inventory. Equal HEAD does not imply equal dirty source. Every replay
saves its full inventory, server PID and actual server revision/device identity.

## Historical 26/27 discrepancy

Historical input is the immutable directory
`benchmarks/runtime/results/online_http/preflight244424-trace-gpu-20261021-a/continuous`.
The client JSONL has 26 records; the Runtime trace has 27 successful submissions.
`probe.json#/faults/cancel` separately records lease `26`, its response ID and an
out-of-band terminal counter with engine ID `26` and status `cancelled`. The raw
Runtime submit is seq 7179. `preflight.disconnect_after_content()` returns a
separate probe object; unlike `observe_request()`, that call is not appended as a
client JSONL row. This directly explains the counting scope, without inventing
a missing row or matching timestamps. Missing planned/client/completion IDs for
that historical probe remain UNKNOWN. Its source fingerprint is historical.

## Opt-in observation boundaries

`identity_server` is a separate correctness-only process entry. The normal host
does not install these observers. `IdentityObserver` wraps existing methods and
records actual objects, with no state transitions or selection decisions:

| Boundary | Direct identity source |
| --- | --- |
| HTTP ingress | Explicit planned/client-record headers; exchange UUID |
| Admission creation/accept | Actual returned RequestState and SchedulerLease |
| Queue entry/removal | Actual before/after RequestQueue object members |
| Owner enqueue/start | Actual `_Delivery`, Future and original RequestState |
| Runtime handoff | Actual returned Context and its original RequestState |
| Worker/Forward | Existing phase IDs, actual member IDs and cache ownership hooks |
| Completion | Actual delivery Future result; its command ID is the terminal-result correlation ID |
| Finalizer | Actual successful Lease.release return and original reference observation |
| Cancel | Actual Event.set intent, separately from existing Facade cancel-end |
| Shutdown | Actual Owner close request and stopped Future; thread exit is checked separately |

The enqueue event is an **attempt** before the existing queue operation; rejection
is recorded separately, and only a matching actual dequeue proves command start.
Observer metadata does not manage lifecycle. Diagnostic wrappers can retain objects
until collector teardown; this finite six-request host is not a memory-soak test.
No prompt/output text is added to the identity trace. Outgoing body bytes are
hashed for frozen-payload verification. The reused terminal observer additionally
records token IDs, never a separate generation implementation.

Observation write failures are recorded in trace-end errors and invalidate proof;
they must not change generation results. Missing original IDs stay UNKNOWN.
Events are ordered by the collector's lock-protected sequence and have monotonic
timestamps; timestamps are not used to associate identities.

## Replay and independent recomputation

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.identity_replay `
  --config benchmarks/runtime/online_http/configs/qwen_admission_v2.json `
  --output benchmarks/runtime/results/online_http/identity244429-manual-new `
  --execute --allow-gpu
```

The output directory must not exist. This invokes the existing `runner.replay`
once with phase `identity_correctness`, not `run_experiment`. It does not require
or fabricate a performance admission certificate. Formal performance samples and
statistics remain 0/null. The original timestamps, client concurrency and 25ms
rule are unchanged. Slow instrumented sends are retained and fail their gate;
there is no shifted origin or second chance replacing an unfavorable replay.

Output length follows the existing gate: `length` must reach max_tokens;
`stop` may terminate earlier. Authoritative client counts are cross-checked
against the original per-request terminal token-ID observation. The initial
new identity checker incorrectly treated every max_tokens as an exact length;
its failed report is retained, and the correction has a regression test. It is
not a change to the existing admission gate or the frozen generation behavior.

Original trace/end hashes, client records, replay/drain evidence and server
metadata are sealed. `identity_check.audit()` verifies seals before independently
rebuilding edges, each with event ID, file, line/pointer and physical SHA-256.
Seven-layer identity, observed queue FIFO, payload, lateness, completion and final
resource drain are distinct proof obligations. Repeated identical events are
idempotent; conflicts fail; missing boundaries stay UNKNOWN.

Reports save identity chains, old local-rule mapping, additional boundary-rule
mapping, combined counts and a fail-closed admission summary. Old mapper UNKNOWNs
are preserved in their own report. The new rules are evidence predicates, not
an assertion of complete concrete pre/post states or a Taxonomy v2 bisimulation.

## Proof boundary

Even a complete six-request observed chain establishes only conformance of that
bounded replay. Tied TCP arrivals are not given an invented total ordering.
Cancellation/failure/retry paths use deterministic CPU/Fake composition tests;
a normal six-request GPU replay does not establish their GPU correctness anew.
No larger state search budget is used. The existing two-request composition
graph does not prove exhaustive six-request or entire-Runtime reachability.

`FROZEN_SCHEDULE_EQUIVALENCE`, `FINITE_STATE_EXHAUSTION`,
`TAXONOMY_V2_STEP_RELATION`, old cache/domain/retry-representation UNKNOWNs and
historical `PREFLIGHT_*` blockers remain unclosed. Liveness/termination continue
to require retry fairness, eventual resource-release success and finalization.
Formal GPU Benchmark is still prohibited.
