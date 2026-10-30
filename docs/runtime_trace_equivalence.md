# Runtime traceability and abstract-model boundary

Mission 24.4.2.4 collects correctness traces, not formal performance samples.
Production generation, scheduling and the frozen workload/configuration are
unchanged. The ordinary HTTP host does not install these hooks.

## Collection and ownership

Set `MINILLM_BENCH_RUNTIME_TRACE=1` only for the correctness preflight host.
Benchmark-only wrappers call and return the original methods/results/exceptions.
The owner thread serializes the facade and worker. RequestState still owns
lifecycle/token counts; the tracer only writes snapshots. It keeps no prompt,
generated text, tensors or second execution state. Cache object integer identity
is compared **only within one live snapshot**, never across object lifetimes.

| Field | Actual source |
| --- | --- |
| Run/PID/config/revision/dtype/device/source | New preflight identity and actual process |
| Iteration | Facade step counter and separate real scheduler iteration counter |
| Lifecycle/phase/token counts/times | Existing RuntimeRequestContext / RequestState |
| Proposed plan | Return of IterationScheduler.build_plan |
| Actual plan and free capacity | Continuous executor.execute_plan arguments and actual pool snapshot |
| Phase members | Executor._execute_batch_phase arguments |
| Worker requested members/results | Actual BatchedTransformersWorker call objects/results |
| Prefill Forward members/Prompt lengths | Actual BatchInputBuilder output request_ids and CPU attention-mask row sums |
| Decode ready members | Cache objects passed to _merge matched by `is` to actual pool.lookup(request_id); ambiguous mapping is null |
| Forward shape and merged cache | Base worker._forward arguments after device transfer; exact input/mask and initialized cache layers |
| Slot/cache ownership | Actual SlotKVPool mapping, lookup cache, session Context and request.kv_cache_ref identity |
| Release/error/retry | Original worker.release_request and actual pool allocate/update/release calls |

No member identity is reconstructed from batch indices, timestamps or terminal
event proximity. Read failures remain explicitly unobservable. Trace collector
failures invalidate proof and are not converted to successful execution.
These diagnostic wrappers add overhead; no timing or throughput comparison is
permitted from these traces. Existing backend synchronization remains unchanged.

## Independent checks

`runtime-trace.jsonl` retains every ordered raw event, phase/forward/event ID,
process/run identity and monotonic timestamp. `trace-end.json` records collector
errors. `trace-index.json` hashes both files; the outer preflight artifact index
also hashes the trace bundle. `trace_check` recomputes checks from raw events,
not from saved trace-check.json flags. External expected identities can be bound
through its Python API. An internally consistent old trace alone is not a
current-source certificate.

Checks account for legitimate differences: the Facade may trim proposed Prefill
to free Slot capacity; pending-first Decode samples without a Forward; invalid
rows may be excluded only with recorded request errors. Protocol mismatches must
be rejected and cleaned, not silently accepted. Successful release must leave no
slot; cleanup error remains unconfirmed until a later actual successful release.
Final drain requires a successful observed close, no Context/Slot, no unconfirmed
release and a complete collector. Existing preflight separately checks HTTP
leases, owner shutdown and finalizers.

## Abstract state and transition proof

The v2 request domain is N (not arrived), W (waiting), F (pending first token),
D (ready Decode), T (terminal), with generation count and physical padded cache
width. Stable snapshots check this domain; phase results verify exact token
increments, first-token cache invariance, merged-width growth and terminal
exclusion. These are **local refinement predicates**, not an exhaustive check of
the complete `taxonomy.transitions()` successor relation.

The complete state would also require owner delivery/admission order, plan
capacity trims, partial phase installation/errors, terminal resources awaiting
cleanup, acknowledgements and retries. V2's T/width-zero state cannot represent
a terminal request whose failed release still owns a Slot. Synthetic failure
traces expose that model gap; a later successful retry does not erase the gap.
`TAXONOMY_V2_STEP_RELATION` and `FROZEN_SCHEDULE_EQUIVALENCE` remain UNKNOWN.

Under the prior fixed-epoch/no-retry assumptions, generation count and padded
width have finite bounds. This does not imply whole-runtime termination:
permanent backend release failure can produce an unbounded retry/self-loop.
No universal termination or exhaustive state/transition count is claimed. This
mission does not raise the old 2,000-state budget. Future quotienting may merge
irrelevant clock values/cache object addresses, but must preserve slot ownership,
pending-first flags, cancellation, deadlines and whether cleanup is confirmed.
No bisimulation/coverage proof for that quotient is provided here.

## Reproduction

```powershell
Set-Location D:\AI_Lab\MiniLLM-Serve
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:MINILLM_BENCH_RUNTIME_TRACE = "1"
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.preflight `
  --config benchmarks/runtime/online_http/configs/qwen_admission_v2.json `
  --execute --allow-gpu --scope full
# Read-only replay of the new run's continuous subdirectory:
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.trace_check <new-run>/continuous
```

Outputs are exclusive new files/directories. Non-PASS preflight/checker exit 1
is expected while closure is partial. Actual GPU evidence must come from a newly
executed run; synthetic CPU traces and old runs are not new CUDA validation.
Passing every observed invariant still cannot authorize formal benchmarking.
