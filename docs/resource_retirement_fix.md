# Minimal resource retirement fix — Mission 24.4.2.6

This supersedes the implementation-defect status in the **historical**
[24.4.2.5 model report](resource_retirement_model.md), not its original evidence.
No performance experiment, CUDA run, admission relaxation or scheduler change.

## Audited ownership and minimal fix

`Owner → Facade → Executor → BatchedTransformersWorker → SlotKVPool` remains
unchanged. Facade and shared WorkerBinding serialize operations; the CUDA Worker
still synchronizes before dropping any execution/cache reference. Owner retries
acknowledge before publishing completion; Executor only records release after
the Worker has returned successfully. No changes to those components were needed.

Old superclass cleanup popped the Session before releasing the cache adapter.
If release raised, the retry had no way to clear the original RequestState.
The subclass's second pool release freed the Slot, concealing that retained ref.

The superclass now keeps a small `_retiring[request_id]` cleanup record **before**
removing the Session. It contains the original RequestState, attempt count,
separate slot/reference confirmations and the most recent cleanup/observation
errors. This is not another lifecycle/token registry. It never starts generation.
It is removed only after both cleanup steps are confirmed.

```text
Session ──transfer original RequestState──> cleanup record
                                            │
                                query this request's adapter entry
                                            │
                             release if present; query again
                                            │
                               clear original kv_cache_ref
                                            │
                                  directly verify None
                                            │
                                  remove cleanup record
```

Slot/cache absence is checked by the existing adapter's `get_batch_kv([id])`
contract: KeyError means unknown/absent entry. Other query errors stay errors,
with `slot_released=None` (UNKNOWN). Successful release returning with an entry
still present raises. No global slot count, Session absence or normal return is
used alone as retirement evidence. For an injected orphan cache with no original
request identity, cleanup fails closed and retains the occupied slot/record.
Unknown IDs with no resources remain idempotent no-ops.

When release raises after physical removal, the first attempt still raises and
retains its error/record. The next attempt queries authoritative absence, skips
redundant removal, clears the same RequestState, and confirms retirement. Failed
reference clearing likewise retains its record even when the Slot is already gone.
`close()` now also visits pending retirement IDs. Repeated cancel, acknowledge,
release and close retain their previous idempotent behavior.

Resource installation stores the Session/identity before calling the cache
adapter, so an adapter that allocates and then raises cannot lose request identity.
The model forward, tensor construction, sampling, outputs, scheduling and frozen
request workload are unchanged. The redundant subclass pool release is removed:
it would otherwise occur *after* the superclass's confirmation boundary.

## Evidence and independent checks

Read-only trace additions include:

* actual `id(original_request)` at release begin/end and retained-object evidence;
* pending cleanup record, attempts, separate confirmations and last errors;
* actual Slot state and direct `kv_cache_ref` observation after release;
* existing sequence/event/iteration/identity fields and sealed raw SHA-256.

Object identity is meaningful only within the same process/run, with the original
object kept alive by the test/cleanup record; it is not compared across processes.
Retirement checker v2 rejects identity changes and marks missing original-object
evidence UNKNOWN. It continues to reject successful-return/occupied-Slot and
successful-return/retained-reference contradictions. New evidence does not alter
the old seq=19 or seq=24–25 records or their hashes.

## Fault coverage

| Injection/path | Required result |
| --- | --- |
| First or first two Slot releases fail | Terminal request and original identity retained; no regeneration; retry clears ref |
| Slot removed then release raises | First result remains error; next attempt confirms absence before clearing ref |
| Release returns but Slot remains | Worker raises; no Executor release marker; occupied Slot not reusable |
| Reference clear raises before/after mutation | Slot absence alone does not acknowledge request; retry verifies original object |
| Reference clear returns without mutation | Direct read detects retained ref and raises |
| Resource query unavailable | UNKNOWN record, original error preserved, no retirement claim |
| Cancel and close interleave with failures | Serialized retries; failed close retains Context and record |
| Peer remains active | Its Slot/cache identity and generated state remain untouched |
| Session already removed | Cleanup record retains the original RequestState across all attempts |
| Repeated cancel/acknowledge/close | Idempotent; retained external request reference remains None |
| Allocation commits then raises | Installed cleanup identity survives; failed cleanup is retryable |
| Orphan cache lacks request identity | Fail closed, do not infer original reference clearance |

Tests also verify calls/token counts do not increase during cleanup retries,
resource-limited admission cannot reuse an occupied Slot, and request Context,
claims, released-ID and finished-ID retention reach zero only after confirmation.

## Reproduction and scope

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_resource_retirement_fix.py tests/test_resource_retirement.py -q
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.retirement `
  benchmarks/runtime/results/pytest244426-related-a/test_fault_then_acknowledge_or0 `
  --output benchmarks/runtime/results/online_http/retirement244426-a/manual-new-replay.json
```

The last output must not already exist. Machine-readable reports are stored in
the new ignored `benchmarks/runtime/results/online_http/retirement244426-a/`
directory. Historical failures remain historical failures under the v2 checker.
Current CPU traces are not CUDA evidence or current-source GPU admission proof.
New fault-injection traces bind the actual worktree SHA/dirty/source fingerprint;
their model/config identities explicitly remain synthetic CPU fixtures.

Safety evidence supports the synchronous retirement fix under the existing
serialized Worker/adapter contract. Liveness still requires eventual resource
observability/cleanup success and fair retry. Permanent failure still permits
infinite retry; unconditional termination is not proven. The 69-state,
268-transition one-request submodel is not a full Runtime proof. Overall closure
remains **PARTIALLY CLOSED**; formal GPU Benchmark prohibition remains in force.

## Validation record

* Fault/retirement/trace tests: 51 passed after adding actual source binding.
* Related Runtime, Worker, HTTP/SSE and Fake TCP tests: 125 passed.
* Full non-slow regression: 715 passed, 13 deselected; final evidence directory:
  `benchmarks/runtime/results/pytest244426-full-final/`.
* Source compilation and `git diff --check`: successful; existing line-ending
  normalization warnings do not indicate whitespace-check failures.
* Before-source manifest: `retirement244426-a/source-before.json`.
* Old counterexample replay: `retirement244426-a/old-seq19-replay.json`.
* Per-fault current-source replays: `retirement244426-a/fault-0.json` through
  `fault-7.json`, and `shutdown-retry.json` for reference-clear failure at close.
* These paths are below `benchmarks/runtime/results/online_http/` unless stated.

The before-source fingerprint is
`3cc49dd4881be5741e81d717d69a538ac737a0a112bea2207e84932dd6cc8c99`.
The after-source fingerprint is saved in every new replay's `current_source`.
Historical raw seq=19 SHA-256 remains
`714e257fbcc609ea52a9969d6017405f8dd649ccca51e77fce266f7fae9b9087`.
No historical PASS/FAIL flag or raw record is rewritten. Only the two Worker
source files change in this mission; Facade, Owner, Executor, pool, generation
algorithm and scheduler remain byte-identical to the before-source manifest.
