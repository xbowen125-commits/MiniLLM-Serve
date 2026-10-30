# Mission 24.1 — Offline Continuous Runtime Integration

## Implemented chain

```text
RuntimeFacade(execution_mode="continuous")
  -> shared RuntimeRequestContext registry
  -> IterationScheduler.build_plan()
  -> ContinuousBatchPlan
  -> continuous.BatchExecutor.execute_plan()
  -> BatchedTransformersWorker
  -> SlotKVPool
```

The default execution mode remains `legacy`, with its prior phase executor path.
HTTP and GenerationEngine are unchanged. This is an offline synchronous API,
not an online Continuous Batching service or production performance claim.

## Construction

```python
from minillm.runtime import RuntimeFacade
from minillm.workers import BatchedTransformersWorker

worker = BatchedTransformersWorker(loaded_model, slot_capacity=2)
runtime = RuntimeFacade(worker, execution_mode="continuous", max_batch_size=2)
try:
    first = runtime.submit("The capital of France is", request_id="first")
    runtime.step()  # prefill, no generated token yet
    runtime.step()  # first token from prefill logits
    runtime.cancel("first")  # safely releases this request's cache
finally:
    runtime.close()
    worker.close()  # caller owns backend by default
```

Continuous mode requires a batch worker with idempotent request release,
published max_batch_size and a pool capacity. Constructor rejects mismatched
Scheduler/Worker/pool capacities. An optional existing IterationScheduler must
match the requested batch size. Current real worker supports at most two rows.

## Ownership and serialization

- ManagedBatchWorkerProtocol adds `release_request(request_id)`. Unknown or
  repeated release is a no-op for the real backend and SlotKVPool.
- WorkerBinding serializes all Facades using the same Worker and claims request
  IDs across those Facades. It stores ownership only, not duplicate lifecycle state.
- A borrowed Worker is not globally closed by runtime.close; only that Runtime's
  requests are cancelled/released. `owns_worker=True` is exclusive and calls
  worker.close on shutdown; attaching another Runtime then fails.
- Shared pool availability bounds new admission. Another Runtime remains waiting
  when all physical slots are occupied. Cross-Runtime fairness is not implemented.
- The shared lock spans worker calls. Cancellation from another thread waits for
  the step to return; there is no forced CUDA interruption. The real backend
  synchronizes before releasing cache references.
- Do not bypass this serialization with concurrent direct Worker/Executor calls.
  Direct Worker and ContinuousExecutor APIs remain caller-serialized.

## Failure handling

Normal completion, request errors, external terminal retirement, cancel and close
release managed request resources. Batch result contract failures mark affected
requests FAILED and retire them rather than retrying WAITING with old caches.
Only submitted IDs are cleaned; an unknown result ID never authorizes releasing
another request's cache.

Cleanup attempts every affected request. Errors are logged and retained in context
observations; a raised execution error remains the original exception with attached
cleanup_errors. Worker-returned execution errors are retained as execution_error
observations. Cleanup-only failures raise ResourceCleanupError. Failed cleanup is
not marked successful; close/cancel can retry after the underlying problem is fixed.
No implementation can guarantee zero slots if the backend itself refuses release.

Legacy Fake Workers without resource capabilities preserve their existing
ContinuousExecutor behavior. Managed real Workers receive the stronger contract.
RuntimeFacade's continuous mode requires that contract explicitly.

## Metrics

In managed continuous execution, RequestState.created_time is arrival, and
started_time is recorded before the Prefill worker call. Prompt counts come from
the actual input attention mask. First token time is recorded only when the
coordinator applies a generated token after backend synchronization, not when
Prefill returns. Finished time records terminal outcome.

Existing first_token_latency_ms remains first token minus execution start;
total_latency_ms remains terminal time minus execution start. Additive
arrival_ttft_ms includes waiting from creation. Additive tpot_ms uses
`(last_token_time - first_token_time) / (decode_tokens - 1)` and is null when
insufficient observations exist. Timestamps are host observations using the
existing clock convention, not CUDA kernel-only profiling. Legacy mode retains
its existing timing semantics; do not mix it with the new mode's Prefill-inclusive
execution boundary or historical GPU Benchmark measurements.

## Verification

CPU/Fake tests cover direct integration, distinct lengths, length completion,
waiting/prefilled/decoding cancellation, malformed result sets, input/sampling/
forward failures, cleanup failure/retry, shared ownership, capacity validation,
close idempotency and cross-thread cancellation serialization.

Real CUDA tests use cached Qwen2.5-0.5B-Instruct, FP32 and greedy generation.
Executed for this change: targeted Facade tests **25 passed**; complete non-slow
suite **390 passed, 11 deselected**; the new real Facade test plus existing real
batch-worker equivalence test **2 passed** on CUDA. Compileall and git diff
--check passed (existing line-ending warnings only).
The new Facade test cancels one of two prefilled requests, admits a new request
alongside active Decode, finishes with zero slots, and compares surviving token
sequences with independent single-request executions. Existing GPU batch=1/2
greedy equivalence is also executed. These bounded cases do not prove arbitrary
model/cache-layout compatibility, long-run memory stability or online safety.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_continuous.py tests/test_runtime_facade.py -q
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_real_runtime_continuous.py tests/test_real_batched_transformers_worker.py -m slow -v
```

Historical Mission 23.6/23.7 JSON/JSONL records are not modified or rerun. Resource
release synchronization can change future timing overhead; old numbers must not
be presented as measurements of this integration.
