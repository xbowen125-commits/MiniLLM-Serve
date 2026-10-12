# Offline Runtime Architecture Demo

Run from the project root using the existing environment with MiniLLM-Serve installed:

```powershell
.\.venv\Scripts\python.exe -m demo.runtime_demo
```

No model, tokenizer download, HTTP service, CUDA or Web UI is used. The demo
uses existing runtime components and local deterministic Fake Workers:

1. `RuntimeFacade`: submit one request, advance prefill and decode, inspect snapshots/results.
2. Continuous runtime: keep request A active, add B on the next cycle, execute
   Decode before new Prefill admission, and finish each independently.
3. `PagedContinuousBatchWorker` with `PagedKVCacheManager`: reserve a synthetic
   two-token prompt, grow two-token pages, and release resources at termination.

Terminal output is one JSON object per line. Events include `submit`,
`phase_history`, `request_state`, `continuous_step`, `kv_snapshot`,
`runtime_snapshot` and `cleanup`. Request token counts come from authoritative
RequestState; paged block IDs and capacity are bookkeeping, not real tensors.

`PREFILL` is transient within a synchronous runtime step: the demo prints the
recorded phase history as well as the current phase. It does not introduce
artificial phase transitions. Lifecycle status and execution phase are distinct.

Expected final observations:

- Facade request: FINISHED, two generated tokens.
- Continuous A/B: FINISHED, three/two generated tokens.
- Final active request count, simulated allocated bytes and registered KV requests: zero.

Snapshots contain wall-clock observation timestamps; they are not benchmark
measurements. Fake token IDs are not meaningful text. This demo does not show
real GPU continuous batching or PagedAttention, and does not modify online serving.
The caller closes both executor and wrapper because worker cleanup is caller-owned.

Architecture reference: [docs/architecture.md](../docs/architecture.md).

Minimal verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_demo.py -q
```
