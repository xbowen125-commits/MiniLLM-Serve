# Mission 24.4.2.10 — Transition refinement and KV evolution

Conclusion: **PARTIALLY CLOSED**. This is independent analysis of saved traces,
not a new CUDA experiment, a change to Runtime, or a performance certificate.

## Immutable target and current auditor

Input: `benchmarks/runtime/results/online_http/identity244429-20261025-gpu-b`.
The run has PID 10920, 1,743 events, six requests and source fingerprint
`6a0db4d2b956e69a845074d2c7f38869236b86987218c7b88c136a133678587c`.
The raw trace SHA-256 is
`5821c63ab17864a5fdc33734c9e233db876f4ce532674aea713178389d07fc8f`.

The verifier authenticates required raw files against the original artifact
index, checks the source inventory self-hash, frozen pristine/prepared workload
hashes and unchanged frozen fields. Current auditor source is recorded separately.
Adding an auditor changes the full dirty-tree fingerprint; the saved run remains
historical evidence, never a new current-source GPU certificate. Matching the
production rule-file hashes does not override that distinction.

## Forward evidence and physical geometry

| Fact | Raw source / association |
| --- | --- |
| Runtime call ID | forward_begin.forward_id; forward_end uses that exact UUID |
| Model call ID | separate observer.jsonl namespace, no direct cross-ID link: UNKNOWN |
| Iteration / phase | collector iteration_id and explicit phase_id |
| Plan / Executor / Worker | iteration ID, phase ID, actual ordered members |
| Forward row ID | ordered actual members from input builder or cache-object lookup during merge |
| Stored physical KV width | actual Slot cache.get_seq_length / keys.shape |
| Merged input/output shapes | actual Forward cache_before/cache_after layers |
| Live row after execution | phase-linked worker_end snapshot, cache/Slot/context object reads |
| Release / original reference | explicit release ID and original RequestState observation |

The new report's `call_id` is explicitly an alias in the Runtime-forward UUID
namespace, not a fabricated join to the independent model observer's call_id.
Request association never uses timestamp proximity or list position between
different logs.

For each Forward, physical checks include:

* all-layer keys/values batch, head, physical sequence and head-dimension shapes;
* input/mask/cache width equations and after = before + input width;
* Prefill max(prompt lengths) left padding and empty input cache;
* Decode maximum stored physical row width, width-one input and ready flag;
* Plan/Executor/Worker membership and relative row order;
* independent live Slot/cache ownership and split physical row widths;
* terminal requests never appearing as active Decode inputs.

The geometry (24 layers, 2 KV heads, head dimension 64) is directly observed in
the target tensors. It is a tensor-consistency observation, **not** a retroactive
pinned-model geometry attestation. The original geometry UNKNOWN is not edited.

Example: the first Prefill contains prompts of 37 and 72 tokens; both stored
cache rows have physical width 72. Physical width is not effective Token length.
Decode mask.sum/nonzero values are absent. They are not reconstructed as actual
observations from prompt length plus generated count. Split row-source indices
or tensor-value fingerprints are also absent: metadata row ownership is checked,
but equal shapes cannot prove correct copied tensor values.

Consequently physical equations can PASS while full FORWARD_KV_EVOLUTION remains
UNKNOWN. A terminal row released inside Worker before worker_end has no saved
post-Forward row tensor; that missing row observation also remains UNKNOWN.

## Abstract relation and projection

The existing Taxonomy v2 is reused without changing transitions:

* N: not submitted to Runtime (HTTP admission can still be queued).
* W: directly observed waiting Context, zero tokens.
* F: directly observed post-Prefill Context, pending_first=True, zero tokens.
* D: active Decode Context, pending_first=False, positive emitted count.
* T: terminal Context **only when resource retirement is directly supported**.

The source-derived execution projection distinguishes HTTP waiting from Runtime
waiting. Context absence is not silently interpreted as N or T: a submitted
Context can disappear only with explicitly paired successful acknowledge and
direct retired snapshots. Missing Contexts, unknown IDs, missing counts/phases
and terminal-with-retained-resource combinations remain UNKNOWN.

| Concrete boundary | Relation / proof obligation |
| --- | --- |
| Admission / queue | actual identity and FIFO evidence rules; not a v2 execution arrival yet |
| Owner enqueue/start | actual Delivery/Future/RequestState identity relation |
| Runtime submit | explicit begin/end snapshots, v2 ordered-arrival if permitted |
| Prefill | P2 physical evidence; explicit step-boundary iteration relation |
| First-token Decode | F2 samples saved logits; token count advances with no Forward |
| Ready Decode | D2 merges maximum stored widths and forwards one token |
| Regrouping | R2; survivors and newly prefilled rows keep their own stored widths until merge |
| Early stop | iteration followed by the observed terminal stop edge, not forced length |
| Cancellation / error | actual lifecycle guards and joint retirement overlay; absent target-path occurrences are not newly validated |
| Release / retry | existing ResourceStatus / JointState rules, Slot/ref/result confirmations separately |
| Acknowledge | retired Context removal; execution projection stutters at T |
| Lease finalization | explicit completion/resource confirmation before release |
| Shutdown | borrowed/owned close contracts and direct final drain evidence |

At a step boundary, the verifier matches one permitted model edge, optionally
followed by terminal edges dictated by directly observed after-lifecycles.
Only terminal suffixes for changed terminal requests are considered; arbitrary
extra arrivals or larger BFS budgets cannot manufacture a witness. Failed
partial macros remain UNKNOWN rather than being restored to WAITING in the model.

Internal phase and resource operations are **not** all declared stuttering.
Read-only observation emission does not mutate the execution projection, but
that fact does not prove adjacent global states equal across other threads.
The event report retains local guards, last-direct-observation field sources,
macro witnesses where available, and explicit UNKNOWN reasons elsewhere. It
does not promote local guard PASS into per-instruction v2 equivalence.

Base v2 cannot encode terminal-with-pending-Slot/reference. The existing resource
overlay does encode release_requested/retry_required/retired/unknown. The two
contracts are reported separately; a retained resource is not erased to satisfy
v2's T(width=0) representation. Cleanup failure need not eventually recover.

## Recalculation and tests

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.trace_refinement `
  --input benchmarks/runtime/results/online_http/identity244429-20261025-gpu-b `
  --output benchmarks/runtime/results/online_http/refinement2444210-manual-new `
  --config benchmarks/runtime/online_http/configs/qwen_admission_v2.json

.\.venv\Scripts\python.exe -m pytest tests/test_trace_kv_refinement.py -q
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
```

Output is exclusive and outside the immutable input. Reports include per-Forward
checks, per-boundary model paths, every raw event's rule/reference/partial
observations, unclosed reasons, unchanged joint retirement results and hashes.
No model is loaded by this command. Deterministic counterexamples are saved by
the tests as synthetic raw/verdict files, explicitly not CUDA evidence.

An initial test found that missing member IDs were being treated like an empty
batch and wrongly classified FAIL. Missing evidence now stays UNKNOWN; explicit
wrong members still FAIL. The original failed test output is retained. A Windows
UTF-8 read error in the new auditor was fixed without editing the input bytes.

## Proof boundary and admission

The observed normal trace can satisfy the **execution-boundary projection** of
Taxonomy v2 without proving full resource/admission refinement. Tensor-value
row lineage, effective mask lengths, model-call IDs, microstep transfer relations
and current-source GPU certification remain separate unknown obligations.

Cancellation/failure/retry tests validate specified synthetic/joint-model cases;
the normal six-request trace does not newly establish GPU behavior for them.
No state search was expanded and no all-path or entire-Runtime exhaustion is
claimed. Safety statements are limited to checked predicates. Liveness and
termination require retry fairness, eventual successful observable cleanup and
consumer finalization; permanent failures admit nonterminating execution.

All prior PREFLIGHT blockers, global schedule equivalence and finite-state
exhaustion obligations remain in force. Formal GPU Benchmark remains prohibited.
