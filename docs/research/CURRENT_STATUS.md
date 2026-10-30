# Current Research and Publication Status

Publication disclosure recorded on **2026-10-28** (Asia/Shanghai), Mission 24.4.2.14.
This is an Experimental LLM Serving & Runtime Research Prototype.
Publishing its source does not authorize formal online GPU benchmarking or
certify a production serving system.

## Implemented paths and their limits

| Area | Implemented boundary | Evidence boundary |
| --- | --- | --- |
| Legacy HTTP JSON/SSE | Default `MINILLM_EXECUTION_MODE=legacy`; `GenerationEngine`, request-local cache, bounded FIFO admission | Historical functional tests; not a current production-performance certificate |
| Continuous HTTP JSON/SSE | Explicit `continuous` mode; one owner drives RuntimeFacade, IterationScheduler, ContinuousBatchExecutor and BatchedTransformersWorker | Experimental CPU/CUDA backend, batch size 1 or 2; synchronous safe-boundary cancellation |
| Continuous resource ownership | RequestState is shared across admission and Runtime; SlotKVPool owns real request cache rows; cleanup retries preserve original request identity | Terminal status alone does not prove retirement; unknown or failed cleanup cannot justify idle/drain |
| Offline Runtime/Fake Worker | Bounded lifecycle, scheduling and virtual-time experiments | Simulated costs are not GPU throughput or latency |
| Paged KV memory | Synthetic capacity, BlockTable mappings and block lookup | Not PagedAttention computation, a GPU allocator or an attention kernel |
| Evidence contract | Four Mission 24.4.2.11 documents, all `draft-0.1` | Unapproved design, not an implemented evidence validator or admission certificate |

The [current configuration](../../src/minillm/config.py) requires KV cache and
`1 <= admission_limit <= continuous_batch_size <= 2` in continuous mode.
The [dedicated launcher](../../src/minillm/continuous_server.py) requests shutdown
before Uvicorn waits for open handlers. The generic Uvicorn launcher does not
provide that early hook. The shipped Compose setup remains the legacy path;
local Continuous validation is not Continuous-container acceptance.

Cancellation cannot interrupt a CUDA kernel. Large forwards delay cancellation.
Permanent cleanup refusal may prevent drain indefinitely. Long soak, forced
termination and independent-process signal behavior are not fully validated.

## Mission 24.4.2.12: review provenance and open findings

**Recorded consistency verdict: `INTERNALLY_INCONSISTENT`.**

Source: the Mission 24.4.2.12 audit delivered in the project's Codex conversation,
also cited by the subsequent GitHub Publication Readiness Audit and the
Mission 24.4.2.14 instructions. At this remediation's baseline inspection,
**no Mission 24.4.2.12 report file existed in the repository**. This page is a
new disclosure and summary; it is not the original report, a backdated copy,
or an independently authenticated transcript. A checkout cannot independently
recompute the identity of that chat-only report. No historical report file has
been invented or overwritten.

The original drafts remain unchanged:

- [Evidence contract](mission_24_4_2_11/evidence-contract.md)
- [Trace schema](mission_24_4_2_11/trace-schema.md)
- [Acceptance matrix](mission_24_4_2_11/acceptance-matrix.md)
- [Design review record](mission_24_4_2_11/design-review-report.md)

The review concerns the **draft evidence contract**. It is not a finding that
production generation has failed. In particular:

| Finding | Unclosed issue | Publication treatment |
| --- | --- | --- |
| F01 | Schema's terminal-before-release chain conflicts with a supported path in which Worker releases before Coordinator applies the finished result | Disclose the contract-ordering conflict; no generation/lifecycle change in this mission |
| F02 | Event payload types, conditional requirements and compatibility rules are not fully executable or uniquely specified | Keep the schema draft; do not call its checker implemented |
| F08 | Required producers, expected spans and the set of GPU work covered by completion evidence are not closed | Preserve UNKNOWN; a complete-looking trace is not proof of complete observation |
| Other findings | Client timing evidence, verdict precedence, duplicate handling, content canonicalization, token accounting, retirement scope, call cardinality, review/authorization assets and proof scope | Remain open; this publication remediation does not resolve F03-F07 or F09-F12 |

## Benchmark admission remains blocked

Historical CUDA correctness, offline timing and TCP tests have different scopes.
They are not formal online performance proof for the current dirty source tree.
HEAD alone does not identify uncommitted code; future evidence must bind its
actual source inventory, configuration, model revision and process.

The saved Mission 24.4.2.10 `combined-admission.json` remains `FAIL`,
`admitted=false`. Its historical and current-unknown scopes are not merged:

| Obligation | Retained status / scope |
| --- | --- |
| `PREFLIGHT_DECLARED_PASS` | Historical FAIL, not re-certified |
| `PREFLIGHT_WARMUP_SHAPES_COVERED` | Historical FAIL, not re-certified |
| `PREFLIGHT_WARMUP_COVERAGE` | Historical FAIL, not re-certified |
| `PREFLIGHT_SHAPE_PROOF_RECORDED` | Historical FAIL, not re-certified |
| `FORWARD_KV_EVOLUTION`, `CURRENT_SOURCE_BINDING` | UNKNOWN in the saved audit; no new CUDA certificate here |
| `TAXONOMY_V2_STEP_RELATION`, `FROZEN_SCHEDULE_EQUIVALENCE`, `FINITE_STATE_EXHAUSTION` | UNKNOWN; local trace conformance is not bisimulation or whole-Runtime exhaustion |
| `CACHE_OWNERSHIP`, `OBSERVED_ABSTRACT_STATE_DOMAIN`, `V2_CLEANUP_RETRY_REPRESENTATION` | Unclosed model/evidence obligations |

The saved evidence path is
`benchmarks/runtime/results/online_http/refinement2444210-20261025-a/combined-admission.json`,
SHA-256 `3d3e4ce22d0f23ac6076d71b4fcaf578ac9982308fb5ebe12b6616c449f95b6c`.
It is an ignored local artifact, not a distributed or current-source certificate.
The [public trace report](../trace_kv_refinement.md) explains the proof boundary.

Older documents that suggest entering a later mission describe historical
checkpoints. They do not override this explicit current prohibition:
**formal online GPU Benchmark is not admitted**. Publication, CPU test success,
design completion and a successful demo cannot lift it.

## Provenance, environments and validation

- The [Paged Memory provenance erratum](PAGED_MEMORY_PROVENANCE_ERRATUM.md)
  records an unresolved historical byte-hash mismatch. Matching metrics are
  not a replacement for matching source bytes.
- [Historical published environments](../../benchmarks/reports/environment.md)
  are dated records. Docker/CI use Python 3.12 and constrained Transformers
  5.17.0; later local CUDA correctness used Transformers 5.19.0 and float32.
  These are separate environments, not one compatibility certificate.
- [Local remediation validation](PUBLICATION_REMEDIATION.md) distinguishes
  actually executed CPU checks from commands merely documented for future use.
- Historical Git author email is non-noreply; effective future Git email is
  also non-noreply. Accepting disclosure remains a human decision. No identity
  or history is rewritten, and no actual email is printed here.

## Future work, not implemented promises

Contract amendments, additional observational hooks, independent content/KV
validation, stronger model correspondence and new correctness traces require
separate review. No completed full Runtime proof, unconditional cleanup
termination, production throughput, PagedAttention, prefix cache, chunked
prefill, tensor parallelism or new CUDA kernel is claimed.
