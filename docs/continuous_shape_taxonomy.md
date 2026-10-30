# Continuous shape taxonomy v2 and closure boundary

Mission 24.4.2.3 is offline evidence analysis, not a GPU performance experiment.
It leaves the v1 manifest, matching rules and admission decisions unchanged.

## Evidence scope

Input: `benchmarks/runtime/results/online_http/preflight244422-full-gpu-20261021-a/`.
The historical v1 proof is independently replayed and its original artifact
hashes verified before classification. Every extra shape retains **all** raw
JSONL line numbers, observation/call IDs, PID, run ID, stage and timestamp.

The v1 collector does **not** record worker input request IDs, per-row logical
Prompt/cache lengths, iteration ID or slot IDs. These fields are explicitly
null/UNKNOWN. Terminal request records cannot reconstruct batch membership by
timestamp proximity. Conditional cohort examples below are not observed members.

## Rules and source

| Rule | Preconditions | Exact consequence | Implementation |
| --- | --- | --- | --- |
| P2 | Valid waiting rows; group B within capacity | W=max Prompt lengths; left-padded input/mask `[B,W]`; empty cache before forward; split rows retain width W | `BatchInputBuilder.build_prefill`; `TransformersBatchWorker.prefill_batch` |
| F2 | Newly prefilled, pending-first session | First token samples saved logits; no new forward; width unchanged | `TransformersBatchWorker.decode_batch` |
| D2 | Ready sessions, not pending-first | K=max stored cache widths; left-pad all layers to K; input `[B,1]`, mask `[B,K+1]`, cache-before layers `[B,H,K,D]`; split rows retain K+1 | `_merge`, `_split`, `decode_batch` |
| R2 | Members leave or newcomers prefill | Removing a request does not shrink a survivor's stored padding. Rejoining takes max width. Admission uses capacity at plan selection, not slots freed later in that same iteration | `IterationScheduler.build_plan`; `BatchExecutor.execute_plan` |
| S2 | Serial worker with validated slot capacity | Each stored request cache has batch dimension 1. Merged batch dimension is the ready group size, not the slot ID or total occupied slots | `SlotKVPool`; `_split` |

The taxonomy records SHA-256 for all six source files. A pre-edit file manifest
whose aggregate fingerprint matches the old preflight can bind those rules to
historical source; absent that evidence, historical-source binding stays UNKNOWN.

## Finite over-approximation, not complete reachability

For a drained epoch with the fixed six requests, each submitted at most once:

1. Initial maximum stored cache width is at most `Pmax`.
2. A ready forward adds at most one to that maximum and consumes at least one
   generated-token allowance. Total ready forwards are at most `sum(G_i-1)`.
3. Joining takes a maximum, while first-token sampling, cancellation and release
   do not increase width. Failed terminal requests are not retried/re-prefilled.
4. Hence width is at most `Pmax + sum(G_i-1)`: **83 + 170 = 253** here. B is
   at most 2; L/H/D are fixed by the pinned model. The tensor-shape envelope is
   finite. This is a conservative upper bound, not a list of reachable widths.

The bound is conditional on this workload and supported full-attention cloned
cache backend. It is not an assertion about unbounded arrivals, arbitrary Prompt
lengths, retries, sliding caches or other models.

The deterministic proof-model enumerator explores ordered arrivals, iteration
execution, first-token handling, normal finish and safe-boundary cancellation /
failure. Cache growth and regrouping are explicit. It has a fixed state budget;
exhaustion of that budget is UNKNOWN, not PASS. Even full exploration of a small
synthetic graph does not establish equivalence to wall-clock arrivals, HTTP
admission or failures inside partial worker installation. These remain separate
proof obligations.

## Manifest version difference

V2 keeps every v1 required shape. It separately derives conditional cohort
shapes by enumerating Prompt/limit combinations, independently of observed data.
For a cohort admitted together, decode step t has K=max(P)+t-1 while all rows
remain active; t ranges from 1 to min(G)-1. Each derivation names its assumptions.
These conditional examples are **not** added to a production allowlist and do
not erase the required interleaving obligation or claim frozen-schedule legality.

Local exact tensor conformance may explain a v1 rule omission. It cannot prove
the historical request path, exhaustive shape coverage or warmup of a future
formal process. Repeated observations retain all IDs; no dimensions are rounded,
normalized away or matched by wildcard.

## Replay

```powershell
Set-Location D:\AI_Lab\MiniLLM-Serve
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.taxonomy_audit `
  --input benchmarks/runtime/results/online_http/preflight244422-full-gpu-20261021-a `
  --admission benchmarks/runtime/results/online_http/admission244422-full-gpu-20261021-a.json `
  --baseline benchmarks/runtime/results/online_http/taxonomy244423-source-baseline-20261021-a.json `
  --output benchmarks/runtime/results/online_http/taxonomy244423-replay-NEW `
  --max-states 2000
```

Every output directory must be new. The tool creates versioned taxonomy,
conditional manifest, diff, evidence schema, per-shape classification, bounded
enumeration and Markdown report with a file-hash index. No models or HTTP
requests are executed. Canonical semantic hashes and serialized file hashes are
reported separately. CLI exit 1 means admission remains blocked; local tensor
rule PASS does not override the unchanged v1 gate.

## Minimum remaining proof

An isolated observer would need actual worker membership, per-row widths and
pending-first flags, iteration/plan IDs and slot association at forward boundaries
to verify the historical path of a new run. No such production observer is added
here. Full closure also needs equivalence of the finite transition model to
owner/admission and within-phase error paths, followed by process-local warmup
coverage. More warmup repetitions or a fixed random seed are not that proof.
