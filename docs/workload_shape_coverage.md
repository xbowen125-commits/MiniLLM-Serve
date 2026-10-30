# Workload shape coverage and admission proof

This is correctness/observability infrastructure, not a performance experiment.
No production Worker, Runtime, scheduler algorithm or arrival offset is changed.
The original protocol and its 25 ms lateness threshold remain frozen.

## Version 1 expected manifest

`expected-shapes.json` is created exclusively before the service starts. It binds
the prepared six-request trace (including tokenizer-verified prompt lengths),
configuration SHA-256, pinned model revision, dtype, device and dirty source-tree
fingerprint. Pinned model geometry is derived from the local `config.json`, which
is copied to `model-config.json` and hash-checked on replay. No weights are copied.

Each required entry declares mode, phase, batch size, input tensor shape, mask
shape, input padding width, cache sequence length before forward, every initialized
layer's key/value dimensions, derivation rule, request/step evidence and shape ID.
Identical exact shapes are deduplicated without losing their request references.

For a singleton prompt of P tokens and limit G, the conservative envelope is:

- Prefill: input/mask `[1,P]`, cache length zero, initialized cache tensors `[]`.
- Decode forward t in 1..G-1: input `[1,1]`, mask `[1,P+t]`, cache length
  `P+t-1`; layer tensors `[1,kv_heads,P+t-1,head_dim]`.

The first output token is sampled from Prefill logits. EOS/stop can shorten the
envelope. A missing max-token suffix is reported as missing, not erased using a
historical stopping point. This conservative obligation may overestimate actual
execution; it is not proof that every such suffix must occur.

Continuous arrival/forward timing and cache re-padding/regrouping are explicitly
recorded as an unresolved **required** interleaving obligation. A singleton
inventory is not claimed to enumerate Continuous execution. Unknown obligations
remain in the denominator of the admission decision, not hidden by a percentage.
The manifest deliberately does not invent an exhaustive batched cache-width model.

## Actual observations and exact matching

The benchmark-only forward observer records unique observation/call IDs, Run ID,
PID, monotonic time, stage, mode/phase, tensor shapes and cache lengths/dimensions,
actual model parameter dtype/device, model revision, configuration/source identity,
manifest SHA-256 and collector version. Failed forwards are retained but cannot
cover shapes. Python return is the execution boundary; this is not CUDA timing.
One-token prompts have ambiguous width-based phase inference and remain UNKNOWN.

Only successful warmup forwards from the declared process, between its hashed
start/end markers, qualify. Matching is exact across all declared shape fields
and every initialized cache layer. Missing/unreadable dimensions are UNKNOWN,
not wildcards. Additional undeclared shapes cause `SHAPE_UNDECLARED_OBSERVATIONS`
FAIL; they are preserved as evidence for a future versioned derivation rule.

## Independent replay and failure closure

`shape-proof-index.json` hashes the expected manifest, configuration, model
geometry, process metadata, warmup markers and raw observer journals. The verifier
validates containment, file SHA-256, current expected identities when supplied,
rebuilds the manifest, checks process/window identity, and recomputes set coverage.
Saved `shape-coverage.json` counts are never an input to that calculation.
The outer preflight artifact check additionally compares replay with the saved
proof and binds it to the prepared trace and current experiment identity.

Reason codes include `SHAPE_MANIFEST_INTEGRITY`, `SHAPE_ARTIFACT_INTEGRITY`,
`SHAPE_OBSERVATION_IDENTITY`, `SHAPE_REQUIRED_DIMENSIONS`,
`SHAPE_REQUIRED_COVERAGE`, `SHAPE_UNDECLARED_OBSERVATIONS`, and
`SHAPE_EXPECTATION_COMPLETE`. Existing preflight and formal-process attestation
requirements remain independent: a preflight process does not warm a future one.
Old proof formats and forward totals alone cannot establish shape coverage.

```powershell
Set-Location D:\AI_Lab\MiniLLM-Serve
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.preflight `
  --config benchmarks/runtime/online_http/configs/qwen_admission_v2.json `
  --execute --allow-gpu --scope full
# Read-only replay; replace with the new directory printed by preflight:
.\.venv\Scripts\python.exe -m benchmarks.runtime.online_http.shapes <new-directory>
```

Non-PASS exits with code 1. All new artifacts are exclusive; old results are
never rewritten. A self-consistent bundle is not a signed attestation against a
malicious author who can rewrite every artifact. Actual GPU execution and coverage
counts must be read from the new run, not inferred from these instructions.
