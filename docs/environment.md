# Benchmark Environment Snapshot

Inspected locally for Mission 22.0 on 2026-10-14 (Asia/Shanghai):

| Field | Observed value |
| --- | --- |
| Python | 3.13.5 |
| PyTorch | 2.14.1+cpu |
| Transformers | 5.19.0 |
| PyTorch CUDA build | None (CPU build) |
| GPU used | None |
| Physical GPU identity | N/A; not queried for this CPU simulation |
| HEAD commit SHA | dc9db6e9cb2585bd7eb141717206586ec0e01fb3 |
| Source state | Dirty working tree; benchmark changes are uncommitted |

This is the current benchmark environment, not the historical CUDA 12.8 / RTX
5060 validation environment. The HEAD SHA alone does not identify uncommitted
sources. Each new result records Python/package versions, CUDA build, HEAD and
dirty state; no GPU initialization is needed to collect metadata.

Versions above are an observed environment snapshot, not a full dependency
lockfile or a declaration that other versions were tested. No packages were
installed, upgraded or downgraded in this fix. For comparable runs, use these
exact Python and package versions in a separate environment and retain its full
dependency inventory; do not replace the existing project environment.

Reproduce from project root:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.runtime.orca.report
```

Schema-v3 JSON, Markdown and per-case JSONL traces go to the Git-ignored
`benchmarks/runtime/results/orca-validity/` directory, preserving earlier output.
Numeric traces are deterministic under the same workload and virtual costs;
artifact timestamps and source-state metadata may change. This experiment uses
no model, GPU, CUDA kernels or selective batching.
