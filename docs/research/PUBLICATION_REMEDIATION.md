# Local Publication Remediation — Mission 24.4.2.14

Recorded on 2026-10-28 (Asia/Shanghai). This is a new local remediation record,
not a replacement for a historical experiment, the chat-only Mission 24.4.2.12
report, or a GitHub CI result. No commit or push is performed by this mission.

## Baseline and preservation

- Branch: `main`; HEAD: `dc9db6e9cb2585bd7eb141717206586ec0e01fb3`.
- Dirty baseline: 11 modified tracked files, 110 untracked files, nothing staged.
- Pre-remediation Git-visible inventory: 288 files, source SHA-256
  `1d5106ed794fed26e620566384f3e69f4d419ecae852f3e738eeb9a87d638e5d`.
  This identifies the tree before these edits, not the final publication tree.
- Private, exclusively created validation bundle:
  `benchmarks/runtime/results/publication-remediation-20261028-ec58ab69e2/`.
  It is ignored and is not an automatic publication asset.
- `baseline.json` stores per-file hashes of the visible tree and 613 readable
  historical files. It excludes 128 pytest temporary directories and disposable
  `.test-reports`; no claim is made about excluded temporary data.
- The four Mission 24.4.2.11 drafts, production source, tests, frozen
  configurations, raw evidence and old report hashes are preservation targets.
  Only public prose and publication-exclusion configuration are edited.

## Changes and open decisions

| Audit item | Local treatment |
| --- | --- |
| P01 | New [current status](CURRENT_STATUS.md) discloses chat-only review provenance, INTERNALLY_INCONSISTENT and unclosed F01/F02/F08 |
| P02 | README/architecture distinguish default legacy from opt-in Continuous JSON/SSE, actual Worker/Slot pool and separate simulations |
| P03 | Correct environment names, new-clone CPU installation, existing-venv preservation, launchers and status inspection |
| P04 | New [provenance erratum](PAGED_MEMORY_PROVENANCE_ERRATUM.md); original report and JSON remain unchanged, source identity UNKNOWN |
| P05 | Private directories, credentials, activation arrays and sidecars excluded; JSON/JSONL are not globally ignored |
| P06 | Historical/effective future email are non-noreply; no actual email printed, config changed or history rewritten; human approval remains required |
| P07 | CPU and static validation recorded below; no current GitHub-hosted CI or GPU experiment is certified |

## Environment matrix

| Scope | Python / dependencies | Meaning |
| --- | --- | --- |
| Docker/CI recipe | Python 3.12, PyTorch 2.11.0, Transformers 5.17.0, Accelerate 1.15.0; constraints-container.txt | CI installs CPU PyTorch; Docker inherits CUDA 12.8 torch and normally uses FP16 |
| Historical legacy reports | Python 3.14.7, torch 2.11.0+cu128, CUDA 12.8, Transformers 5.17.0, FP16 | Dated local measurements, not current compatibility proof |
| Later recorded Qwen CUDA correctness | torch 2.11.0+cu128, CUDA 12.8, Transformers 5.19.0, float32 | Historical offline/ASGI/TCP evidence; not new online performance admission |
| This remediation | Python 3.13.5; torch 2.11.0+cu128 with CUDA hidden; Transformers 5.19.0, Accelerate 1.15.0, FastAPI 0.143.0, Uvicorn 0.54.0, pytest 9.1.1 | CPU-only non-slow execution; not a CPU-wheel replacement or a Docker/CI environment test |

New-clone installation and model-service launch commands in README are statically
reviewed instructions. They are not claimed executed here. An existing CUDA
environment must not be overwritten by the separate new-CPU-environment recipe.

## Validation record

The existing venv was retained. `CUDA_VISIBLE_DEVICES=-1`, CPU device selection,
legacy default selection, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` were
enforced. The guard observed `torch.cuda.is_available() == False`; the installed
torch build still reports CUDA 12.8. No dependency was installed and no real
model or GPU inference was executed.

| Attempt | Actual result | Preserved record |
| --- | --- | --- |
| Full non-slow run inside sandbox | Interrupted after the first Owner case stopped making progress; no completed pytest summary, not a PASS | `first-attempt-interrupted.json`, `pytest-output.txt`, original `pytest-temp/` |
| Bounded single-case sandbox diagnosis | Timed out at 100 seconds; faulthandler at 20 seconds showed stdlib `_fallback_socketpair` blocked in `accept` during asyncio event-loop creation, before the Runtime coroutine ran | `diagnostic-result.json`, `diagnostic-output.txt`, `diagnostic-temp/` |
| Authorized full run outside sandbox, same CPU/offline restrictions | **836 passed, 13 deselected, 0 failed; 80.49 seconds**, process exit 0 | `cpu-environment-unsandboxed.json`, `pytest-unsandboxed-result.json`, `pytest-unsandboxed-output.txt` |

The successful command was:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -m "not slow" -q `
  -o faulthandler_timeout=60 `
  --basetemp benchmarks/runtime/results/publication-remediation-20261028-ec58ab69e2/pytest-unsandboxed-temp
```

The environment restrictions above are required; this is not a command to
repeat into that already used `--basetemp` directory. Choose a new exclusive
directory for another run. The faulthandler option emits diagnostics; it does
not skip tests or change assertions. No tests or production source were edited
to obtain the result. Sandbox attempt records are retained, not counted as
successful runs or erased by the later successful run. The stack plus the
successful unchanged-code run outside the sandbox localize the observed block
to that execution context; they do not establish every underlying OS cause.

Additional checks:

- Read-only AST parsing: all 234 candidate Python files, no syntax errors.
- Initial Markdown file-link check: 42 Markdown files, 90 local file links,
  no missing or nonpublic file targets; final checks are saved separately.
  The subsequent anchor check found an existing historical-report link to
  `README.md#quick-start-with-docker-compose`. A compatibility anchor was added
  to README rather than rewriting that historical report. The first finding
  remains in `final-validation.json`; the corrected check is recorded in
  `final-validation-v2.json`, not written over the first result.
- Git exclusion probes: 25 excluded and 7 retained examples, all matched;
  no previously visible candidate was newly hidden. Public JSON/JSONL fixture
  paths remain eligible. New raw `.npy`/`.npz` files are private by default;
  future intentional synthetic array fixtures need specific privacy review.
- Docker exclusions were statically reviewed and the deployment contract tests
  were included in the full CPU suite. No Docker build/context export was run.
- Post-test preservation: all 613 historical file hashes and the four draft
  hashes unchanged; no production/test/frozen configuration changes and the
  Git index unchanged. See `post-cpu-preservation.json`.
- `cpu-tested-tree.json` captures the post-regression tree before final
  prose-only edits, SHA-256
  `0716f62e458362b6552114af2614c1c9dc357092106c3620cbe3bb8d62a1ed3e`.
  It was not sampled at test startup and is not a GPU admission certificate.
  Final prose/command edits receive fresh static checks; the source/test files
  exercised by the CPU run remain byte-identical.

The final check covers all 291 public candidates (including these three new
documents), 42 Markdown files and all 234 Python files. Detailed hashes and
the full un-staged status list are in the ignored validation bundle; they are
not a request to stage every file. The only edited pre-existing paths are
`.gitignore`, `.dockerignore`, `README.md`, `docs/architecture.md`,
`benchmarks/reports/environment.md` and `benchmarks/runtime/online_http/README.md`.
The added paths are `docs/research/CURRENT_STATUS.md`,
`docs/research/PAGED_MEMORY_PROVENANCE_ERRATUM.md` and this remediation record.

The new-clone installation, service launch examples, Docker/Compose execution,
GitHub-hosted CI, external links and GitHub Mermaid rendering were **not**
executed or certified. Logs and detailed manifests remain in the ignored local
bundle; public summaries do not expose actual author email or private tensors.

## Publication boundary

Local remediation result: **READY_WITH_CHANGES**, for an explicitly scoped
research-prototype publication, not automatic release authorization. Pending
human decisions include historical/future author email disclosure and approval
of the exact staged file list. Fresh-environment and Hosted CI validation remain
unverified and must not be described as passed. Neither README cleanup nor successful
CPU tests repairs the draft evidence contract or closes Paged Memory provenance.
The project may be presented as a research prototype with these limitations;
formal online GPU Benchmark remains prohibited and all existing admission
FAIL/UNKNOWN states remain unchanged. No entire-Runtime safety, liveness or
termination proof is asserted.
