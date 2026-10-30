# Paged Memory Source-Hash Erratum

Recorded on **2026-10-28** for Mission 24.4.2.14. This is a new public erratum,
not an alteration of a historical report or experimental artifact.

## Declared and currently available bytes

The unchanged [unified benchmark report](../benchmark_report.md) identifies
`benchmarks/runtime/results/paged_memory/paged_memory.json` as its source.

| Item | SHA-256 |
| --- | --- |
| Historical hash declared by that report | `f58963b9368134961706ef321096e70e9bc4bd525628054f4d1b56c724429e1a` |
| Current local artifact, 13,648 bytes | `83dc50db6e7dd3ee9b309b444f25e073b79d863f3dc6200fadd1d59a456fddbd` |

The actual byte hashes do not match. No matching original was found among
613 readable historical files in `benchmarks/results/` and
`benchmarks/runtime/results/`, or among 420 local Git blob objects. The search
excluded 128 pytest temporary directories and the disposable `.test-reports`
path; it did not search external backups or remote-only objects. Therefore
**the original is not located within the stated scope**, not proven lost.

Neither UTF-8 BOM removal nor CRLF-to-LF normalization yielded the declared
hash. These were in-memory diagnostic comparisons; no file was rewritten.
The cause, modification history and identity of the originally inspected bytes
remain **UNKNOWN**. A mismatch is not evidence of intentional fabrication.

## Metric correspondence

Read-only extraction from the current JSON agrees with the historical public
table on the following values. This was not a rerun of the experiment.

| Strategy | Requested tokens | Allocated tokens | Wasted tokens | Blocks | Utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contiguous | 12,288 | 30,720 | 18,432 | 30 | 40% |
| Paged | 12,288 | 12,288 | 0 | 768 | 100% |

The current artifact records seed 42, 30 requests, maximum length 1024,
16 tokens per block and growth step 16; cleanup allocated tokens are zero.
These are simulated capacity units, not GPU memory measurements. Agreement
of these fields does **not** establish that this is the historical source file.

The other two local sources referenced by the unified report still match its
declared byte hashes at this inspection:

| Source | SHA-256 |
| --- | --- |
| `continuous_batch/continuous_batch.json` | `1a7c932ad543ae0f2b43e87e5a07cbf2a1f2d90ed9f386d997f289930c73ba0a` |
| `scheduler_research/scheduler_research.json` | `6938c7530457113fb5561d78b0adfd2b6f6a08596a99005e0c10b013fb75c873` |

All three source files remain under the ignored local results directory.
This checkout publishes summaries and tools, not these raw private bundles.

## Consequence and next action

Paged Memory's historical source-byte provenance is **not closed**. The old
hash remains in the old report; the JSON is unchanged. Do not replace the hash,
delete the artifact, or rerun into that path to manufacture agreement.

Locate an original file with the declared byte hash and document its custody,
or publish a separately identified future simulation with new provenance and
an exclusive output path. Neither action is performed or authorized by this
erratum. [Current research restrictions](CURRENT_STATUS.md), including the
formal GPU Benchmark prohibition, remain in effect.
