# Benchmark Environment and Reproduction Notes

This document describes the environment recorded by the private local benchmark artifacts and the controls required to reproduce them. It deliberately excludes local absolute paths, user names, credentials, model weights, and full logs.

## Recorded environment

| Field | Engine benchmark | HTTP benchmark | Final stress test |
| --- | --- | --- | --- |
| Timestamp | 2026-10-09 12:13:43 +08:00 | 2026-10-09 23:38:38 +08:00 | 2026-10-10 00:03:23 +08:00 |
| Metric scope | Model engine | HTTP client observed | Bounded service stability |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` |
| Device / precision | CUDA / `torch.float16` | CUDA / `torch.float16` from `/health` | CUDA / `torch.float16` from `/health` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU | Same | Same |
| Python | 3.14.7 | 3.14.7 | 3.14.7 |
| PyTorch | 2.11.0+cu128 | 2.11.0+cu128 | 2.11.0+cu128 |
| CUDA runtime | 12.8 | 12.8 | 12.8 |
| Transformers | 5.17.0 | 5.17.0 | 5.17.0 |
| Accelerate | Not stored by this artifact | 1.15.0 from `/health` | 1.15.0 from `/health` |
| Operating system | Windows 11 | Windows 11 | Windows 11 |
| KV Cache | Both off and on | On | On |

The raw reports contain a more specific Windows platform build string. It is unnecessary for public reproduction and is omitted to minimize machine fingerprinting.

## Provenance

The public values in [benchmark-summary.md](benchmark-summary.md) are extracted from these ignored local artifacts:

- `benchmarks/results/mission08-engine-kv.json`
- `benchmarks/results/http-smoke.json`
- `benchmarks/results/stress-disconnect-fixed.json`

`benchmarks/results/stress-bounded.json` is a retained pre-fix diagnostic artifact. Its non-idle scheduler result was superseded by `stress-disconnect-fixed.json` after disconnect cleanup was fixed and verified.

The artifacts record timestamps, runtime versions, parameters, summaries, and request samples, but no Git commit SHA. Consequently, the reports are historical evidence that matches the current benchmark schema; they are not a signed or commit-addressed provenance chain.

## Reproduction controls

For a comparable run:

1. Use the same model, CUDA device, FP16 precision, prompt and generation lengths, seed, warmup count, and measured repetition count.
2. Load and warm the model before steady-state HTTP measurements. Do not include model download or startup in inference latency.
3. Keep `MINILLM_MAX_CONCURRENT_REQUESTS=1` when comparing with the published HTTP and stress data. Client concurrency may be higher, but that does not increase GPU execution concurrency.
4. Keep KV Cache state explicit and record scheduler queue capacity and timeout settings.
5. Run the engine benchmark in a separate process when comparing CUDA peak memory, because it resets PyTorch peak statistics for each measured run.
6. Record laptop power mode, temperatures, and background load separately if tighter repeatability is required; the historical artifacts do not contain that telemetry.

## Service setup

PowerShell environment matching the published service-side settings:

```powershell
$env:MINILLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
$env:MINILLM_DEVICE = "cuda"
$env:MINILLM_DTYPE = "float16"
$env:MINILLM_USE_KV_CACHE = "true"
$env:MINILLM_MAX_CONCURRENT_REQUESTS = "1"
$env:MINILLM_QUEUE_CAPACITY = "16"

.\.venv\Scripts\python.exe -m uvicorn minillm.server:app `
  --host 127.0.0.1 `
  --port 8000 `
  --workers 1
```

The raw HTTP artifacts record the running service health and concurrency, but not every scheduler timeout environment variable. Capture `/health` and `/scheduler/status` together with future reports when exact service configuration provenance is required.

## Engine reproduction

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_kv_cache.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --device cuda `
  --dtype float16 `
  --prompt-lengths 32 `
  --generation-lengths 16 `
  --warmup 2 `
  --repeats 10 `
  --seed 42 `
  --offline `
  --output-prefix benchmarks\results\mission08-engine-kv
```

This command is reconstructed from fields stored in the engine artifact. It writes private machine-specific output back to the ignored `benchmarks/results/` directory.

## HTTP reproduction

With the service already loaded and warm:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_http.py `
  --base-url http://127.0.0.1:8000 `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --prompt-lengths 32 `
  --generation-lengths 16 `
  --concurrency 1,2,4 `
  --mode both `
  --kv-cache-state on `
  --warmup 1 `
  --repeats 3 `
  --seed 42 `
  --max-total-requests 24 `
  --output-prefix benchmarks\results\http-smoke
```

This exactly represents the fields recorded by the HTTP artifact. It produces six cases, six warmup requests, and 18 measured requests.

## Stress reproduction

```powershell
.\.venv\Scripts\python.exe benchmarks\stress_http.py `
  --base-url http://127.0.0.1:8000 `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --requests 20 `
  --concurrency 4 `
  --disconnect-requests 2 `
  --prompt-tokens 64 `
  --max-tokens 64 `
  --timeout 180 `
  --cleanup-timeout 30 `
  --output-prefix benchmarks\results\stress-disconnect-fixed
```

The final stress artifact stores request count, concurrency, disconnect count, guard limits, and request timeout. It does not store prompt length, maximum generation length, cleanup timeout, or seed; the command uses the current script defaults for those fields. Future artifacts should persist those arguments before claiming exact historical reproducibility.

## Docker smoke reproduction

```powershell
docker compose build --pull
docker compose up -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
docker compose down --timeout 300
```

The existing Docker acceptance record is functional only: image build, dependency check, health, Chinese non-streaming generation, 76 SSE chunks ending in `[DONE]`, and graceful shutdown were observed. It has no structured benchmark artifact, so no latency, throughput, token, or memory result is published from that run.

## Report retention and privacy

- Keep `benchmarks/results/` ignored. Raw request samples and host metadata are useful locally but are not publication artifacts.
- Commit only curated Markdown under `benchmarks/reports/`.
- Do not embed user directories, environment files, access tokens, model caches, or generated content in public reports.
- A new performance claim should include the raw artifact, a privacy-reviewed summary, the exact source revision, and enough repetitions for its intended statistical use.
