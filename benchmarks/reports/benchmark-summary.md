# MiniLLM-Serve Benchmark Evidence

This report publishes a privacy-reviewed summary of local measurements. It is a historical snapshot, not a production capacity claim and not a guarantee for other hardware. Machine-specific raw artifacts remain under `benchmarks/results/` and are intentionally excluded from Git.

## Evidence inventory

| Scope | Private source artifact | Recorded at | Evidence used here |
| --- | --- | --- | --- |
| Engine benchmark | `benchmarks/results/mission08-engine-kv.json` | 2026-10-09 12:13:43 +08:00 | Engine latency, token throughput, and per-run CUDA peak memory |
| HTTP benchmark | `benchmarks/results/http-smoke.json` | 2026-10-09 23:38:38 +08:00 | Client-observed latency, request throughput, queue wait, and success counts |
| Stress test | `benchmarks/results/stress-disconnect-fixed.json` | 2026-10-10 00:03:23 +08:00 | Bounded load, disconnect cancellation, FIFO observation, cache isolation, and final idle state |
| Docker smoke test | Manual acceptance record; no structured result artifact | Date not recorded | Build, health, inference, streaming completion, and graceful shutdown only |

The earlier `benchmarks/results/stress-bounded.json` captured the pre-fix cleanup defect and is superseded by `stress-disconnect-fixed.json`. It is retained locally for diagnosis but is not presented as a successful result.

## Test environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Engine precision: FP16 (`torch.float16`)
- Python: 3.14.7
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- Transformers: 5.17.0
- Accelerate: 1.15.0 in the HTTP service health record
- Operating system: Windows 11

See [environment.md](environment.md) for provenance, configuration, and reproduction caveats.

## 1. Engine benchmark

Scope: direct model-engine measurement. Both cases used prompt length 32, maximum generation length 16, greedy decoding, seed 42, two warmups, and ten measured repetitions. Model loading was outside the measured runs.

| KV Cache | Measured samples | Completion tokens, median | TTFT, median | Prefill, median | TPOT, median | Tokens/s, median | Total time, median | CUDA peak | Peak increment |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | 10 | 16 | 24.237 ms | 23.811 ms | 24.654 ms | 40.267 | 397.348 ms | 978.482 MB | 27.034 MB |
| On | 10 | 16 | 24.678 ms | 24.216 ms | 22.723 ms | 42.516 | 376.333 ms | 961.510 MB | 10.062 MB |

These two local cases are a controlled comparison, not proof of a general performance gain. Laptop power limits, thermals, clocks, prompt length, output length, and model architecture can change the result.

Equivalent reproduction command reconstructed from the parameters stored in the raw report:

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

## 2. HTTP benchmark

Scope: client-observed HTTP and SSE behavior against the already loaded service. The run used KV Cache on, prompt length 32, maximum generation length 16, seed 42, one warmup per case, and three measured repetitions per case. The planned request count was 24: six warmups plus 18 measured requests. All 18 measured requests succeeded.

| Mode | Clients | Samples | Success | Request throughput | E2E P50 / P95 / P99 | TTFT P50 | TPOT P50 | Tokens/s P50 | Queue wait P50 / P95 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Streaming | 1 | 3 | 3 | 1.334 req/s | 756.923 / 766.560 / 767.416 ms | 53.379 ms | 45.682 ms | 21.138 | 0.000 / 0.000 ms |
| Streaming | 2 | 3 | 3 | 1.359 req/s | 1453.673 / 1498.958 / 1502.984 ms | 756.289 ms | 46.492 ms | 11.007 | 700.399 / 744.413 ms |
| Streaming | 4 | 3 | 3 | 1.262 req/s | 1623.728 / 2301.023 / 2361.227 ms | 824.654 ms | 46.683 ms | 9.854 | 763.384 / 1523.047 ms |
| Non-streaming | 1 | 3 | 3 | 1.356 req/s | 735.808 / 746.518 / 747.470 ms | N/A | N/A | 21.745 | 0.000 / 0.000 ms |
| Non-streaming | 2 | 3 | 3 | 1.380 req/s | 1432.398 / 1447.129 / 1448.439 ms | N/A | N/A | 11.170 | 697.460 / 712.362 ms |
| Non-streaming | 4 | 3 | 3 | 1.484 req/s | 1320.828 / 1949.258 / 2005.119 ms | N/A | N/A | 12.114 | 582.580 / 1244.667 ms |

TTFT and TPOT are intentionally N/A for non-streaming requests because the client receives one complete JSON response. Client concurrency is not GPU execution concurrency; the scheduler was configured for one running request.

Equivalent reproduction command reconstructed from recorded parameters:

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

## 3. Stress test

Scope: bounded service stability, not latency benchmarking. The run used 20 load requests, client concurrency 4, server GPU concurrency 1, two intentional streaming disconnects, a 200-request guard, a 32-client guard, and a 180-second request timeout.

| Check | Result |
| --- | --- |
| Load requests | 20/20 succeeded |
| Failed / rejected / queue timeout / inference timeout | 0 / 0 / 0 / 0 |
| Intentional client disconnects | 2/2 observed |
| Scheduler cancellation delta | 2 |
| Observed request throughput | 0.408 req/s |
| FIFO observation | Passed |
| Consecutive-request KV Cache isolation | Passed |
| Scheduler idle after cleanup | Passed |
| Final running / queued requests | 0 / 0 |
| Scheduler completed delta | 22: 20 load requests plus 2 deterministic isolation requests |
| Final accepted count represented by this run | 24: 22 completed plus 2 cancelled disconnects |
| Process-lifetime CUDA peak reported by status | 989.706 MB |

The raw artifact does not persist the stress prompt length, generation length, cleanup timeout, or seed. The command below uses the current script defaults for those fields (`64`, `64`, `30`, and `42`) and must not be read as proof of the unavailable historical values:

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

FIFO is an observational client-order check, not a formal proof under every operating-system scheduling interleaving. Cache isolation compares two deterministic consecutive responses; it does not inspect internal cache memory.

## 4. Docker smoke test

Scope: functional deployment acceptance only. The recorded acceptance states that the GPU image built successfully, `pip check` passed, `/health` succeeded, Chinese non-streaming inference succeeded, an SSE response produced 76 chunks and terminated with `[DONE]`, and Uvicorn shut down gracefully. No structured artifact records its date, prompts, elapsed time, token count, or memory consumption, so no performance values are attributed to this test.

Reproduce the functional smoke test with the commands in the repository [README](../../README.md#quick-start-with-docker-compose). The SSE chunk count is protocol activity, not a token count or throughput measurement.

## Metric definitions

- **Engine TTFT:** generator start to the first non-empty text delta.
- **Prefill:** first full-prompt model forward pass plus sampling.
- **Engine TPOT:** accumulated decode model-step time after prefill divided by decode steps.
- **Engine tokens/s:** generated token count divided by complete engine generation time.
- **HTTP TTFT:** request start to the first non-empty `delta.content`; role-only chunks do not count.
- **HTTP TPOT:** `(E2E - TTFT) / (completion_tokens - 1)` for successful streaming requests.
- **HTTP E2E:** request start to complete JSON or the SSE `[DONE]` marker.
- **HTTP tokens/s:** completion tokens divided by client E2E time, including queueing and transport.
- **Request throughput:** successful requests divided by the case wall-clock interval.
- **Queue wait:** server-provided `X-MiniLLM-Queue-Wait-Ms`, not inferred from E2E.
- **P50/P95/P99:** linear-interpolation percentiles over individual request samples.

An SSE chunk may contain zero, one, or multiple tokens. Chunk count is never substituted for token count.

## Limitations

- The engine cases have ten measured samples each; HTTP cases have only three. HTTP P95/P99 values are descriptive and not production-grade tail-latency estimates.
- Tests used one laptop GPU, one small model, one precision, and short prompt/output lengths.
- The raw reports do not embed a Git commit SHA. They can be schema-checked against the repository, but cannot be cryptographically tied to the current source revision.
- HTTP CUDA memory is a process-lifetime status value. Only the engine benchmark resets peak-memory statistics per measured run.
- GPU utilization, power, temperature, and clock telemetry were not collected and remain N/A.
- Model download and loading were excluded from steady-state latency.
- Re-run the tools on the checked-out revision before making deployment or capacity decisions.
