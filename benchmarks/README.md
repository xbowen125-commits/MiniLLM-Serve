# MiniLLM-Serve Benchmarks

已公开的、经过隐私整理的历史实测证据见：

- [Benchmark Summary](reports/benchmark-summary.md)
- [Environment and Reproduction Notes](reports/environment.md)

公开数据来自 2026-10-09 至 2026-10-10 的本地原始报告。它们是历史快照，不代表生产容量，也不能替代在当前代码版本上的重新测试。原始报告未保存 Git commit SHA，因此无法与当前源码建立提交级证明。

Mission 08 将测试分为两层，禁止把 Mock 或客户端指标写成模型引擎性能：

- `benchmark_kv_cache.py`：直接调用模型引擎，测量 Prefill、Decode 和 CUDA 显存。
- `benchmark_http.py`：通过真实 HTTP/SSE 请求测量用户侧延迟、并发吞吐与排队时间。
- `stress_http.py`：有界压力和稳定性验证，记录拒绝、超时、取消、清理与 KV Cache 隔离。

所有工具默认将报告写入 `benchmarks/results/`，生成同名前缀的 JSON、CSV 和 Markdown 文件。该目录中的结果是运行产物并继续由 Git 忽略；公开仓库仅提交 `benchmarks/reports/` 下经过隐私审核的摘要。不应把 Mock 数据或未复现的历史结果提交为通用结论。

## 指标口径

### 引擎侧

- **TTFT**：调用生成器至首个非空文本增量的时间。
- **Prefill**：完整 Prompt 第一次模型前向及采样耗时。
- **TPOT**：Prefill 后 Decode 模型步耗时总和除以 Decode 步数。
- **Tokens/s**：实际 completion token 数除以完整引擎生成时间。
- **CUDA peak**：每次正式运行重置 PyTorch 峰值统计后得到的峰值；模型加载时间不计入。

### HTTP 客户端侧

- **TTFT**：开始 HTTP 请求到首个非空 `delta.content`；role-only chunk 不算。
- **TPOT**：`(E2E - TTFT) / (completion_tokens - 1)`，仅适用于成功的 Streaming 请求。
- **E2E**：客户端发起请求至完整 JSON 或 `[DONE]`。
- **Tokens/s**：实际 completion token 数除以客户端 E2E 时间，因此包含排队和网络开销。
- **Request Throughput**：一组并发请求中成功请求数除以该组墙钟时间。
- **P50/P95/P99**：对单请求样本使用线性插值分位数。
- **Queue Waiting Time**：来自响应头 `X-MiniLLM-Queue-Wait-Ms`，不是用 E2E 反推。

非流式请求从服务端 `usage.completion_tokens` 读取 Token 数。Streaming 会拼接全部文本后用与模型匹配的 tokenizer 计数；如果本地 tokenizer 不可用，则 Streaming Token、TPOT 和 Tokens/s 明确记录为 N/A。SSE chunk 可能包含零个、一个或多个 Token，工具从不使用 chunk 数作为 Token 数。

## 1. 引擎与 KV Cache Benchmark

默认不会下载模型；加 `--offline` 可强制 Hugging Face 离线模式。正式数据建议至少 `--repeats 10`：

```powershell
# 先进入克隆后的 MiniLLM-Serve 项目根目录。

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe benchmarks\benchmark_kv_cache.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --device cuda `
  --dtype float16 `
  --prompt-lengths 32,256 `
  --generation-lengths 16,64 `
  --warmup 2 `
  --repeats 10 `
  --seed 42 `
  --offline `
  --output-prefix benchmarks\results\engine-kv-formal
```

模型只加载一次；预热不会进入正式统计。KV Cache 开关使用相同模型、设备、精度、Prompt 和确定性生成参数。

## 2. HTTP Benchmark

先在一个 PowerShell 窗口启动服务：

```powershell
$env:MINILLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
$env:MINILLM_DEVICE = "cuda"
$env:MINILLM_DTYPE = "float16"
$env:MINILLM_USE_KV_CACHE = "true"
$env:MINILLM_MAX_CONCURRENT_GENERATIONS = "1"
$env:MINILLM_REQUEST_QUEUE_CAPACITY = "16"
$env:MINILLM_QUEUE_TIMEOUT_SECONDS = "120"
$env:MINILLM_INFERENCE_TIMEOUT_SECONDS = "180"
.\.venv\Scripts\python.exe -m uvicorn minillm.server:app --host 127.0.0.1 --port 8000
```

在另一个窗口执行：

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_http.py `
  --base-url http://127.0.0.1:8000 `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --prompt-lengths 32,256 `
  --generation-lengths 16,64 `
  --concurrency 1,2,4,8 `
  --mode both `
  --kv-cache-state on `
  --warmup 2 `
  --repeats 10 `
  --seed 42 `
  --max-total-requests 500 `
  --output-prefix benchmarks\results\http-kv-on
```

比较 KV Cache 时，停止服务、将 `MINILLM_USE_KV_CACHE` 改为 `false`、重新启动，然后用相同参数运行并把输出前缀改为 `http-kv-off`。工具不会在基准过程中修改服务配置。

`--allow-model-download` 是显式选择；默认 tokenizer 只从本地缓存加载。客户端并发数与 `MINILLM_MAX_CONCURRENT_GENERATIONS` 是不同概念，报告会同时记录两者。

## 3. 有界稳定性测试

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
  --output-prefix benchmarks\results\stress-bounded
```

安全上限固定为最多 200 个普通请求、客户端并发最多 32、主动断开请求最多 20。脚本结束前轮询调度状态；若仍有运行或排队请求，会以非零状态退出。

报告分别保存主负载后、主动断开后、KV Cache 隔离检查后和 cleanup timeout 后的调度器快照。`client_disconnect_attempts` 是计划尝试数；只有收到有效内容后确实关闭连接的请求才计入 `client_disconnects`。客户端拒绝/超时统计覆盖所有阶段，因此可与 scheduler delta 对账。

要观察不同保护路径，应每次只改变一组服务配置并重启：

- 队列满：将 queue capacity 设为 1，并让客户端并发显著大于 GPU 并发。
- 排队超时：使用较短 queue timeout 和足够长的生成任务。
- 推理超时：使用较短 inference timeout；CUDA kernel 仍需完成当前前向才会释放名额。
- 主动断开：`--disconnect-requests` 会在收到首个有效内容后关闭 SSE 连接。
- 服务关闭：压力测试运行时在服务窗口发送 `Ctrl+C`，观察关闭日志，并在重启后检查上一次报告；工具不会自动杀死未知服务进程。

生产 API 没有故障注入端点，因此模型前向异常恢复使用 Mock 自动化测试验证，不会为了压测而在生产接口中增加隐藏故障开关。

## 报告解读与限制

- 公开摘要中的 Engine、HTTP、Stress 和 Docker Smoke 是四种不同证据，不能互换延迟、吞吐量或显存指标。
- 已公开的 HTTP 每个 case 只有 3 个正式样本；其中 P95/P99 仅用于描述该次小样本运行，不是生产级尾延迟结论。
- Docker Smoke 只有功能验收记录，没有结构化性能报告；76 个 SSE chunk 不等于 76 个 Token。
- 最终成功的压力测试证据来自 `stress-disconnect-fixed.json`；较早的 `stress-bounded.json` 是修复前诊断记录，不作为成功结果。

- JSON 保存元数据、汇总、调度器前后状态和原始请求样本。
- CSV 保存便于电子表格分析的汇总行或请求行。
- Markdown 是自动生成的可读摘要。
- 服务 `/health` 提供模型、device、dtype、KV Cache 和核心库版本。
- `/scheduler/status` 提供队列累计指标和可获得的 PyTorch CUDA 当前/峰值显存。
- HTTP 暴露的 CUDA peak 是进程生命周期峰值，不等于独立测试峰值；严格显存对比应使用引擎 Benchmark。
- P95/P99 在少量样本下不稳定；正式测试建议每组至少 10 次，更严肃的尾延迟分析应使用更多样本。
- 笔记本 GPU 会受温度、功耗墙、后台进程和动态频率影响。单机单次结果不能推广到其他硬件。
- 当前没有 GPU 利用率、功耗和温度采样依赖；这些字段为 N/A，而不是估算值。
