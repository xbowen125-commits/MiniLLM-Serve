# Mission 24.4.2.11 — Design Review Report

版本：`M24.4.2.11 / design-review-report / draft-0.1`。
日期：2026-10-25，Asia/Shanghai。项目根：`D:\AI_Lab\MiniLLM-Serve`。

## 1. 本轮性质与实际动作

仅只读审计、设计和新增四份 Markdown：

- [evidence-contract.md](evidence-contract.md)
- [trace-schema.md](trace-schema.md)
- [acceptance-matrix.md](acceptance-matrix.md)
- 本文件。

**测试执行数 0；GPU 推理 0；模型加载/服务启动停止/Benchmark 0；生产源码修改 0。**
未运行 pytest、compileall、重算器或其他测试；未修改测试、配置、已有文档或实验文件。
只使用文件读取、JSON 内容查看、SHA-256 与只读 Git 文件枚举/HEAD 读取。
未执行 Git stage/commit/push/reset/checkout/stash，也未调用 git status 刷新索引。
新增文档是明确授权的工作区新文件，不改变既有文件、索引、分支或引用。

所有必要资产此次均可读取，未使用提权、权限修改或访问绕过。
下文的历史 PASS/数量是**读取到的既有报告值**；本轮未重新执行其验证器、
逐条验证全部事件，或重做历史报告声称的全量完整性核验。

## 2. 源码身份与算法

只读确认 HEAD：`dc9db6e9cb2585bd7eb141717206586ec0e01fb3`。
新增文档之前，284 个 Git-visible 现有文件的字节清单重新计算得到：

`7e1e187f79610b71fddfcf74ff922e3835df667289f2205f337693f1d91044d8`。

该值与实际读取的 Mission 24.4.2.10 `source-final.json` 中的记录相同。
既有清单记录 dirty=true；没有通过 git status 重新更新该字段。

算法依据 E13：对 tracked 与 non-ignored untracked、实际存在的文件逐个 SHA-256，
将 `相对路径 → 小写文件哈希` 按键序生成无空格规范 JSON，再 SHA-256。
本轮使用原生只读字节/哈希运算重现，不导入/执行项目的 git_snapshot 函数。
本次路径为 ASCII，使用 ordinal 排序；若存在编码/转义差异，不得宣称算法等价。
未遍历被忽略的模型/环境目录。文件字节不做换行归一化。

上述是 **文档写入前** 的来源身份，不是四份新文档加入后的完整树身份。
写入新文档必然改变包含未提交文档的全树摘要；交付时文档自身 SHA-256 单列，
不把摘要写回自身形成循环引用。新采集器/新轨迹未来仍须绑定它们实际使用的最终源码。

## 3. 实际读取的资产清单

下表路径相对于项目根；SHA-256 为本轮实际文件字节读取的哈希。
为便于辨认，路径前缀缩写严格展开为：

- `B/` = `benchmarks/runtime/online_http/`
- `H/` = `benchmarks/runtime/results/online_http/`
- `S/` = `src/minillm/`
- `R10/` = `H/refinement2444210-20261025-a/`
- `T9/` = `H/identity244429-20261025-gpu-b/continuous/`

| ID | 实际路径 | 读取范围/版本依据 | SHA-256 |
| --- | --- | --- | --- |
| E01 | B/kv_evolution.py | 全文；输出 version=1，四类缺口单列 | `9c8b106986d0ffa2d36b9d9c531edb5ca6644ac70877dc07e937ec3b6c5a7d3b` |
| E02 | B/trace_refinement.py | 全文；按实际函数和文件哈希绑定，不假定另有独立版本号 | `9515630b55a4641dc56387c48a3f741aaaf111540ed30020f59a44b2a98e6a65` |
| E03 | tests/test_trace_kv_refinement.py | 定义阅读，未执行；synthetic 反例与 UNKNOWN 断言 | `29c6e2446a6d0540745fc8f4b793d85dbb6013aebffca7429b755df630966c8d` |
| E04 | docs/trace_kv_refinement.md | 全文；Mission 24.4.2.10 定位和证明边界 | `55b787e9d054382a790bab225e66358b3ed1aee3d3e1bdc688132153f3943f01` |
| E05 | docs/abstract_transition_relation.md | 开头至规则表等必要片段（前 88 行）；Mission 24.4.2.8 | `e38901c2bde27b939fe60282a91991228269cdfd630f4453cf16223b799fbd45` |
| E06 | B/runtime_trace.py | 信封、cache/runtime view 及 Worker/Forward/merge/release/close 片段；VERSION=1 | `2edb313388c6f03e9ed5670d79e705aad57800a48da53e2dd86c73c0d574710d` |
| E07 | B/preflight_server.py | 前 155 行，模型观察器和调用边界 | `00927285cb2847aa25a15aa82e9e3c77d5595749484506bee61e628e56d6b0a3` |
| E08 | S/workers/transformers_batch_worker.py | 全文，split/merge/forward、首 Token、retirement 与释放 | `c3693650f6799704080ade9fb4e4c29c6fd1b282b9e9cb427d9289b3d955804f` |
| E09 | S/workers/batched_transformers_worker.py | 全文，设备转换与 release 前 synchronize | `da0c86f64734bc15423202b402342b52b6234cb78b8cb0cb6a2bb6c728ac475e` |
| E10 | S/memory/slot_kv_pool.py | 全文，Slot/cache 引用与复用 | `58fa8343dffa9d145de7e308edeb55fcc80d36e7b04775060dcb16e2ce300c36` |
| E11 | S/runtime/http_owner.py | terminal、Runtime acknowledge、Future 与关闭片段（221–320 行范围） | `3cbb1585b4e152be7d24af5c409f6d2f8eb0793d29b115c0b89bcd36f16da466` |
| E12 | S/server.py | ContinuousStreamSession/finalize 片段（241–295 行范围） | `280c17c7534c048758535985841c763f13f41588b8a4ccb9fd271e64fb1f8979` |
| E13 | B/artifacts.py | 前 68 行，指纹/原始写入契约；未执行函数 | `4de77c9151223122fa502111c4f092504119f9b958a5c4778c257dc7409f7385` |
| E14 | B/retirement.py | 前 46 行，VERSION=2、JointState/ResourceStatus | `fe50271199c32a8d20a449f32134e951f35255511b1d61f5543cf8aed0f6a312` |
| E15 | H/transition244428-a/historical-gpu-confirmed/report.md | 全文，Mission 24.4.2.8 报告；PARTIALLY CLOSED | `40087ab280e72ee07581e37984857586da3b3daa41d7caa1f83861c58d0f29da` |
| E16 | R10/mission-report.md | 全文，历史数量/测试结果只作为报告陈述 | `347f99df60573cf6ab7bcf88d6e5b2806498ec23c392737cf070f0e4699cb494` |
| E17 | R10/verified-a/audit.json | 全部摘要字段；无独立 schema_version，不自造版本 | `097dc50d7c6ca5726997ded0db498d3cec378138f8adb600ce753ce4b531eb37` |
| E18 | R10/verified-a/kv.json | version=1、计数/UNKNOWN 与第一条 Forward/引用清单；未执行重算 | `8b8a208bbb6210eaec02d711c3f090bb33e47cb8c610e5607bab54076f3025c8` |
| E19 | R10/verified-a/identity.json | version=1、六请求摘要与首条七层链；未重新判全链 | `454b660e815e57c779de33a4069af0e90728abea53293ad56e025b773d290cac` |
| E20 | R10/combined-admission.json | 全部状态/阻塞；status=FAIL、admitted=false | `3d3e4ce22d0f23ac6076d71b4fcaf578ac9982308fb5ebe12b6616c449f95b6c` |
| E21 | T9/runtime-trace.jsonl | 读取筛选实际 seq 74、79、100、319 等样例字段；schema_version=1 | `5821c63ab17864a5fdc33734c9e233db876f4ce532674aea713178389d07fc8f` |
| E22 | T9/observer.jsonl | 首个 forward_attempt 实际字段；该样例未声明统一新版本 | `1aa4547faff09f2af3ed87d732595504409e7dc8000f67b7a78e2b7524f4b54b` |
| E23 | R10/source-final.json | 已存 HEAD/dirty/source/files 清单，与本轮写前字节指纹比较 | `25427b0c8780ae5eebaab5fca5f93530514f8ffbce5084fed354a1ef2bf610ed` |
| E24 | B/configs/qwen_admission_v2.json | 配置标量、请求 ID/到达/max_tokens；不复述正文 | `5bf9d4b1134a0f9bb7d181b4d8efe9366427b9a640db8c624653313a9f33f4c0` |
| E25 | S/continuous/inputs.py | 全文，按长度而非 pad ID 生成 mask；Decode mask 仅新 Token | `ef2b5dc58bc37aabe6be7730542ae1ef04421d30566e8964a7fb2111bebaa918` |

没有不可访问的必要资产；因此本轮资产访问项没有被填为猜测内容。
未读取/未重算的其他报告、完整 Runtime 所有路径、原始 tensor bytes、其他设备状态仍是
NOT_VERIFIED；不能用本表某个文件的哈希冒充对其所有历史结论的重验证。

## 4. 直接证据与设计决定

### A. 有效长度不是物理宽度

E01 明确把 `DECODE_EFFECTIVE_KV_LENGTH` 保留 UNKNOWN，并排除其参与物理方程摘要。
E25 的 Decode builder 返回的 mask 只覆盖新增输入；E08 再拼上 session mask。
因此新契约必须观察**实际传给模型的完整 mask**，不能只采集 builder 的新 Token mask。
E21 seq 74 的物理输入是 `[2,72]`；E18 将它关联到不同 prompt 长度，不能据此认定两行
有效 Token 都是 72。新的四概念及条件公式见契约 §3。

### B. 身份、结构、内容不等价

E08 的实际 split 是逐层、逐行 clone 后 target.update；E10 再保存引用。
E18 即使物理方程报告 PASS，`SPLIT_ROW_VALUE_LINEAGE` 仍 UNKNOWN。
本设计因此要求来源/目标快照和安装后查询三段关联；强内容重算依赖获准的原始字节，
不是只比较形状或接受观察器的单个 true。

### C. 两种调用 ID 不天然相等

E18 中 Runtime Forward ID 为 `931ce1e1fe0d47f491bba5939f039ac3`；
E21 seq 74 与 79 直接使用该 ID。E22 模型观察器样例另有 call_id
`38f3634dcca54ed583e96750c607f693`，没有共同 invocation binding。
**本轮没有依据相邻日志将两者认定为同一调用**；这里只核对命名空间和缺失的连接。
E07 还把 phase 从 input width 推断，并注明单 Token prompt 歧义；新 Schema 要求实际阶段来源。

### D. 终态、资源、消费和结束分别观察

E08 允许在 Worker decode 返回前调用 release；E18 的后置行观察缺口不能用空 Slot 填补。
E21 seq 319 对请求 `1` 有直接 `request_cache_ref=false` 与 release_id，且对端 Slot `2` 仍存；
这不是“所有资源都为零”的证据，更不是 Worker 已结束。
E08/E09 说明设备同步、pool absence 和原引用清除有不同步骤与异常路径；E11/E12 又说明
Runtime acknowledge、Owner Future、SSE consumer_done、Lease release 有不同依赖。
新契约不改变这些行为，仅规定需要独立记录的依据。

### 历史结论仍然有边界

E17 报告 102 次物理方程 PASS、完整 Forward 102 UNKNOWN、111 个边界投影 PASS、
全事件 1,743 局部 PASS 与 1,632 个非独立 v2 边 UNKNOWN。这些是读取到的历史摘要，
不是本轮执行结果。E15 也只声称局部事件映射，不证明冻结全路径。
E20 中四项历史 PREFLIGHT FAIL 与全局 UNKNOWN 继续保留，不因本设计重写或解除。

## 5. 设计门禁审阅结果

| 设计标准 | 结果 | 限定范围 |
| --- | --- | --- |
| 四种长度/有效性操作性定义 | PASS（要求已写明） | 契约 §3；实际 mask/位置新证据未采集 |
| 身份/结构/内容独立判定 | PASS（要求已写明） | 私有字节和可信基尚未获准/验证 |
| 调用基数、取消、异常与重试 | PASS（要求已写明） | 不支持的基数不自动合法 |
| 终态、释放、消费者与结束分离 | PASS（要求已写明） | 未实现新 hook 或退出协议 |
| 每项断言有来源、空值与 UNKNOWN 规则 | PASS（要求已写明） | Schema 是草案，未运行验证器 |
| 历史隔离与禁令保留 | PASS（本轮遵守） | 仅四份新设计文档；不进入后续阶段 |
| 新实现/新运行满足契约 | NOT_VERIFIED | 本轮没有实现或执行授权 |

这是人工设计审阅，不是测试 PASS 或可提交给 gate.py 的机器认证结果。

## 6. 尚未解决的问题

1. 模型内部 causal 可见性、写入位置语义与多 stream 完成协议需后续审阅；外部 mask 本身不能证明。
2. 私有 tensor/Token 字节保存是否获准、如何限制访问和保留、独立重算可信基仍 NOT_VERIFIED。
3. 8 GiB/20k 事件提案未经成本测量或批准；超界会令证据不完整，而不是允许丢样本。
4. 实际 split 点与释放前的安全取样位置、对象 UID 生命周期、异步快照冻结均尚未接线或验证。
5. 1:多/多:1、Forward 重试与子模块观察的实际 profile 扩展尚未授权，不自动进入当前验收。
6. 一条正常六请求轨迹不能覆盖全部取消/失败/永久清理失败/共享 Worker 场景；需独立设计与授权。
7. 内部微步、联合资源模型与 v2 的完整对应、所有调度交错及整个状态空间穷尽仍 UNKNOWN。
8. 即使未来新轨迹通过有限范围检查，也不证明无条件活性/终止性，且不能替代完整 PREFLIGHT。

## 7. 停止与交付声明

四份文档为本任务的全部新文件。文档 SHA-256 在写入完成后只读计算并在交付消息列出；
不回写自身哈希、不新建运行目录、不重封历史产物。

**测试执行数为 0，GPU 推理为 0，生产源码修改为 0。**
历史测试数量不是本轮测试。正式 GPU Benchmark 禁令和所有既有 PREFLIGHT 阻塞明确保留。
设计交付后停止，不自动进入实现、接线、测试、服务操作或 GPU Benchmark。
