# Bounded Correctness Trace Evidence Contract

版本：`M24.4.2.11 / evidence-contract / draft-0.1`。日期：2026-10-25，Asia/Shanghai。

**仅设计；未实现、未测试、未采集新轨迹。正式 GPU Benchmark 继续禁止。**
本文件与 [Schema](trace-schema.md)、[验收矩阵](acceptance-matrix.md)、
[审阅记录](design-review-report.md) 共同构成一个待审批契约，不是准入证书。

## 1. 范围与不可变条件

下一条候选轨迹仅验证固定六请求的实际执行证据，不验证生产吞吐量。
沿用既有模型/tokenizer revision、float32、设备、请求正文、到达时间、采样参数、
Admission/batch 上限和 25ms 发送迟到门槛；不把观测开销从迟到记录中扣除。
当前冻结配置是 `benchmarks/runtime/online_http/configs/qwen_admission_v2.json`。
六请求原始到达偏移为 0、0、.25、.5、.75、1 秒，max_tokens 为
16、48、16、32、48、16；max_tokens 是上限，不是 stop 必须达到的数量。

以下边界不变：

1. 请求生命周期仍由现有 Runtime/RequestState 管理；观察器只有元数据身份注册表，
   不分配 Slot、不设置取消标志、不改变请求状态、不重试 Forward。
2. 终态、清理、消费、Lease 和 Worker 生命周期分开判定。
3. 不以时间邻近、数组在不同日志中的相同位置、对象地址整数或总计数建立身份链。
4. 历史日志不得补字段。新目录、新 run_id、新协议/采集器/源码身份；旧 FAIL/UNKNOWN 保留。
5. 此处没有实现或执行授权。任何后续接线、内容快照或运行都须单独获准。

### 有界采集提案，不是配置变更

| 限制 | 设计要求 |
| --- | --- |
| 正常轨迹 | 仅原六请求、最多两条实际 batch 行；不循环重放 |
| 既有时间保护 | 保留配置的 startup=180s、request=180s、run=600s、drain=60s |
| 观察器预算 | 提议最多 20,000 个事件、8 GiB 私有证据文件；运行前审批并冻结 |
| 内容覆盖 | 必需层、K/V、行、区间完整覆盖；预算不足不得抽样后宣称完整 PASS |
| 故障路径 | 本契约规定其证据，正常六请求未触发的路径为 NOT_OBSERVED；不得悄悄注入或混入 |

预算耗尽只使采集不完整并阻止证明；观察器不能据此强制归还 Slot。
未来控制端只能在另行授权下使用现有取消/排空机制。永久清理失败可能使服务无法终止，
不能靠强制关闭伪造有界排空；需保留 UNKNOWN 和人工处置要求。日志有界不等于清理必然终止。

## 2. 统一信封、身份与证据依据

每个事件携带 Schema/契约版本与哈希、run_id、进程实例 UUID/PID、线程、采集器身份、
源码树与配置/工作负载哈希、模型/tokenizer revision、dtype/device、时钟域、事件 ID、
生产者序号、采集序号、操作阶段、显式父/因果事件 ID、观测状态及 payload。
各字段的必需性见 [trace-schema.md](trace-schema.md)。

身份链保留：planned ID → client record ID → Lease ID → 原始 RequestState UID →
Context UID → Worker request ID → terminal result/Future UID；OpenAI response ID 单列。
相等只在直接绑定事件支持时成立。Slot 使用 `(pool_uid, slot_id, allocation_epoch)`；
cache/tensor 使用运行内对象 UID 和不可变版本，不能只用 `id()` 或 data_ptr。
相同内容哈希不表示对象相同，也不表示允许共享缓存。

每项断言至少引用独立原始事件及不可变状态版本，不引用最终 PASS 标志。
离线引用格式为：相对文件、完整文件 SHA-256、1-based 行号或 JSON pointer、事件 ID。
源码契约依据使用文件哈希和函数/规则 ID，不能把源码规定冒充执行观察。

采集序号表示日志提交顺序，不证明跨线程/进程/GPU 的真实 happens-before。
因果边必须来自真实调用栈、Owner 命令/Future、既有串行所有权或设备完成证据。
未知时钟域或缺失因果边保持 UNKNOWN，不按时间排序补齐。

## 3. 证据链 A：有效 KV 长度

适用配置：已核对的 full-attention、无 sliding/truncation/Prefix Cache、显式二值
key-padding mask；不满足条件时不得套用下列公式。

| 概念 | 操作性定义 | 必需证据与边界 |
| --- | --- | --- |
| 物理宽度 P | 输入缓存每层 K/V 的序列轴长度 | 直接记录 layer/axis/layout/shape/get_seq_length；层间不一致须报告 |
| mask 允许位置 A | 传入模型的二值 key-padding mask 中值为 1 的位置集合 | 实际位图/区间及计数，记录语义、dtype、是否实际传入；非二值/additive mask 不套 sum |
| 实际处理 Token T | 与该请求输入来源绑定、被成功完成的 Forward 消费的非 padding Token 序号集合 | 输入来源/序号、有效输入位、调用完成、缓存写入关联；不是 generated_tokens 长度 |
| 可验证缓存位置 V | 物理 cache version 中可关联到 T、已完成物化且有有效性来源的各层位置集合 | 每位置/区间来源、mask/position_ids、模型写入契约与完成/内容证据；不只计数 |

对一次调用，令新增输入宽度 S、实际 key mask 长度 K：必须观察 `K=P+S`。
把 mask 分成 `[0,P)` 与 `[P,P+S)`，分别得到允许的过去位置 A_past 与输入位置 A_new。
`|A|` 包含新增输入，不能当作 Forward 前的有效 KV 长度。
二值 mask 的 count_nonzero 可复算为 `|A|`，但相同计数不证明位置或请求归属正确。

对预先审阅的无裁剪追加模型，成功写入后才验证位置级关系：
`V_after = relocated(V_before) ∪ written_valid_inputs`，并逐层核对物理位置映射。
允许集合与物化有效集合可以在调用前后不同；只有对应时点、模型语义和写入来源均已确认，
才可验证 `V_before = A_past`。未完成调用、部分写入、unsupported layout 或未知模型语义为 UNKNOWN。
causal mask 对各 query 的可见位置另列，2D key-padding A 不能证明模型内部完整 attention 可见性。

关键特殊情况：

- Prefill 左 padding 的 mask=0 不等于对应 K/V 值必须为零。
- Merge 新插入的 padding 才按实际变换契约验证填充值；不能把二者混淆。
- pad_token_id 相等不是 padding 的判据；实际 mask 和输入来源才是。
- 首 Token Decode 从 Prefill logits 采样，可增加生成数而没有新增 KV 写入。
- EOS/stop/隐藏 Token 分别记录“生成/消费/写入”；最后生成的 Token 可能尚未进入缓存。

每个输入行须保留实际 mask 位图、position_ids、过去/新增分区、物理位置→逻辑 Token
来源映射以及每层写入完成引用。可用连续区间压缩，但不能只存数值总和。
缺失这些字段不得用 `prompt_length + generated_tokens` 回填成观察值。

## 4. 证据链 B：split 来源、变换、目标与内容

必须形成：
`forward output cache_version → source tensor/layer/KV/row/slice → transform_uid →
target tensor/cache_version/row → install_uid → Slot epoch → original RequestState UID`。

变换记录需要来源/目标的 dtype、设备、形状、stride、offset、切片轴/区间、padding 方向及
常量、clone/复制操作名、开始/完成和真实实体绑定。目标需要在 **split 完成、安装后、释放前**
分别观察；若终态在 Worker 返回前释放，不能用稍后的空 Slot 代替这些阶段。

严格区分三个判定：

1. **身份一致**：真实对象与版本、来源行、请求和 Slot epoch 均可追溯。
2. **结构一致**：形状/布局/层/行/区间满足变换声明；不证明值正确。
3. **内容一致**：按原操作定义，从稳定来源字节独立构造预期目标，与实际目标字节比较。

强内容证明提案：获准后保存有界私有 sidecar，覆盖所有必需层的 K/V 来源切片、实际目标及
安装后目标。规范表示采用原 dtype、声明的 byte order、逻辑 C-order，无 lossy cast，
保留 signed zero/NaN bit pattern；原 stride/layout 同时保留。变换 spec 单独哈希。
完整 sidecar 由 SHA-256 绑定，离线重算器重读字节并重做切片/pad/concat/clone 后逐字节比较。
若预期 clone 还需独立存储，则单独验证非别名/存储区间关系；内容相等不证明不别名。

不在公共日志写张量值、提示词或生成文本。sidecar 是潜在敏感激活数据，必须另行批准访问、
存储和保留政策；默认不公开、不进入 Git。不能获准保存足够字节时，只能提供受观察器信任
条件限制的 digest/比较结果，**独立可重算的强内容义务仍为 UNKNOWN**。
不能把一次 torch.equal 的布尔值或两个未经独立复算的 digest 当作完整证明。

快照必须在相关 GPU 工作完成且版本不会再变化的安全边界取得。后续快照应为不可变 CPU
证据副本，不长期持有 GPU cache 来影响退役。SHA-256 提供完整性/抗碰撞假设，不证明采集器诚实。

## 5. 证据链 C：Runtime 与模型调用身份

候选机制：在实际 Runtime Forward 入口创建 `invocation_uid` 并绑定已有 forward UUID；
通过进程内显式作用域栈传播至真正 model.forward 入口，模型观察器独立创建 model_call_id，
记录共同 invocation_uid、直接父调用和 binding event。退出用 finally 恢复作用域。
跨线程/异步边界必须显式传递命令上下文；缺失传播保持 UNKNOWN。

| 基数/路径 | 契约 |
| --- | --- |
| 1:1 | 当前普通后端预期；同一次成功 Runtime Forward 对应一个顶层 model.forward，成员/行/版本一致 |
| 1:多 | 必须在新运行前获准并冻结 segment 规则与输入/行覆盖；当前 profile 未允许时为基数违例，不能自动解释为分片 |
| 多:1 | 必须有真实 fusion/binding 操作和每行父调用覆盖；当前 profile 不支持，不根据 batch 大小推断 |
| 子模块调用 | 与顶层 model.forward 分 kind/namespace，不算第二次顶层调用 |
| 重试 | 新 attempt_uid 与新调用 IDs，显式 retry_of；当前生成路径不授权 Forward 重试，清理重试不能伪装为 Forward 重试 |
| 首 Token 无 Forward | 记录 decode branch/pending_first 和 logits 来源；model_call_id 为 null，理由是 NOT_APPLICABLE，不制造空调用 |
| 入场前取消 | 记录 intent 与 safe-boundary effective；未调用模型须有取消前入口证据，不仅是日志中没有调用 |
| 入场后取消/异常 | host return/raise 与 device complete/error 分开；未确认完成不得允许缓存释放 |

关联不靠 UUID 文本巧合。相同 invocation_uid 必须有真实绑定事件；重复或冲突的 binding
不得去重。CUDA 完成记录需 device/stream/event 或等价同步依据，Python 返回不等于设备完成。
内部多 stream 无法审阅时，完成性保持 UNKNOWN；不能用任意默认 stream 的 event 冒充所有工作完成。

## 6. 证据链 D：终态与释放

分别记录请求终态、release intent、attempt result、实际资源查询、引用清除、Runtime acknowledge、
Owner Future 完成、HTTP consumer 完成、Lease release、Worker close、Owner thread exit。
当前源码中 Runtime acknowledge 在 Owner Future 完成之前；SSE finalizer 又等待 consumer_done。
它们不能统称一个 acknowledge。

释放前证据：原始 RequestState UID、生命周期/phase、Slot epoch/cache version、session/retirement
身份、原始 kv_cache_ref UID、相关设备调用完成证明、没有正在使用该版本的执行。
释放后证据：该 request/epoch 的权威 pool 查询、原始对象的实际引用读取、retirement/session
状态、未完成使用者、attempt 的原始异常及清理异常分别保存。Session 消失不是完成条件。

完整退役需要：Slot/缓存归属已明确撤销、原始引用已清除、必要清理确认完备、无待确认 retirement，
且没有执行仍使用它。不要求模型权重或 PyTorch allocator reserved memory 归零。

- 抛错但 Slot 已不存在：仍按两条证据记录，不能吞掉异常；可在真实重试确认后恢复为退役。
- 返回成功但 Slot/ref 仍存在：FAIL；Slot 不存在但引用/完成证据缺失：UNKNOWN。
- 清理失败：保留同一原始请求及 retry target；不重复生成、Prefill 或 Decode。
- 重复清理：不同 attempt IDs；相同 attempt/event 重放只容许完全相同的幂等记录。
- Slot 复用：必须在上一 epoch 已确认退役后真实分配新 epoch，不能只观察总数为零。
- Worker 结束与请求退役不同：borrower 不能关闭共享 Worker；owned Worker 关闭也不能代替逐请求确认。

## 7. 判定、可信边界与停止

PASS：限于指定义务/运行/范围，原始证据完整且身份、时点、值关系均一致。
FAIL：有直接证据违反已声明规则；模型不支持某转移时先区分模型缺口与实现反例。
UNKNOWN：缺失、不可观测、身份不符、未定义/未触发、模型语义或可信依赖未确认。
NOT_VERIFIED 专用于本次无法读取/核查的资产；不是未来运行的 PASS。

观察器必须显式启用且为 correctness-only。捕获错误不能改变业务返回、取消、缓存或调度。
记录写入丢失、序号洞、异步快照失败、预算耗尽、异常脱钩、未封口文件均使相关义务 UNKNOWN，
并停止证明/准入；出现明确串用、过早释放等违例则 FAIL。若错误日志本身无法写入，末尾缺失或
独立控制端的缺失心跳/封口也必须使运行不可认证，不能把沉默当成功。

采集器本身、快照规范化、内容变换重算器、模型写入语义与设备完成协议属于可信基。
完整性哈希不证明这套可信基正确；需要后续独立审阅和反例验证，本任务未执行。
观察器会扰动时间和内存，记录原始影响，不形成性能结论。

局部方程、边界投影和内部微步关系分开报告；没有前后状态就不能一律标 stutter。
即使未来所有观察义务 PASS，也不能外推所有调度交错、双模拟、状态空间穷尽或无条件终止。
任何现存 GPU Benchmark 禁令及 PREFLIGHT 阻塞都不会因本设计完成而解除。
