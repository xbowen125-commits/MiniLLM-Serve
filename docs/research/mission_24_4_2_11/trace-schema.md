# Trace Schema 草案

版本：`M24.4.2.11 / trace-schema / draft-0.1`。
**仅文档；没有 Schema 实现、验证器实现或新实测记录。**
规范含义以 [证据契约](evidence-contract.md) 为准，验收见 [矩阵](acceptance-matrix.md)。

## 1. 类型与空值规则

- `M`：key 必须存在；`C`：触发对应操作时必需；`O`：可选。
- `UID`：运行内非空字符串，身份作用域包含 run_id、process_uid；不能跨运行复用。
- `SHA256`：实际文件或规范内容的 64 位小写十六进制 SHA-256；不是“声明已通过”。
- `Obs<T>`：`{"state":"OBSERVED|UNKNOWN|NOT_APPLICABLE","value":T|null,"reason_code":string|null,"source_event_ids":[UID]}`。
  OBSERVED 必须有非 null 值和采集依据；0、false、[] 是合法观察值。
  UNKNOWN/NOT_APPLICABLE 的 value 必须为 null 且 reason_code 非空。
  NOT_APPLICABLE 只能用于契约确实不适用的字段，不能隐藏缺失的必需证据。
- ID 数组按真实操作顺序记录；集合关系另行标明。不得排序后丢失行顺序。
- 事件中的缺失 key 属 Schema 不完整；不可观测的值显式 UNKNOWN。
  字符串 `"unknown"`、null、空数组和实际不存在资源不是同一个含义。

## 2. 不可变运行身份清单（未来单独产物）

每个运行先冻结并封存一个清单，事件通过 `identity_ref` 引用其字节哈希；这是间接携带完整身份，
不是允许后补身份。清单丢失或哈希错误时，事件不得取得身份 PASS。

| 字段 | 必需性 / 可空性 | 含义 |
| --- | --- | --- |
| schema_version, contract_version | M / 非空 | 明确拒绝不兼容版本，不将 v1 日志当作新版本 |
| contract_set_sha256 | M / 非空 | 四份设计文件冻结后的排序路径/文件哈希清单摘要；不写回文档产生自引用 |
| run_id, process_uid, pid | M / 非空 | 每次进程出生独立身份；PID 不够 |
| git_head, source_tree_sha256 | M / 非空 | Git-visible 文件实际字节清单；未提交源码也纳入 |
| collector_source_sha256, checker_source_sha256 | M / 非空 | 采集与重算器分别绑定，不能用新重算器伪装旧采集源码 |
| config_file_sha256, config_canonical_sha256 | M / 非空 | 物理文件和语义规范哈希分列 |
| pristine_workload_sha256, prepared_workload_sha256 | M / 非空 | 原始输入/时间轨迹与 tokenizer 后元数据分列 |
| requested/resolved model_revision, tokenizer_revision | M / 非空 | 请求的 pin 与实际加载身份分列；加载核查证据有引用 |
| model_config_sha256, tokenizer_files_sha256 | M / 非空 | 模型语义与格式化依据，不能仅靠版本字符串 |
| runtime_environment | M / 非空 | Python/库/CUDA/GPU、设备实例；来自实际进程，非历史推测 |
| expected_dtype/device, observed_dtype/device | M / 非空 | 不一致不得静默回退 |
| backend_semantics | M / 非空 | full attention、无裁剪、mask/layout/axis、调用基数 profile、stream 完成协议 |
| frozen_bounds, evidence_budget, stop_policy | M / 非空 | 请求/时间/事件/字节边界，超界处理；不修改冻结业务配置 |
| evidence_kind, example_only | M / 非空 | 实测必须 correctness_trace / false；样例永远不能成为准入证据 |

## 3. JSONL 统一事件信封

| 字段 | 必需性 / 可空性 | 规则 |
| --- | --- | --- |
| schema_version, example_only | M / 非空 | 本草案版本，实测 example_only=false |
| run_id, process_uid, pid, thread_id | M / 非空 | 对应实际进程与线程，不跨 PID 补身份 |
| identity_ref | M / 非空 | 清单相对路径和 SHA-256，禁止绝对个人路径或跳出 run 根目录 |
| event_id, producer_uid | M / 非空 | 每个原始事件唯一身份；生产者是实际观察边界 |
| producer_seq, collector_seq | M / 正整数 | 各生产者顺序与串行写入顺序分开；缺口不可静默压缩 |
| monotonic_ns, clock_domain | M / 非空 | CPU 单调时钟域；不能推导另一进程/GPU 的全序 |
| kind, phase | M / phase 可 null | phase=null 仅限非执行事件；执行阶段不可从 input width 猜测 |
| iteration_id | M / 可 null | 非 iteration 事件为 null，需有原因；真实执行调用必须可关联到 iteration |
| span_id, parent_event_ids | M / span 可 null | 调用跨度和真实因果边，首个 run 事件才允许无父边 |
| observation | M / 非空 | 观测本身状态，不是操作成功状态 |
| payload | M / 非空对象 | 按 kind 的必需字段，见下表 |

`identity_ref` 与所有相对 sidecar 引用均包含物理哈希。离线索引再记录事件 ID、JSONL 行号、
完整文件哈希；运行中的事件不能包含自身所属文件的最终哈希而造成自引用。
重复 event_id/相同字节重放必须显式标为重复；原始行保留。
相同 ID 不同 payload 是冲突，禁止静默去重。独立重算不得信任派生 PASS 标志。

## 4. payload 事件族

本表描述未来采集义务，不声称现有 hook 已能提供所有字段。

| kind | 必需 payload / 条件字段 |
| --- | --- |
| identity.bind | 各层真实 ID、request_uid、原对象身份+出生代、Context/Future UID、绑定来源事件；尚未创建的层为 Obs UNKNOWN/NOT_APPLICABLE |
| admission.accept/reject, queue.enter/leave | admission_uid、原始请求身份、queue_uid、enter/leave 原因、sequence、原调用结果；拒绝与未观察到接受不同 |
| owner.command.enqueue/start/end | command_uid、Delivery/Future UID、原请求身份、enqueue attempt/result、实际取出、completion 结果和清理确认引用 |
| runtime.submit/step/result_apply.begin/end | operation_uid、实际成员、原始对象、直接前后状态、业务结果/原始异常；不能只记数量 |
| plan.created, worker.dispatch | plan_uid/phase_uid/iteration_id、decode/prefill 原始顺序、实际成员、容量读取引用 |
| runtime.forward.enter/exit | runtime_forward_id、invocation_uid、attempt_uid、父 phase、实际行映射、host return/error 和 device completion 分列 |
| model.forward.enter/exit | model_call_id、同一 invocation_uid、实际参数对象/输入/缓存版本、实际设备/stream、完成引用 |
| call.binding | runtime_forward_ids、model_call_ids、invocation_uid、relation、binding 真实来源、segment/fusion coverage；不是 UUID 文本相等 |
| decode.no_forward | 实际分支、pending_first、logits 来源 cache/model-call 版本、生成结果应用引用；无模型调用的原因 |
| mask.snapshot | mask_uid/version、实际参数绑定、binary 语义、[B,K]、每行 request_uid/row、P/S、实际位图/区间、过去/新增计数、position_ids 证据 |
| input.provenance | 实际输入行/序号、prompt/generated origin、Token 序号来源、private identity/hash 引用、input valid 位；不要求公共日志写 token/text |
| cache.snapshot | cache_uid/version、每层 K/V tensor_uid/version、layout/axes/shape/stride/offset/dtype/device、物理宽度、位置有效性 Obs 与来源 |
| transform.begin/end | transform_uid、merge/split/clone/pad/update 类型、实际来源/目标对象版本与行/切片、参数、逐层内容快照引用、完成证据 |
| cache.install | install_uid、request_uid、原引用前后、Slot epoch、实际 pool lookup 的 cache/version；与 transform 目标显式关联 |
| content.snapshot/check | snapshot_uid、原始来源/目标、canonicalization_version、私有 sidecar/offset/length/hash、覆盖范围、读稳定性和设备完成引用；布尔结果不能代替字节 |
| device.completion | completion_uid、对应调用集合、device/stream/event UID、等待/同步依据、success/error/unknown；非 Python 返回的别名 |
| request.terminal | 原 RequestState UID、生命周期、finish reason、生成/消费/写入计数分列、当时 Slot/ref/session/retirement 状态 |
| cancel.intent/effective | cancel_uid、实际发起源、原请求、有效安全边界、未完成调用；intent 不代表设备已停 |
| release.request/begin/end | retirement_uid、attempt_uid/attempt_no、retry_of、原请求与资源 epoch、before、操作结果、原异常/清理异常分别保存 |
| resource.observe | 当前权威 pool 查询、原对象 ref、session/retirement 存在性、在用调用、观测时点；absence/unobservable/null 区别 |
| resource.retired | 引用 Slot absence、ref cleared、pending-record absence、无在用执行的独立事件；不是释放函数返回时自动发出 |
| runtime.ack, owner.completion, consumer.ack | 三种确认分 kind；运行时遗忘/Future 完成/传输消费各自真实来源与依赖 |
| lease.release, worker.close, owner.stopped | Lease flag/权威 scheduler 归属、borrowed/owned、逐请求退役引用、Worker 方法结果和 Owner thread exit 分列 |
| observer.error, run.end | 丢失/异常/预算、是否完整、未完成 spans、最终资源明细、封口清单引用；无尾事件不得视作成功 |

### 实体和内容引用结构

`EntityRef = {entity_uid, entity_kind, version_uid, raw_object_identity:Obs<int>, birth_event_id}`。
tensor 还需 storage_uid/allocation_epoch、storage_offset 和可观察存储区间；data_ptr 单独不足。
`SlotRef = {pool_uid, slot_id, allocation_epoch, allocate_event_id}`。
`ContentRef = {snapshot_uid, tensor_ref, layer_index, kv_kind, row, slice, dtype, shape,
canonicalization_version, blob:{relative_path,byte_offset,nbytes,sha256}, completion_ref}`。
blob 路径仅在获准的私有 bundle 内解析；没有字节时明确 Obs UNKNOWN，不能填空 blob 假装内容相等。

## 5. 顺序与关联基数

基础因果图：

`plan → worker dispatch → runtime forward enter → model enter → model host exit →
device completion → stable output snapshot → split/transform complete → target snapshot → install lookup`。

出现异常时使用实际可观察的部分图，不伪造完整成功路径。cache 可原地改变，
同一个 cache_uid 的不同 version_uid 必须区分；每次快照声明是否因果一致。

Slot 释放前的最后一次 cache/mask/原 ref 观察必须在真实 release 开始前，且在相关设备操作完成后。
成功安装的目标不能到 worker_end 才首次观察，因为终态分支可能已清理。

释放链：
`terminal（可与资源存活共存）→ release request/attempt → pool/ref/record observations →
retirement confirmation → runtime ack → owner completion`；
SSE 还需 `consumer ack`，然后才是对应 Lease finalization。不同确认可并行，但依赖边必须明确。
borrowed Worker 的方法 close 不应被借用 Runtime 调用；Owner thread exit 不等于 Worker 的所有 GPU
资源已证实关闭。每条异常/重试保留原资源 epoch，不新造请求或重复生成。

1:多/多:1 的字段可以表达，但当前单模型普通 profile 不自动允许这两种关系。
意外基数、重复 binding、交叉请求/epoch 为 FAIL；缺少 binding/不清楚基数为 UNKNOWN。
重试必须有新的 attempt IDs；取消没有终止已提交 GPU 工作的语义。

## 6. 设计样例 JSONL（不是实测数据）

以下每行是一个独立的**设计样例**；不是完整可验收轨迹。
所有 `DESIGN-*` 标识、PID/时钟和全零哈希均为虚构占位；没有对应原始文件或设备事件。
`example_only=true` 必须使准入拒绝，不能将其与历史日志合并。
identity_ref 指向概念上的设计清单，**本任务没有创建该清单或 sidecar**。

```jsonl
{"schema_version":"m24.4.2.11-draft-0.1","example_only":true,"run_id":"DESIGN-run","process_uid":"DESIGN-process","pid":1,"thread_id":1,"identity_ref":{"path":"DESIGN-manifest.json","sha256":"0000000000000000000000000000000000000000000000000000000000000000"},"event_id":"DESIGN-binding","producer_uid":"DESIGN-model-observer","producer_seq":1,"collector_seq":1,"monotonic_ns":100,"clock_domain":"DESIGN-host","kind":"call.binding","phase":"decode","iteration_id":"DESIGN-it","span_id":"DESIGN-invocation","parent_event_ids":["DESIGN-runtime-enter","DESIGN-model-enter"],"observation":{"state":"OBSERVED","value":true,"reason_code":null,"source_event_ids":["DESIGN-runtime-enter","DESIGN-model-enter"]},"payload":{"runtime_forward_ids":["DESIGN-runtime-forward"],"model_call_ids":["DESIGN-model-call"],"invocation_uid":"DESIGN-invocation","attempt_uid":"DESIGN-attempt-1","relation":"one_to_one","binding_source":"actual_call_scope","members":[{"request_uid":"DESIGN-request","row":0}]}}
{"schema_version":"m24.4.2.11-draft-0.1","example_only":true,"run_id":"DESIGN-run","process_uid":"DESIGN-process","pid":1,"thread_id":1,"identity_ref":{"path":"DESIGN-manifest.json","sha256":"0000000000000000000000000000000000000000000000000000000000000000"},"event_id":"DESIGN-mask","producer_uid":"DESIGN-forward-observer","producer_seq":1,"collector_seq":2,"monotonic_ns":101,"clock_domain":"DESIGN-host","kind":"mask.snapshot","phase":"decode","iteration_id":"DESIGN-it","span_id":"DESIGN-invocation","parent_event_ids":["DESIGN-binding"],"observation":{"state":"OBSERVED","value":true,"reason_code":null,"source_event_ids":["DESIGN-binding"]},"payload":{"mask_uid":"DESIGN-mask-object","version_uid":"DESIGN-mask-v1","bound_model_call_id":"DESIGN-model-call","mask_semantics":"binary_key_padding","dtype":"int64","shape":[1,4],"rows":[{"request_uid":"DESIGN-request","row":0,"physical_past_width":3,"input_width":1,"one_ranges":[[1,4]],"allowed_past_count":2,"allowed_new_count":1,"position_ids":{"state":"UNKNOWN","value":null,"reason_code":"DESIGN_SAMPLE_OMITS_POSITION_BYTES","source_event_ids":[]},"verified_cache_positions":{"state":"UNKNOWN","value":null,"reason_code":"MASK_ALONE_DOES_NOT_PROVE_CACHE_VALIDITY","source_event_ids":[]}}]}}
```

这两个样例没有提供 device completion、Token 来源、完整层/内容、释放或最终排空，不能证明任何
实际 KV 正确性。格式可表达一项观察，不意味着对应强义务已满足。
