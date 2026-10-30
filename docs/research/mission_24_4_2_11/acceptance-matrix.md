# 验收矩阵（仅设计）

版本：`M24.4.2.11 / acceptance-matrix / draft-0.1`。
**本次没有执行下列运行验收；未来运行证据状态均未获认证。**
PASS / FAIL / UNKNOWN 按 [契约](evidence-contract.md) 逐义务、逐范围判定。
历史报告的 PASS 不迁移到新 Schema；NOT_OBSERVED/NOT_VERIFIED 不能被当作 PASS。

## 1. 通用与身份门禁

| ID / 不变量 | 独立证据与新字段 | PASS 条件 | FAIL 条件 | UNKNOWN / 现有缺口 | 停止条件 |
| --- | --- | --- | --- | --- | --- |
| G01 新旧隔离 | 新 run/process IDs、契约/源码/配置/工作负载原始清单及哈希 | 新目录独占、实际身份一致、原始文件可重读 | 覆盖旧文件、直接哈希冲突、sample 被当实测 | 旧来源与新源码不匹配、清单缺失 | 不认证新轨迹，保留旧禁令 |
| G02 身份链 | 各层绑定原始事件、EntityRef、Slot epoch、完成 Future UID | 每条边直接、唯一或符合已声明基数 | 冲突/串用/同一 epoch 多个活跃所有者 | 缺一层、只有 id() 整数、只靠时间/计数 | 禁止合并/自动消解记录 |
| G03 文件与事件完整性 | producer/collector seq、原始行、sealed hashes、run.end | 文件和期望跨度齐全，无未解释丢失 | 文件实际被修改、相同 ID 不同 payload | 未封口、缺事件、未记录写入失败 | 证明停止，不把缺日志解释为无操作 |
| G04 有界与冻结 | frozen_bounds、预算/异常记录、原计划及实际发送时刻 | 未改六请求/参数/时间，迟到≤25ms，全部记录保留 | 修改轨迹/阈值、迟到超限 | 时刻/预算证据缺失；8 GiB/20k 预算尚未实测校准 | 运行证据不可准入，不强制释放资源 |
| G05 观察器非干预 | 实际挂接范围/源码身份、捕获失败记录、业务返回/异常/原对象独立记录 | 不写生命周期/Slot，不改变业务结果 | 观察器直接改变状态、吞掉业务异常或伪造事件 | 未验证非干预、采集器可信基未审阅 | 停止证明；后续实现需单独审阅 |
| G06 隐私与内容授权 | sidecar 许可、私有目录、访问/保留政策、脱敏说明 | 最小字段、无凭据、敏感字节仅在获准私有 bundle | 密钥/个人路径或未授权激活被公开 | 无内容保存许可或无法获得足够原始字节 | 不采集敏感内容；强内容义务 UNKNOWN |

## 2. 链 A：长度与有效位置

| ID / 不变量 | 独立证据与新字段 | PASS 条件 | FAIL 条件 | UNKNOWN / 现有缺口 | 停止条件 |
| --- | --- | --- | --- | --- | --- |
| A01 物理宽度 | cache version、各层 K/V layout/shape/axis、实际 get_seq_length | 观察值符合该模型布局和输入/输出方程 | 已知维度矛盾、非法宽度/层 | 缺轴/层/版本；旧物理检查不证明有效长度 | 停止该调用证明，保留反例 |
| A02 真实 mask | 实际模型参数绑定、mask 位图/区间、P/S、binary 语义、position_ids | A_past/A_new 可从原始位图独立重算；不是 pad_token 判定 | 计数与位不符、参数绑定错误、冻结 profile 明确违例 | 旧 Decode 无非零值；内部 causal 可见性未观察 | 不用物理宽度或推算值补齐 |
| A03 实际消费 Token | 实际 input 来源/序号/有效位、原生成/Tokenizer 证据、调用完成引用 | 非 padding 消费来源明确且不重复计入 | 串用 Token、重复消费/写入违背已声明规则 | 只有 generated_tokens、文本重新分词或输入哈希无来源 | 该请求 Token/KV 演化 UNKNOWN |
| A04 可验证缓存位置 | V 映射、A_past、写入/迁移 provenance、逐层版本、模型写入语义依据 | 在相应时点与已完成写入规则一致，位置与来源均匹配 | 已知有效位置丢失/多写/串行或跨请求串用 | mask=1 但无写入证据；unsupported sliding/additive mask | 不认证有效缓存集合 |
| A05 Prefill/首 Token/结束 | pending_first、logits 来源、Forward/生成/写入分开计数、EOS/stop 原因 | 首 Token 无冗余 Forward；最后生成未消费时不虚增 KV | 用“生成+1”冒充“缓存写入+1”、错误重复 Prefill | 旧首 Token分支部分可见，但完整 logits/位置来源尚不足 | 不改 max_tokens 或提前 stop 语义来通过 |

## 3. 链 B：split 与内容

| ID / 不变量 | 独立证据与新字段 | PASS 条件 | FAIL 条件 | UNKNOWN / 现有缺口 | 停止条件 |
| --- | --- | --- | --- | --- | --- |
| B01 来源身份 | source cache/tensor version、layer/KV/row/slice、transform_uid、实际输入成员 | 每个目标有唯一、实际观察的来源路径 | 来源/请求/行明确冲突 | 旧 split 行值来源缺失，不能只枚举输出顺序 | 不将 Shape/Slot 对齐提升为内容 PASS |
| B02 结构变换 | source/target layouts、操作与 pad/clone/slice spec、完成引用 | 结构和所有层覆盖符合该操作 | shape、padding、范围或层不符 | 少层、目标释放前未观察、变换分支不明 | 缺失保持 UNKNOWN，禁止采样冒充全量 |
| B03 内容相等 | 稳定来源切片和目标的私有原始字节、规范化版本、文件哈希、独立变换重算 | 重算预期与实际逐字节一致，覆盖完整 | 相同映射下实际字节不符 | 只有 observer bool/digest；字节不可读或未授权 | 强内容认证停止；不得仅比较两个自报摘要 |
| B04 clone 独立性 | storage_uid/epoch、offset/区间、别名查询依据 | 要求独立时实际不别名，同时内容正确 | 必须 clone 的目标仍共享可变区间 | 只有相等内容、对象 ID 或可能复用的 data_ptr | 不宣称缓存请求隔离已完整证明 |
| B05 安装与短命终态行 | split 后/安装后/释放前快照、pool lookup、原 ref、Slot epoch | 安装目标与已证明的 transform 目标一致 | 安装不同目标/旧版本/他人 Slot | 旧六个终态行缺少释放前后置行观察 | 空 worker_end 不作替代证据 |

## 4. 链 C：调用对应

| ID / 不变量 | 独立证据与新字段 | PASS 条件 | FAIL 条件 | UNKNOWN / 现有缺口 | 停止条件 |
| --- | --- | --- | --- | --- | --- |
| C01 共用关联键 | Runtime/model 各自入口事件、invocation_uid、直接 binding、父跨度 | 当前 profile 实际 1:1，行/版本/父边一致 | 已知错误关联或意外基数 | 旧 model call_id 与 Runtime UUID 无直接绑定 | 不按形状/时间配对 |
| C02 基数异常 | segment/fusion coverage、实际调用成员和冻结 profile | 仅预先授权且有依据的 1:多/多:1 关系可通过专属规则 | 普通 profile 出现未授权分拆/融合或重复绑定 | 关系/覆盖缺失；当前不支持不能假定未来支持 | 不新增兜底规则解释异常 |
| C03 重试/取消/无调用 | attempt_uid/retry_of、intent/effective、no_forward/logits 源 | 原操作/清理重试分开；取消生效及无 Forward 分支有直接依据 | 终态重新生成、重试冒用旧 ID、取消时提前释放 | 缺少实际分支、只有调用日志缺席 | 不把“没有日志”判为“没有执行” |
| C04 GPU 完成 | device/stream/event 与调用的显式关联、完成/错误状态 | 所有使用目标缓存的工作完成后再读/释放 | 明确仍在用却释放，或对错误 stream 的确认被冒用 | Python 返回但 GPU 未确认、多 stream 协议未知 | 不认证快照或清理安全性 |

## 5. 链 D：退役与确认

| ID / 不变量 | 独立证据与新字段 | PASS 条件 | FAIL 条件 | UNKNOWN / 现有缺口 | 停止条件 |
| --- | --- | --- | --- | --- | --- |
| D01 终态不等于退役 | 原 RequestState、terminal、Slot/ref/record、在用调用 | 两个生命周期分别真实表达 | 将终态直接标成资源已释放 | 仅终态或总 Slot 数 | 不报告完整 drain |
| D02 释放尝试与实际结果 | release UID/attempt、原异常、pool 查询与原 ref 分别观察 | 所有实际退役条件满足且无待确认步骤 | 返回成功但 Slot/ref 明确仍存，或状态矛盾 | 抛错后资源状态不明、查询失败、ref 未观察 | 保留错误/必要引用，停止准入 |
| D03 重试与幂等 | 同一原对象 UID、retirement_uid、distinct attempts、retry_of | 不重复生成、不丢身份、不重复计数，真实成功后才确认 | 身份丢失、其他请求资源被释放、提前复用 epoch | 重试未触发、证据缺失、永久失败 | 不假设最终恢复；不得有限重试后虚报成功 |
| D04 消费与 Lease | Runtime ack、Owner Future、consumer ack、Lease actual flag/归属 | 相应依赖已确认后 Lease 才释放 | Lease 早于真实退役/必要消费完成 | 只有 Future done/函数返回、消费者证据缺失 | Lease/drain 不得虚报完成 |
| D05 Worker/Owner 关闭 | owns_worker、逐请求退役、worker method result、Owner thread exit、最终各 registry 明细 | 按真实所有权关闭且确认请求排空 | 借用者关闭共享 Worker、带未确认资源宣称退出成功 | 只有 worker_end/close-return 或最终计数 | 保留 UNKNOWN，不强制清空字典或归零 |

## 6. Refinement 与整体门禁

| ID / 不变量 | 独立证据与新字段 | PASS 条件 | FAIL 条件 | UNKNOWN / 现有缺口 | 停止条件 |
| --- | --- | --- | --- | --- | --- |
| R01 范围分离 | 每项结论的 scope、投影/规则版本、原始前后状态与来源 | 局部、边界、微步、完整路径分层报告 | 用边界 PASS 冒充完整转移证明 | 旧 1,632 事件未作独立 v2 边；内部微步状态不足 | 不标任意 stutter |
| R02 真实路径对应 | plan、dispatch、call、资源、确认的显式因果图与联合状态 | 仅对实际已观察且有规则的转移给限定结论 | 已知非法转移且无模型缺口解释 | 取消/失败等未触发；模型遗漏清理状态 | 不外推所有交错或整个 Runtime |
| R03 可信基与准入 | 采集器/模型完成语义/规范化/重算器的独立审阅证据、既有门禁 | 将来单独满足各自义务；设计本身永不放行 | 以设计完成解除禁令或把历史 FAIL 覆写 | 目前全部实现与新轨迹验证 NOT_VERIFIED | 正式 GPU Benchmark 继续禁止 |

## 7. 本轮设计审阅结果（不是运行验收）

| 设计门禁 | 本轮状态 | 依据 |
| --- | --- | --- |
| 物理/允许/实际处理/可验证位置分离 | PASS（已明确规定） | 契约 §3，矩阵 A01–A05 |
| 身份/结构/内容分离 | PASS（已明确规定） | 契约 §4，矩阵 B01–B05 |
| 两套调用 ID 与异常基数分离 | PASS（已明确规定） | 契约 §5，矩阵 C01–C04 |
| 终态/释放结果/资源/消费/关闭分离 | PASS（已明确规定） | 契约 §6，矩阵 D01–D05 |
| 每项关键断言有独立来源要求 | PASS（已明确规定） | Schema 引用与矩阵证据列 |
| UNKNOWN、未触发与历史隔离 | PASS（已明确规定） | G01–G06、R01–R03 |
| 已实现并验证上述门禁 | NOT_VERIFIED（本轮禁止执行） | 测试 0、GPU 推理 0、实现修改 0 |

任何设计 PASS 都只表示要求已写明，不能成为新的 `PREFLIGHT_DECLARED_PASS`。
上述矩阵不是测试计划的自动执行授权。文档交付后停止，不进入接线或实验。
