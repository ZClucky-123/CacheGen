# Mistral 原始实验与验收索引

当前框架为 [共享 v3](../../docs/reproduction/SHARED_FRAMEWORK.md)。结果按 `minimal/`、`small/`、`larger/` 和唯一批次分层，默认状态查询覆盖这些目录。当前论文边界见 [对照表](../../docs/reproduction/PAPER_ALIGNMENT.md)。

## 较大规模完整结果（2026-10-26）

[v3-larger-stable-20261025](larger/v3-larger-stable-20261025/) 已完成 12594/12594 条正式观测、7/7 任务，退出码 0；含报告、图表、逐条原始结果及完整分卷归档，可离线进行本地分析。

## 大上下文稳定性验收

[small/v3-larger-validation-03](small/v3-larger-validation-03/) 完成 469/469 条观测、105 组配对、15 个严格文本 KV 和6个 NLL 对照，包含 TERM 及环境重建后恢复；[新验收报告](../../reports/shared-v3-stable-20261025/验收报告.md)。这是较大长度的预验证，不是 12594 条正式较大矩阵。

## 初始 v3 功能验证（2026-10-25）

[原始批次](minimal/v3-final-03/) · [完整报告](minimal/v3-final-03/report.md) · [两模型中文验收](../../reports/shared-v3-20261025/验收报告.md)。7/7 任务、274/274 正式观测完成，真实 TERM 后同字节恢复，退出 0。以下 v1/v2 小规模批次分别保留。

## 历史 v2 / v1 批次

共享 v2 正式验收：[small/v2-small-final-20261024](small/v2-small-final-20261024/)，6/6 任务、552/552 观测、退出码 0；[中文报告](../../reports/mistral/v2-small-final-20261024/验收报告.md)。

以上目录下的 20261024T064543Z / 064353Z 现标记为历史 mistral-v1 批次，原始文件不变。新共享协议批次保存到 `minimal/`、`small/`、`larger/` 下的唯一 run ID。详见 [历史口径](../../docs/reproduction/HISTORY.md)及[当前入口](../../docs/reproduction/SHARED_FRAMEWORK.md)。

| 批次 | 北京时间与状态 | 内容 |
| --- | --- | --- |
| [20261024T064543Z](20261024T064543Z/) | 2026-10-24 14:45:47–15:20:30；completed，退出码 0 | 30 个自然文本条件、10 个完整问答、540 次正式 TTFT 计时 |
| [20261024T064353Z](20261024T064353Z/) | 同日 14:43:57–14:44:17；failed，退出码 1 | 首次启动因缺少 Python.h 编译失败，无正式测量；保留诊断日志 |

成功批次的 `artifacts.json` 保存 14 项输出校验；`source/` 是运行时源码快照，`metadata.json` 保存模型、数据和环境指纹。历史绝对路径不随目录重构而改写。

[中文完整报告](../../reports/mistral/20261024-mistral7b/小规模复现实验报告.md) · [图表 PDF](../../reports/mistral/20261024-mistral7b/图表合集.pdf) · [运行方式](../../experiments/mistral/README.md)。新运行保存到此目录的唯一批次，`latest` 指向最近启动的一轮；新批次默认不加入 Git。
