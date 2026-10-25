# TinyLlama 原始实验与验收索引

当前入口为 [共享 v3 框架](../../docs/reproduction/SHARED_FRAMEWORK.md)。固定版本的较大长度审计及本轮验收见 [稳定版本报告](../../reports/shared-v3-stable-20261025/验收报告.md)；不将最小验证称为完整较大规模结果。

本轮固定源码完整验收为 [minimal/v3-stable-acceptance-20261025](minimal/v3-stable-acceptance-20261025/)，274/274 条、7/7 任务、退出 0，与旧 final-03 分别保存。较大长度另有 15 个逐元素相同的文本 KV 和 6 个 NLL 检查，未将它们记成 12516 条较大矩阵结果。

## 初始 v3 功能验证（2026-10-25）

[原始批次](minimal/v3-final-03/) · [完整报告](minimal/v3-final-03/report.md) · [两模型中文验收](../../reports/shared-v3-20261025/验收报告.md)。7/7 任务、274/274 正式观测完成，退出 0。以下 v2 与历史 v1 指标分别保留。

## 历史 v2 验证

验收批次为 [minimal/v2-final-check](minimal/v2-final-check/)，6/6 任务、134/134 正式观测、退出 0。该批次与 Mistral 完整小规模使用完全相同的仓库源码；[源码核对](../../docs/reproduction/shared-helper-validation-20261024.json)和[完整交付](../../reports/mistral/v2-small-final-20261024/验收报告.md)可供核查。

这是最小实际功能检查，不代替 TinyLlama 较大样本测量。原 v1 正式批次保留在 results/reproduction，其他本轮最小运行作为本机诊断记录保留，见 [批次索引](../../docs/reproduction/batch-index-20261024.json)。
