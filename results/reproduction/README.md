# 小规模实验结果索引

本索引区分已经入库的 TinyLlama 正式批次与已核验的 Mistral 批次。两次成功实验的模型、硬件和问答口径不同，分别保留原始数据，不合并统计。验收、冒烟、中断及零散调试结果保留在本机。

## 最新 Mistral 实验（已核验归档）

| 批次 | 用途与状态 | 报告 | 原始结果 |
| --- | --- | --- | --- |
| `20261024T064543Z` | Mistral-7B / RTX 4090 D；completed；退出码 0 | [中文报告](../../reports/mistral/20261024-mistral7b/小规模复现实验报告.md) · [图表 PDF](../../reports/mistral/20261024-mistral7b/图表合集.pdf) | [批次目录](../mistral/20261024T064543Z/) |

北京时间 2026-10-24 14:45:47–15:20:30，30 个自然文本条件、10 个完整问答、540 次正式 TTFT 计时。报告和图表保存在 `reports/mistral/20261024-mistral7b/`，ZIP 已按要求删除。运行方式见 [Mistral README](../../experiments/mistral/README.md)。

同日批次 `20261024T064353Z` 因编译扩展缺少 `Python.h` 失败、退出码为 1，无正式测量结果，保留在 [失败批次目录](../mistral/20261024T064353Z/)。

## 已入库的 TinyLlama 正式实验

| 批次 | 用途与状态 | 机器配置及运行记录 | 实验结果 |
| --- | --- | --- | --- |
| `20261013T060232923276Z` | `formal-small-scale`；6/6 任务完成 | [README.md](20261013T060232923276Z/README.md) | [report.md](20261013T060232923276Z/report.md) |

该批次包括 60 个续写上下文、90 个问答条件、1080 次 TTFT 测量及 1800 次自适应测量。完整执行表示预定测量已完成，不代表论文全部结论得到验证。

执行方式见 [mini_repro/README.md](../../mini_repro/README.md)，固定条件与评价口径见 [实验协议](../../mini_repro/EXPERIMENT_PROTOCOL.md)。各批次的 `manifest.json` 保留执行时的参数和命令，`metadata.json` 保留机器、模型、源码及数据指纹。

## 归档规则

- 确认用途为 `formal-small-scale`，批次和全部任务的状态均为 `completed`，按 manifest 校验样本、方法和重复次数。
- 一并保存原始 JSON、汇总 CSV、报告、manifest、metadata，以及该机器的配置和运行说明；原始记录保留执行时的内容。
- 在 `.gitignore` 中为完整批次添加目录白名单，再更新本索引。新批次、日志和本机缓存默认不提交。
- 已有历史和未完成记录仅取消 Git 跟踪，保留本机文件；不通过删除本地数据整理仓库。
