# 历史批次索引

以下记录在共享 v2 重构前已存在，原始数据、配置、日志、模型/数据指纹、报告和 source/ 快照保持原字节；没有移动文件，因此迁移映射为原路径到同一路径。

| 模型 / 批次 | 历史协议和覆盖 | 已知限制 |
| --- | --- | --- |
| TinyLlama / [20261013T060232923276Z](../../results/reproduction/20261013T060232923276Z/) | tinyllama-v1；3 组、6/6 任务；60 自然条件、90 受控 QA、1080 TTFT、1800 自适应观测 | NLL 漏首续写 token；短语包含 QA；自适应网络估算；原文本全词表投影；不能与 v2 混合 |
| Mistral / [20261024T064543Z](../../results/mistral/20261024T064543Z/) | mistral-v1；30 自然条件、10 完整 TQA、540 TTFT；无自适应 | QA 按全部输出 EM/F1，raw EM=0；混合长度 TQA 不能充作逐长度受控 QA；NLL 漏首 token；高带宽需看实测 |
| Mistral / [20261024T064353Z](../../results/mistral/20261024T064353Z/) | mistral-v1；失败，退出 1 | Python.h 缺失导致扩展编译失败，无正式测量 |

[历史 Mistral 中文报告](../../reports/mistral/20261024-mistral7b/小规模复现实验报告.md)沿用当时口径，不能当作新协议的验收。未对旧结果重评分或覆写旧主指标；后续重评分应使用独立带协议的 analysis 目录。v2 新批次写到 `results/<model>/<scale>/<run-id>`，与历史平行分批保存。首次缺权重或其他失败的新批次也保留。

旧 TinyLlama 批次只有源码指纹，没有完整 source/ 快照；旧 Mistral 快照遗漏了部分公共辅助模块。这是历史归档限制，需结合当时 Git 版本核查，不能用当前代码补写成当时已经保存的证据。新公共批次完整保存测量辅助源码。

本轮各批次的状态、用途和观测数见 [机器可读索引](batch-index-20261024.json)。`v2-small-20261024` 因缺少离线 prefill 计时而被替代，保留 interrupted、443 条观测与 superseded.json；未用于最终主统计。最终 Mistral 批次仅为 `small/v2-small-final-20261024`，TinyLlama 验收仅使用 `minimal/v2-final-check`。其他最小批次保留为环境、传输边界和真实中断恢复证据。
