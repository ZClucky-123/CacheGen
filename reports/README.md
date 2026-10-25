# 可分享的实验报告

- [共享 v3 稳定版本：大上下文修复与验收](shared-v3-stable-20261025/验收报告.md)：Mistral 469 条完整预验证、严格 KV/NLL、真实恢复和固定源码证据，附当前运行入口。

- [共享 v3：两模型功能验收](shared-v3-20261025/验收报告.md)：各 7/7 任务、274 条观测，含流水线配对、时间线、历史吞吐及消融。

- [共享 v2：Mistral 完整小规模验收](mistral/v2-small-final-20261024/验收报告.md)：6/6 任务、552 条观测，含逐例 QA、三组 PNG/SVG/PDF 与完整原始记录链接。

- [Mistral-7B / RTX 4090 D，2026-10-24](mistral/README.md)：中文报告、7 组图表和逐题问答。
- [TinyLlama / RTX 4070 Laptop，2026-10-13](../results/reproduction/20261013T060232923276Z/README.md)：历史正式批次与自动报告。

原始记录位于 `results/<model>/<scale>/<run-id>/`。上面两份 v3 报告对应不同源码和执行设置，分别解释；旧验收不重写。报告展示不替代原始状态、参数、记录和校验清单。配置与论文边界见 [框架指南](../docs/reproduction/SHARED_FRAMEWORK.md)及 [论文对照](../docs/reproduction/PAPER_ALIGNMENT.md)。
